#!/usr/bin/env python3
# coding=utf-8

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.models.sc_omni_attention import omni_attn_mask_vectorized

# 注入项目路径
PROJECT_ROOT = Path("/mnt/c20250607/user/wanghaoran/zxy/zxy/zxy/project/sc_showo")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.common_eval_utils import (
    SPECIAL_TOKEN_IDS,
    build_system_prompt,
    load_cells_by_ids,
    load_stage2_model,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 GeneTransformer on QA conversations")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config/config.json"))
    parser.add_argument("--ckpt-path", type=str, default = '/mnt/c20250607_hs/wanghaoran/wanghaoran/sc_showo_ckpts/stage2_v1_adamw_from7500/easy_step_200/checkpoint-step-200')
    parser.add_argument("--feature-path", type=str, default ='/mnt/c20250607/user/wanghaoran/zxy/data_and_features/zxy/features/cw_test_features/test_200', help="特征 h5ad 所在目录")
    parser.add_argument(
        "--qa-jsons",
        type=str,
        nargs="+",
        default = ['/mnt/c20250607/user/wanghaoran/zxy/data_and_features/zxy/sft_conversations/conversations/main_conversations.json'],
        help="评测用的 QA JSON 文件列表"
    )
    parser.add_argument("--output-json", type=str, default="/mnt/c20250607/user/wanghaoran/zxy/data_and_features/zxy/answers/v1_stage2_answers_7500.json")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    
    # 生成超参
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0) 
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-cells", type=int, default=None, help="最大评测细胞数")
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def build_generation_inputs(
    cell: Dict,
    turns: List[Dict],
    tokenizer,
    config: Dict,
    generated_ids: Optional[List[int]] = None
):

    dataset_cfg = config.get("dataset", {})
    max_seq_len = dataset_cfg.get("max_seq_len", 2048)
    cell_feature_tokens = dataset_cfg.get("cell_feature_tokens", 8)
    max_genes = dataset_cfg.get("max_genes", 1200)
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643
    bos_id = getattr(tokenizer, "bos_token_id", None)

    input_ids: List[int] = []
    
    if bos_id is not None:
        input_ids.append(bos_id)

    # 1. System Prompt
    system_prompt = build_system_prompt(cell)
    sys_tokens = tokenizer.encode(f"<|im_start|>system\n{system_prompt}<|im_end|>\n", add_special_tokens=False)
    input_ids.extend(sys_tokens)

    # 准备特殊 Token 序列
    rank_seq = cell["rank_seq"].tolist()
    gene_tokens = [SPECIAL_TOKEN_IDS["sog_id"]] + rank_seq + [SPECIAL_TOKEN_IDS["eog_id"]]
    cell_tokens = [SPECIAL_TOKEN_IDS["soc_id"]] + [pad_id] * cell_feature_tokens + [SPECIAL_TOKEN_IDS["eoc_id"]]

    gene_start_pos = 0
    cell_start_pos = 0
    sog_pos = 0
    eog_pos = 0
    soc_pos = 0
    eoc_pos = 0
    is_first_user = True

    # 2. 拼接多轮对话
    for idx, turn in enumerate(turns):
        role = turn["from"]
        value = turn.get("value", "")
        
        if role == "human":
            text = value.replace("<image>", "given this cell (with its gene sequence)")
            user_tokens = tokenizer.encode(f"<|im_start|>user\n{text}<|im_end|>\n", add_special_tokens=False)
            input_ids.extend(user_tokens)
            
            # 第一轮 User 之后强制注入基因和细胞信息
            if is_first_user:
                sog_pos = len(input_ids)
                input_ids.extend(gene_tokens)
                gene_start_pos = sog_pos + 1
                eog_pos = sog_pos + len(gene_tokens) - 1

                soc_pos = len(input_ids)
                input_ids.extend(cell_tokens)
                cell_start_pos = soc_pos + 1
                eoc_pos = soc_pos + len(cell_tokens) - 1
                is_first_user = False
                
        elif role == "gpt":
            prefix_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            input_ids.extend(prefix_tokens)
            
            # 如果是最后一步（我们要生成的目标步），拼接当前已生成的文本并保持开放
            if value is None:
                if generated_ids:
                    input_ids.extend(generated_ids)
            else:
                # 历史对话，正常闭合
                content_tokens = tokenizer.encode(f"{value}<|im_end|>\n", add_special_tokens=False)
                input_ids.extend(content_tokens)

    # 截断保护
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids_tensor)

    # 3. Position IDs (纯 1D，Qwen2 标准 RoPE)
    seq_len = len(input_ids_tensor)
    position_ids_tensor = torch.arange(seq_len, dtype=torch.long)

    # Inference 期间我们不需要掩码基因
    gene_mask = torch.zeros(seq_len, dtype=torch.bool)

    # 扩维以满足 batch=1
    return {
        "input_ids": input_ids_tensor.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
        "position_ids": position_ids_tensor.unsqueeze(0),
        "cell_features": torch.tensor(cell["cell_features"], dtype=torch.float32).unsqueeze(0),
        "cell_positions": torch.tensor([[cell_start_pos, cell_feature_tokens]], dtype=torch.long),
        "modality_positions": torch.tensor([[[gene_start_pos, max_genes]]], dtype=torch.long),
        "gene_mask": gene_mask.unsqueeze(0),
        "data_type": ["stage2"],
    }




def forward_with_explicit_mask(model, batch):
    # Manual forward that mirrors training path but explicitly builds omni-attn mask.
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    position_ids = batch.get("position_ids")
    cell_features = batch.get("cell_features")
    cell_positions = batch.get("cell_positions")
    modality_positions = batch.get("modality_positions")
    gene_mask = batch.get("gene_mask")

    b, seq_len = input_ids.shape
    device = input_ids.device

    input_embeds = model.showo.model.embed_tokens(input_ids)

    if attention_mask is not None:
        attention_mask = model._to_model_dtype(attention_mask)

    if cell_features is not None and cell_positions is not None:
        cell_features = model._to_model_dtype(cell_features)
        cell_embeds = model.cell_embedder(cell_features).view(
            b, model.cell_feature_tokens, model.hidden_size
        )
        new_input_embeds = []
        for i in range(b):
            start_pos = cell_positions[i, 0].item()
            length = cell_positions[i, 1].item()
            end_pos = start_pos + length
            row = torch.cat([
                input_embeds[i, :start_pos],
                cell_embeds[i],
                input_embeds[i, end_pos:],
            ], dim=0)
            new_input_embeds.append(row)
        input_embeds = torch.stack(new_input_embeds)

    if modality_positions is not None:
        new_input_embeds = []
        for i in range(b):
            g_start = modality_positions[i, 0, 0].item()
            g_len = modality_positions[i, 0, 1].item()
            if g_len > 0:
                gene_tokens = input_ids[i, g_start:g_start + g_len].clone()
                if gene_mask is not None:
                    gene_tokens[gene_mask[i][:g_len]] = model.gene_mask_idx
                else:
                    gene_tokens[gene_tokens == model.mask_gene_id] = model.gene_mask_idx
                gene_embeds = model.gene_embedder(gene_tokens)
                row = torch.cat([
                    input_embeds[i, :g_start],
                    gene_embeds,
                    input_embeds[i, g_start + g_len:],
                ], dim=0)
                new_input_embeds.append(row)
            else:
                new_input_embeds.append(input_embeds[i])
        input_embeds = torch.stack(new_input_embeds)

    combined_modalities = []
    for i in range(b):
        mods = []
        if cell_positions is not None:
            mods.append((cell_positions[i, 0].item(), cell_positions[i, 1].item()))
        if modality_positions is not None:
            g_start = modality_positions[i, 0, 0].item()
            g_len = modality_positions[i, 0, 1].item()
            if g_len > 0:
                mods.append((g_start, g_len))
        combined_modalities.append(mods)

    omni_mask_binary = omni_attn_mask_vectorized(
        B=b,
        LEN=seq_len,
        modalities=combined_modalities,
        device=device,
        inverted=False,
    ).bool()

    if attention_mask is not None:
        omni_mask_binary &= attention_mask.unsqueeze(1).unsqueeze(2).bool()

    dtype = input_embeds.dtype
    omni_mask_4d = torch.zeros((b, 1, seq_len, seq_len), dtype=dtype, device=device)
    omni_mask_4d.masked_fill_(~omni_mask_binary, torch.finfo(dtype).min)

    outputs = model.showo(
        inputs_embeds=input_embeds,
        attention_mask=omni_mask_4d,
        position_ids=position_ids,
        return_dict=True,
    )
    return outputs.logits


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """从 Logits 中采样下一个 Token"""
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0
        
        denom = sorted_probs.sum(dim=-1, keepdim=True)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        sorted_probs = sorted_probs / denom
        
        next_idx = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_indices.gather(-1, next_idx).item())

    return int(torch.multinomial(probs, num_samples=1).item())


def generate_answer(model, tokenizer, cell: Dict, turns: List[Dict], args, config: Dict) -> str:
    """自回归逐词生成回答"""
    device = torch.device(args.device)
    stop_sequences = [
        tokenizer.encode("<|im_end|>", add_special_tokens=False),
        tokenizer.encode("<|im_end|>\n", add_special_tokens=False),
    ]
    generated_ids: List[int] = []

    for step in range(args.max_new_tokens):
        batch = build_generation_inputs(cell, turns, tokenizer, config, generated_ids=generated_ids)
        
        # 将数据推到设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.no_grad():
            logits = forward_with_explicit_mask(model, batch)
            
        next_token_id = sample_next_token(logits[:, -1, :], args.temperature, args.top_p)
        generated_ids.append(next_token_id)
        
        # 判断停止符
        for stop_ids in stop_sequences:
            if stop_ids and len(generated_ids) >= len(stop_ids) and generated_ids[-len(stop_ids):] == stop_ids:
                return tokenizer.decode(generated_ids[:-len(stop_ids)], skip_special_tokens=False).strip()

    return tokenizer.decode(generated_ids, skip_special_tokens=False).strip()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("🚀 启动 Stage 2 多轮 QA 评测")
    print("="*60)

    # 1. 使用通用工具加载 Stage 2 模型
    config, tokenizer, model = load_stage2_model(args.config, args.ckpt_path, args.device, args.dtype)

    # 2. 读取所有的 QA JSON 文件
    qa_list = []
    for jp in args.qa_jsons:
        with open(jp, "r") as f:
            qa_list.extend(json.load(f))
            
    if args.max_cells is not None:
        qa_list = qa_list[:args.max_cells]

    # 3. 根据提取出来的 ID 去找对应的细胞特征
    target_ids = [str(item["id"]) for item in qa_list]
    print(f"📥 正在从特征库提取 {len(target_ids)} 个细胞特征...")
    cells_dict = load_cells_by_ids(args.feature_path, target_ids)

    # 4. 开始评测与生成
    results = []
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for item_idx, item in enumerate(qa_list):
        cell_id = str(item["id"])
        if cell_id not in cells_dict:
            continue
            
        cell = cells_dict[cell_id]
        conversations = item["conversations"]
        
        running_history = []
        question_index = 0
        
        print(f"\n🧪 [Cell ID: {cell_id}] ({item_idx+1}/{len(qa_list)})")
        
        # 遍历每一轮对话
        for turn in conversations:
            role = turn.get("from")
            value = turn.get("value", "")

            if role == "human":
                running_history.append({"from": "human", "value": value})
                current_question = value
                continue

            if role == "gpt" and running_history:
                # 把最后一步的回答置为 None，指示生成函数去填空
                turns_for_model = list(running_history)
                turns_for_model.append({"from": "gpt", "value": None})
                
                # 执行生成
                pred_answer = generate_answer(model, tokenizer, cell, turns_for_model, args, config)
                
                # 打印展示 Case
                print("-" * 50)
                print(f"🔹 Q{question_index + 1}: {current_question}")
                print(f"✅ Ref : {value}")
                print(f"🤖 Pred: {pred_answer}")
                print("-" * 50)

                results.append({
                    "cell_id": cell_id,
                    "h5ad_path": cell["h5ad_path"],
                    "question_index": question_index,
                    "question": current_question,
                    "reference_answer": value,
                    "predicted_answer": pred_answer,
                })

                # 将真实的回答（或者预测的回答）塞回历史中用于下一轮
                # 默认使用真实历史 (Teacher Forcing) 以防止多轮级联误差
                running_history.append({"from": "gpt", "value": value})
                question_index += 1

        # 定期保存
        if (item_idx + 1) % args.save_every == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # 最终保存
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(f"🎉 评测完成! 共回答了 {len(results)} 个问题。")
    print(f"💾 结果已保存至: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()