#!/usr/bin/env python3
# coding=utf-8

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from common_eval_utils import (
    SPECIAL_TOKEN_IDS,
    load_cells_by_ids,
    load_stage2_model,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 ablation model on QA conversations")
    parser.add_argument("--config", type=str, default='/mnt/c20250607/user/wanghaoran/zxy/zxy/zxy/project/sc_showo/src_ablation_cw/config/config_cw_ablation_cell_only.json')
    parser.add_argument("--ckpt-path", type=str,
                        default='/mnt/c20250607_hs/wanghaoran/wanghaoran/sc_showo_ablation_with_caption/stage2/cw_ablation_stage2/checkpoint-step-277')
    parser.add_argument("--feature-path", type=str,
                        default='/mnt/c20250607/user/wanghaoran/zxy/data_and_features/zxy/features/cw_test_features')
    parser.add_argument("--qa-jsons", type=str, nargs="+",
                        default=['/mnt/c20250607/user/wanghaoran/zxy/data_and_features/zxy/sft_conversations/conversations/tabula_sapiens_conversations.json'])
    parser.add_argument("--output-json", type=str, default="/mnt/c20250607/user/wanghaoran/zxy/zxy/zxy/project/sc_showo/eval_results/ablation_cw_with_paired/answers.json")
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")

    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5)
    
    parser.add_argument('--use-explicit-mask', action='store_true', help='Use explicit omni-attn mask (Depreciated, leave False)')
    parser.add_argument('--append-image-tag', action='store_true', default=True, help='Prepend image tag placeholder to first user turn (align with training)')
    parser.add_argument('--no-append-image-tag', action='store_false', dest='append_image_tag', help='Disable image tag prepending')

    return parser.parse_args()


def build_generation_inputs(
    cell: Dict,
    turns: List[Dict],
    tokenizer,
    config: Dict,
    generated_ids: Optional[List[int]] = None,
    append_image_tag: bool = True,
):
    dataset_cfg = config.get("dataset", {})
    max_seq_len = dataset_cfg.get("max_seq_len", 1024)
    cell_feature_tokens = dataset_cfg.get("cell_feature_tokens", 8)
    cell_feature_dim = dataset_cfg.get("cell_feature_dim", 768)
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643
    bos_id = getattr(tokenizer, "bos_token_id", None)

    input_ids: List[int] = []
    if bos_id is not None:
        input_ids.append(bos_id)

    system_prompt = "You are a helpful assistant."
    sys_tokens = tokenizer.encode(f"<|im_start|>system\n{system_prompt}<|im_end|>\n", add_special_tokens=False)
    input_ids.extend(sys_tokens)

    cell_tokens = [SPECIAL_TOKEN_IDS["soc_id"]] + [pad_id] * cell_feature_tokens + [SPECIAL_TOKEN_IDS["eoc_id"]]
    cell_start_pos = 0
    is_first_user = True

    for turn in turns:
        role = turn["from"]
        val = str(turn.get("value", ""))

        # 🚨 [关键清洗] LLaMA 垃圾格式屏蔽
        val = val.replace("[INST]", "").replace("[/INST]", "")
        val = val.replace("<<SYS>>", "").replace("<</SYS>>", "")
        val = val.replace("<s>", "").replace("</s>", "").strip()

        if role == "human":
            if is_first_user:
                val = val.replace("<image>", "").strip()
                if append_image_tag and val:
                    val = "\n" + val

            prefix_tokens = tokenizer.encode(f"<|im_start|>user\n", add_special_tokens=False)
            input_ids.extend(prefix_tokens)

            if is_first_user:
                soc_pos = len(input_ids)
                input_ids.extend(cell_tokens)
                cell_start_pos = soc_pos + 1
                is_first_user = False

            content_tokens = tokenizer.encode(f"{val}<|im_end|>\n", add_special_tokens=False)
            input_ids.extend(content_tokens)

        elif role == "gpt":
            prefix_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            input_ids.extend(prefix_tokens)
            
            if not val or val == "None":
                if generated_ids:
                    input_ids.extend(generated_ids)
            else:
                content_tokens = tokenizer.encode(f"{val}<|im_end|>\n", add_special_tokens=False)
                input_ids.extend(content_tokens)

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids_tensor, dtype=torch.bool)
    seq_len = len(input_ids_tensor)

    cell_len = cell_feature_tokens if cell_start_pos > 0 and (cell_start_pos + cell_feature_tokens) <= seq_len else 0
    if cell_len == 0:
        cell_start_pos = 0

    raw_feature = torch.tensor(cell["cell_features"], dtype=torch.float32)
    if raw_feature.shape[0] < cell_feature_dim:
        pad_size = cell_feature_dim - raw_feature.shape[0]
        raw_feature = torch.nn.functional.pad(raw_feature, (0, pad_size))
    elif raw_feature.shape[0] > cell_feature_dim:
        raw_feature = raw_feature[:cell_feature_dim]

    # 🚀 斩断 Expand，返回 1D [768]，然后增加 batch 维度 -> [1, 768]
    processed_cell_feature = raw_feature

    return {
        "input_ids": input_ids_tensor.unsqueeze(0),
        "attention_mask": attention_mask.unsqueeze(0),
        "cell_features": processed_cell_feature.unsqueeze(0), # 最终给到模型的是 [1, 768]
        "cell_positions": torch.tensor([[cell_start_pos, cell_len]], dtype=torch.long),
        "modality_positions": torch.tensor([[0, 0]], dtype=torch.long),
        "gene_mask": torch.zeros((1, 0), dtype=torch.bool),
        "data_type": "stage2",
    }

def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
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
    device = torch.device(args.device)
    # 🛑 终极刹车片：151645 = <|im_end|>, 151643 = pad
    stop_ids = [151645, 151643]
    generated_ids: List[int] = []

    for _ in range(args.max_new_tokens):
        batch = build_generation_inputs(cell, turns, tokenizer, config, generated_ids=generated_ids, append_image_tag=args.append_image_tag)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                cell_features=batch["cell_features"],
                cell_positions=batch["cell_positions"],
                modality_positions=batch.get("modality_positions"),
                labels=None,
                use_cache=True,
            )
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits

        next_token_id = sample_next_token(logits[:, -1, :], args.temperature, args.top_p)
        generated_ids.append(next_token_id)

        # 🛑 监控刹车
        if next_token_id in stop_ids:
            # 移除最后一个 stop token 再解码
            return tokenizer.decode(generated_ids[:-1], skip_special_tokens=False).strip()

    return tokenizer.decode(generated_ids, skip_special_tokens=False).strip()

def main():
    args = parse_args()
    print("\n" + "=" * 60)
    print("🚀 启动 Stage 2 ablation 多轮 QA 评测 (自回归生成)")
    print("=" * 60)

    config, tokenizer, model = load_stage2_model(args.config, args.ckpt_path, args.device, args.dtype)

    qa_list = []
    for jp in args.qa_jsons:
        with open(jp, "r") as f:
            qa_list.extend(json.load(f))

    if args.max_cells is not None:
        qa_list = qa_list[:args.max_cells]

    target_ids = [str(item["id"]) for item in qa_list]
    print(f"📥 正在提取 {len(target_ids)} 个细胞特征...")
    cells_dict = load_cells_by_ids(args.feature_path, target_ids)

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
        generated_turns = []

        for turn_idx, turn in enumerate(conversations):
            if turn["from"] == "human":
                running_history.append({"from": "human", "value": turn["value"]})
            elif turn["from"] == "gpt":
                pred = generate_answer(model, tokenizer, cell, running_history + [{"from": "gpt", "value": None}], args, config)
                generated_turns.append({
                    "turn_index": turn_idx,
                    "question": running_history[-1]["value"] if running_history else "",
                    "reference_answer": turn.get("value", ""),
                    "pred_answer": pred,
                })
                running_history.append({"from": "gpt", "value": pred})

        results.append({
            "id": cell_id,
            "generated_turns": generated_turns,
            "final_history": running_history,
        })

        if (item_idx + 1) % args.save_every == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 已保存中间结果: {item_idx + 1}/{len(qa_list)}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ 评测完成，结果保存到: {output_path}")

if __name__ == "__main__":
    main()