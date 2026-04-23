#!/usr/bin/env python3
# coding=utf-8

import argparse
import json
from pathlib import Path

from common_eval_utils import (
    build_stage2_eval_sample,
    collate_eval_batch,
    compute_response_ppl,
    compute_response_ppl_explicit_mask,
    load_cells_by_ids,
    load_stage2_model,
    save_csv,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Stage-2 Tabula Sapiens forced-choice evaluation (ablation cw)')
    parser.add_argument('--config', type=str,
                        default='/mnt/c20250607/user/wanghaoran/zyh/scData/src_ablation_cw/config/config_cw_ablation_cell_only.json')
    parser.add_argument('--ckpt-path', type=str,
                        default='/mnt/c20250607/user/wanghaoran/zyh/scData/outputs/cw_ablation_stage1/checkpoint-step-1650')
    parser.add_argument('--feature-path', type=str,
                        default='/mnt/c20250607/user/wanghaoran/zyh/scData/outputs/eval_prep/tab_sap_top1200_intersection.json')
    parser.add_argument('--eval-json', type=str,
                        default='/mnt/c20250607/user/wanghaoran/zxy/data_and_features/zxy/sft_conversations/conversations/tabula_sapiens_conversations.json')
    parser.add_argument('--candidate-celltypes-json', type=str, default=None)
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/c20250607/user/wanghaoran/zyh/scData/eval_results/ablation_cw/tabsap_forced_choice_plus')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dtype', type=str, choices=['bf16', 'fp16', 'fp32'], default='bf16')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--use-system-metadata', action='store_true', help='Use metadata-rich system prompt during eval.')
    
    # [核心修复] 将坑人的 --no-explicit-mask 改为可选开启的 --use-explicit-mask，且默认为 False (与模型训练 forward 对齐)
    parser.add_argument('--use-explicit-mask', action='store_true',
                        help='Use explicit omni-attn mask path (default is False to strictly align with causal training mask)')
                        
    parser.add_argument('--question-template', type=str, default=None)
    parser.add_argument('--answer-template', type=str, default='This cell is a {celltype}')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--append-image-tag', action='store_true', default=True, help='Prepend image tag placeholder to first user turn (align with training)')
    parser.add_argument('--no-append-image-tag', action='store_false', dest='append_image_tag', help='Disable image tag prepending')
    return parser.parse_args()


def batched_ppl(model, tokenizer, device, batch_size, payloads, use_explicit_mask=False):
    values = []
    for start in range(0, len(payloads), batch_size):
        chunk = payloads[start:start + batch_size]
        # [统一 Pad 策略]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643
        batch = collate_eval_batch(chunk, pad_id)
        if use_explicit_mask:
            ppls, _ = compute_response_ppl_explicit_mask(model, batch, device)
        else:
            ppls, _ = compute_response_ppl(model, batch, device)
        values.extend(ppls.tolist())
    return values


def normalize_celltype(text: str, answer_template: str) -> str:
    text = text.strip()
    if '{celltype}' in answer_template:
        prefix, suffix = answer_template.split('{celltype}')
        prefix = prefix.strip()
        suffix = suffix.strip()
        if prefix and text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
        if suffix and text.lower().endswith(suffix.lower()):
            text = text[:-len(suffix)].strip()

    if text.lower().startswith('this cell is a '):
        text = text[len('this cell is a '):]
    elif text.lower().startswith('this cell is an '):
        text = text[len('this cell is an '):]

    return text.rstrip('.').strip()


def load_eval_items(eval_json: str, answer_template: str):
    with open(eval_json, 'r') as f:
        raw_items = json.load(f)

    if not raw_items:
        return [], []

    first_item = raw_items[0]
    if isinstance(first_item, dict) and 'conversations' in first_item:
        eval_items = []
        candidate_labels = []
        for item in raw_items:
            conversations = item['conversations']
            if len(conversations) < 2:
                continue
            question = conversations[0]['value']
            answer = conversations[-1]['value']
            label = normalize_celltype(answer, answer_template)
            eval_items.append({
                'id': str(item['id']),
                'label': label,
                'raw_answer': answer,
                'question': question,
            })
            candidate_labels.append(label)
        return eval_items, sorted(set(candidate_labels))

    return raw_items, []


def main():
    args = parse_args()
    config, tokenizer, model = load_stage2_model(args.config, args.ckpt_path, args.device, args.dtype)

    eval_items, auto_candidate_celltypes = load_eval_items(args.eval_json, args.answer_template)
    if args.max_samples is not None:
        eval_items = eval_items[:args.max_samples]

    if args.candidate_celltypes_json:
        with open(args.candidate_celltypes_json, 'r') as f:
            raw_candidates = json.load(f)
        candidate_celltypes = [normalize_celltype(str(x), args.answer_template) for x in raw_candidates]
        candidate_celltypes = sorted(set(candidate_celltypes))
    else:
        candidate_celltypes = auto_candidate_celltypes

    if not candidate_celltypes:
        raise ValueError('No candidate cell types found. Provide --candidate-celltypes-json or use a conversation JSON with assistant labels.')

    target_ids = [str(item['id']) for item in eval_items]
    cells = load_cells_by_ids(args.feature_path, target_ids, config=config)

    rows = []
    correct = 0
    for idx, item in enumerate(eval_items):
        cell = cells[str(item['id'])]
        true_label = item['label']
        question = item.get('question') or args.question_template
        if question is None:
            raise ValueError('No question provided. Pass --question-template or use a conversation-style eval JSON.')
        conversations = [
            {'from': 'human', 'value': question},
            {'from': 'gpt', 'value': ''},
        ]

        payloads = []
        for celltype in candidate_celltypes:
            answer = args.answer_template.format(celltype=celltype)
            payloads.append(build_stage2_eval_sample(cell, conversations, tokenizer, config, last_answer_override=answer, include_system_metadata=args.use_system_metadata, append_image_tag=args.append_image_tag))

        ppls = batched_ppl(model, tokenizer, args.device, args.batch_size, payloads, use_explicit_mask=args.use_explicit_mask)
        best_idx = min(range(len(ppls)), key=lambda i: ppls[i])
        pred_label = candidate_celltypes[best_idx]
        correct += int(pred_label == true_label)

        rows.append({
            'sample_index': idx,
            'cell_id': str(item['id']),
            'true_label': true_label,
            'pred_label': pred_label,
            'pred_ppl': ppls[best_idx],
            'all_ppl_json': json.dumps(dict(zip(candidate_celltypes, ppls)), ensure_ascii=False),
        })

    metrics = {
        'num_samples': len(rows),
        'num_candidates': len(candidate_celltypes),
        'accuracy': correct / max(len(rows), 1),
        'eval_target': 'stage2_ablation_cw',
        'use_explicit_mask': args.use_explicit_mask,
        'use_system_metadata': args.use_system_metadata,
        'question_template': args.question_template,
        'answer_template': args.answer_template,
        'candidate_source': 'candidate_celltypes_json' if args.candidate_celltypes_json else 'eval_json_labels',
        'notes': 'CW-style forced-choice evaluation for cell-only ablation.',
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(output_dir / 'metrics.json'), metrics)
    save_csv(
        str(output_dir / 'predictions.csv'),
        rows,
        fieldnames=['sample_index', 'cell_id', 'true_label', 'pred_label', 'pred_ppl', 'all_ppl_json'],
    )
    print(metrics)


if __name__ == '__main__':
    main()