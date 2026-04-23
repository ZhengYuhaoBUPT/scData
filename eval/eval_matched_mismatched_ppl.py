#!/usr/bin/env python3
# coding=utf-8

import argparse
import copy
import math
import random
import statistics
import re
from pathlib import Path

from common_eval_utils import (
    build_stage2_eval_sample,
    collate_eval_batch,
    compute_response_ppl,
    compute_response_ppl_explicit_mask,
    extract_last_assistant_answer,
    load_cells_by_ids,
    load_conversations,
    load_stage2_model,
    save_csv,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 matched vs mismatched PPL evaluation")
    parser.add_argument('--config', type=str, default='/mnt/c20250607/user/wanghaoran/zxy/zxy/zxy/project/sc_showo/config/config.json')
    parser.add_argument('--ckpt-path', type=str, default='/mnt/c20250607_hs/wanghaoran/wanghaoran/sc_showo_ckpts/stage2_v1_adamw_from7500_only_sft_plus/easy_step_100/checkpoint-step-100')
    parser.add_argument('--feature-path', type=str, default='/mnt/c20250607/user/wanghaoran/zxy/data_and_features/zxy/features/cw_test_features/test_200')
    parser.add_argument('--qa-jsons', type=str, nargs='+', default=['/mnt/c20250607/user/wanghaoran/zxy/data_and_features/zxy/sft_conversations/conversations/main_conversations.json'])
    parser.add_argument('--output-dir', type=str, default='/mnt/c20250607/user/wanghaoran/zxy/zxy/zxy/project/sc_showo/eval_results/ppl_eval/ex_sft_100')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dtype', type=str, choices=['bf16', 'fp16', 'fp32'], default='bf16')
    parser.add_argument('--num-negatives', type=int, default=30)
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument(
        '--stratify-negatives-by-id-kind',
        action='store_true',
        help='Sample mismatched negatives only from the same ID kind (e.g., SRX vs non-SRX), without replacement.',
    )
    parser.add_argument('--use-system-metadata', action='store_true', default = False ,help='Use metadata-rich system prompt during eval (default: disabled).')
    parser.add_argument('--eval-scope', type=str, choices=['easy', 'full'], default='easy', help='easy: only easy QA turns; full: all QA turns')
    parser.add_argument(
        '--allow-negative-replacement',
        action='store_true',
        help='When negative pool is smaller than num_negatives, sample with replacement instead of raising error.',
    )
    parser.add_argument('--use-explicit-mask', action='store_true', help='Use explicit omni-attn mask path (show-o2 style)')
    parser.add_argument('--no-explicit-mask', action='store_true', help='Disable explicit mask path (use model forward)')
    parser.add_argument(
        '--mismatch-mode',
        type=str,
        choices=['cell_only', 'gene_only', 'both'],
        default='both',
        help='cell_only: only swap cell_features; gene_only: only swap rank_seq; both: swap both.',
    )
    parser.add_argument(
        '--avoid-same-answer',
        action='store_true',
        help='If possible, sample negatives with different reference answers.',
    )
    parser.set_defaults(avoid_same_answer=True)
    parser.set_defaults(allow_negative_replacement=False)
    return parser.parse_args()


def quantile(matched, mismatched):
    return sum(matched < x for x in mismatched) / max(len(mismatched), 1)


def batched_ppl(model, tokenizer, device, batch_size, payloads, use_explicit_mask=True):
    out = []
    for start in range(0, len(payloads), batch_size):
        chunk = payloads[start:start + batch_size]
        batch = collate_eval_batch(chunk, tokenizer.pad_token_id)
        if use_explicit_mask:
            chunk_ppl, _ = compute_response_ppl_explicit_mask(model, batch, device)
        else:
            chunk_ppl, _ = compute_response_ppl(model, batch, device)
        out.extend(chunk_ppl.tolist())
    return out


def build_mismatch_cell(true_cell, neg_cell, mode: str):
    mixed = copy.deepcopy(true_cell)
    if mode in ('cell_only', 'both'):
        mixed['cell_features'] = neg_cell['cell_features']
    if mode in ('gene_only', 'both'):
        mixed['rank_seq'] = neg_cell['rank_seq']
        mixed['log1p'] = neg_cell.get('log1p', mixed.get('log1p'))
    return mixed



def id_kind(cell_id: str) -> str:
    """
    Heuristic split for cw test IDs:
    - SRX*: bulk-like IDs
    - others: single-cell style IDs
    """
    cid = str(cell_id)
    return 'srx' if cid.startswith('SRX') else 'non_srx'

def is_easy_qa(question: str, answer: str) -> bool:
    q = str(question).strip().lower()
    a = str(answer).strip().lower()
    if not q or not a:
        return False

    q_words = re.findall(r"\w+", q)
    a_words = re.findall(r"\w+", a)
    total_words = len(q_words) + len(a_words)
    if len(q_words) > 30:
        return False
    if total_words > 180:
        return False

    starts_ok = q.startswith(("what", "which", "is this", "identify", "name", "describe", "tell me"))
    has_question_mark = "?" in q
    if not (starts_ok or has_question_mark):
        return False

    has_cell = bool(re.search(r"\bcells?\b", q))
    has_type_intent = bool(re.search(r"\b(type|kind|identify|classification|classify|describe|tell)\b", q))

    easy_signals = [
        "what type",
        "what kind",
        "cell type",
        "what cells",
        "which cell type",
        "identify",
        "describe the cell",
        "describe these cells",
        "what is the cell",
        "tell me about the cell type",
        "what can you tell me about the cell types",
    ]
    if not (any(sig in q for sig in easy_signals) or (has_cell and has_type_intent)):
        return False

    complex_keywords = [
        "pathway",
        "pathways",
        "function",
        "functions",
        "mechanism",
        "mechanisms",
        "regulation",
        "signal transduction",
        "gene ontology",
        "go term",
        "enrichment",
        "metabolic",
        "immune response",
        "differential",
        "trajectory",
    ]
    if any(k in q for k in complex_keywords):
        return False

    return True


def collect_eval_cases(all_items, easy_only: bool = True):
    cases = []
    for item in all_items:
        cell_id = str(item.get('id'))
        conv = item.get('conversations', [])
        if not cell_id or not conv:
            continue

        for i in range(len(conv) - 1):
            if conv[i].get('from') != 'human' or conv[i + 1].get('from') != 'gpt':
                continue
            q = conv[i].get('value', '')
            a = conv[i + 1].get('value', '')
            if easy_only and (not is_easy_qa(q, a)):
                continue

            # Keep all history up to this QA pair; score the last assistant answer in this truncated context.
            truncated_conv = [dict(t) for t in conv[: i + 2]]
            cases.append({
                'id': cell_id,
                'conversations': truncated_conv,
                'question_turn_index': i,
                'question': q,
                'reference_answer': a,
            })
    return cases



def main():
    args = parse_args()
    # default to explicit mask for show-o2 style evaluation
    use_explicit_mask = not getattr(args, 'no_explicit_mask', False)
    random.seed(args.seed)

    config, tokenizer, model = load_stage2_model(args.config, args.ckpt_path, args.device, args.dtype)
    all_items = load_conversations(args.qa_jsons)
    all_cases = collect_eval_cases(all_items, easy_only=False)
    easy_cases = collect_eval_cases(all_items, easy_only=True)

    if args.eval_scope == 'easy':
        selected_cases = easy_cases
        if not selected_cases:
            raise ValueError('No EASY QA turns matched the heuristic filter.')
    else:
        selected_cases = all_cases
        if not selected_cases:
            raise ValueError('No QA turns found in qa-jsons.')

    if args.max_samples is not None:
        k = min(args.max_samples, len(selected_cases))
        items = random.sample(selected_cases, k)
    else:
        items = selected_cases

    target_ids = [str(item['id']) for item in items]

    pool_ids = list(
        dict.fromkeys(
            str(item.get('id'))
            for item in all_items
            if item.get('id') is not None
        )
    )
    if not pool_ids:
        raise ValueError('No IDs found in qa-jsons for negative sampling pool.')

    required_ids = list(dict.fromkeys(pool_ids + target_ids))
    cells = load_cells_by_ids(args.feature_path, required_ids)

    id_to_answers = {}
    for item in all_items:
        cid = str(item.get('id'))
        conv = item.get('conversations', [])
        if not cid or not conv:
            continue
        answers = id_to_answers.setdefault(cid, set())
        for t in conv:
            if t.get('from') == 'gpt':
                val = str(t.get('value', '')).strip()
                if val:
                    answers.add(val)

    print(f'[INFO] eval_scope={args.eval_scope}, eval samples={len(items)}, easy cases total={len(easy_cases)}, all cases total={len(all_cases)}, negative pool ids={len(pool_ids)}, loaded cells={len(cells)}')

    per_sample_rows = []
    quantiles = []
    top1_hits = []
    matched_values = []
    mismatched_values = []

    for item_idx, item in enumerate(items):
        cell_id = str(item['id'])
        true_cell = cells[cell_id]
        conversations = item['conversations']
        true_answer = extract_last_assistant_answer(conversations)

        matched_payload = [build_stage2_eval_sample(true_cell, conversations, tokenizer, config, include_system_metadata=args.use_system_metadata)]
        matched_ppl = batched_ppl(model, tokenizer, args.device, 1, matched_payload, use_explicit_mask=use_explicit_mask)[0]

        candidate_ids = [cid for cid in pool_ids if cid in cells and cid != cell_id]
        if args.stratify_negatives_by_id_kind:
            true_kind = id_kind(cell_id)
            candidate_ids = [cid for cid in candidate_ids if id_kind(cid) == true_kind]
        if args.avoid_same_answer:
            filtered = [cid for cid in candidate_ids if true_answer not in id_to_answers.get(cid, set())]
            if len(filtered) >= args.num_negatives:
                candidate_ids = filtered

        if len(candidate_ids) < args.num_negatives:
            # stratified mode is strict no-replacement by design
            if args.stratify_negatives_by_id_kind:
                raise ValueError(
                    f'Not enough same-kind negatives (no replacement): '
                    f'cell_id={cell_id}, kind={id_kind(cell_id)}, pool={len(candidate_ids)}, '
                    f'required={args.num_negatives}. Reduce num-negatives or disable stratification.'
                )
            if not args.allow_negative_replacement or len(candidate_ids) == 0:
                raise ValueError(
                    f'Not enough mismatched cells for requested num_negatives: '
                    f'pool={len(candidate_ids)}, required={args.num_negatives}, cell_id={cell_id}'
                )
            print(
                f'[WARN] negative pool too small for cell_id={cell_id}: '
                f'pool={len(candidate_ids)} < num_negatives={args.num_negatives}; '
                f'fallback to sampling with replacement.'
            )
            mismatch_ids = random.choices(candidate_ids, k=args.num_negatives)
        else:
            mismatch_ids = random.sample(candidate_ids, args.num_negatives)
        mismatch_payloads = []
        for mismatch_id in mismatch_ids:
            neg_cell = cells[mismatch_id]
            mixed_cell = build_mismatch_cell(true_cell, neg_cell, args.mismatch_mode)
            mismatch_payloads.append(build_stage2_eval_sample(mixed_cell, conversations, tokenizer, config, include_system_metadata=args.use_system_metadata))

        mismatch_ppls = batched_ppl(model, tokenizer, args.device, args.batch_size, mismatch_payloads, use_explicit_mask=use_explicit_mask)

        q = quantile(matched_ppl, mismatch_ppls)
        quantiles.append(q)
        top1_hits.append(1 if matched_ppl < min(mismatch_ppls) else 0)
        matched_values.append(matched_ppl)
        mismatched_values.extend(mismatch_ppls)

        per_sample_rows.append({
            'sample_index': item_idx,
            'cell_id': cell_id,
            'question_turn_index': item.get('question_turn_index', -1),
            'question': item.get('question', ''),
            'matched_ppl': matched_ppl,
            'mismatched_mean_ppl': statistics.mean(mismatch_ppls),
            'mismatched_std_ppl': statistics.pstdev(mismatch_ppls) if len(mismatch_ppls) > 1 else 0.0,
            'quantile': q,
            'top1_match': int(matched_ppl < min(mismatch_ppls)),
            'reference_answer': true_answer,
            'num_negatives': len(mismatch_ppls),
            'mismatch_mode': args.mismatch_mode,
        'use_explicit_mask': use_explicit_mask,
        'use_system_metadata': args.use_system_metadata,
        'stratify_negatives_by_id_kind': args.stratify_negatives_by_id_kind,
        })

    matched_mean = statistics.mean(matched_values)
    mismatched_mean = statistics.mean(mismatched_values)
    metrics = {
        'num_samples': len(per_sample_rows),
        'num_all_items': len(all_items),
        'num_easy_cases': len(easy_cases),
        'num_all_cases': len(all_cases),
        'eval_scope': args.eval_scope,
        'num_negative_pool_ids': len(pool_ids),
        'num_negatives': args.num_negatives,
        'mean_quantile': statistics.mean(quantiles) if quantiles else 0.0,
        'top1_match_rate': statistics.mean(top1_hits) if top1_hits else 0.0,
        'matched_mean_ppl': matched_mean,
        'mismatched_mean_ppl': mismatched_mean,
        'log_perplexity_ratio': math.log(matched_mean / mismatched_mean) if matched_mean > 0 and mismatched_mean > 0 else None,
        'eval_target': 'stage2',
        'mismatch_mode': args.mismatch_mode,
        'use_explicit_mask': use_explicit_mask,
        'use_system_metadata': args.use_system_metadata,
        'stratify_negatives_by_id_kind': args.stratify_negatives_by_id_kind,
        'avoid_same_answer': args.avoid_same_answer,
        'notes': 'Matched keeps true cell+QA; mismatched keeps QA fixed and swaps selected conditioning modality only.',
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(output_dir / 'metrics.json'), metrics)
    save_csv(
        str(output_dir / 'per_sample.csv'),
        per_sample_rows,
        fieldnames=[
            'sample_index', 'cell_id', 'question_turn_index', 'question', 'matched_ppl', 'mismatched_mean_ppl',
            'mismatched_std_ppl', 'quantile', 'top1_match', 'reference_answer',
            'num_negatives', 'mismatch_mode', 'use_explicit_mask', 'use_system_metadata',
            'stratify_negatives_by_id_kind'
        ],
    )
    print(metrics)


if __name__ == '__main__':
    main()
