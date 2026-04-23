# src_ablation_cw

This directory contains the active CW ablation training and evaluation code. The current implementation no longer uses a single cell feature vector as the main Q-Former input. It uses ranked gene-token embeddings for each cell.

## Data Flow

1. Dialogue JSON provides the sample id and conversation.
2. `ExpressionH5ADRegistry` finds the matching cell in configured H5AD files.
3. The cell's `obsm['rank']` gene ids are read and truncated to `dataset.gene_input_tokens`.
4. Gene ids are mapped to rows in the static gene embedding checkpoint.
5. The resulting `[gene_input_tokens, 768]` tensor is passed as `cell_features` for compatibility with older code names.
6. `PathwayCellFeatureQFormer` compresses it into `dataset.cell_feature_tokens` tokens.
7. `cell_embedder` projects Q-Former outputs into the LLM hidden size and inserts them into the text sequence.

## Main Components

- `config/config_cw_ablation_cell_only.json`: main config.
- `datasets/cw_sft_cell_only_dataset.py`: dialogue dataset for stage 1 and stage 2.
- `datasets/expression_h5ad_registry.py`: H5AD rank lookup and static gene embedding construction.
- `models/modeling_cell_transformer_for_sft_cw.py`: Qwen wrapper, Q-Former bridge, and token insertion.
- `train/train_stage1_cw_cell_only.py`: stage-1 trainer.
- `train/train_stage2_cw_cell_only_lora.py`: stage-2 LoRA trainer.
- `eval/common_eval_utils.py`: shared checkpoint loading and PPL helpers.
- `eval/eval_tabsap_forced_choice.py`: forced-choice cell-type evaluation.

## Current Q-Former Notes

The Q-Former is enabled when:

```json
"use_pathway_cell_qformer": true,
"train_pathway_cell_qformer": true
```

The main size knobs are:

```json
"cell_feature_tokens": 80,
"qformer_num_layers": 4,
"qformer_num_heads": 16,
"qformer_ffn_mult": 8
```

`qformer_ffn_mult` controls the FFN hidden width inside each cross-attention block. With hidden size `768`, `qformer_ffn_mult=8` gives an FFN of `768 -> 6144 -> 768`.

## Stage Data

Stage 1 uses `training.cw_ablation.stage1_json_paths` plus the shared H5AD rank files and static gene embeddings.

Stage 2 uses `training.cw_ablation.stage2_json_paths` plus the same shared H5AD rank files and static gene embeddings. If `stage2_use_lora=true`, the LLM is wrapped with LoRA and the Q-Former bridge remains trainable according to config.

## Evaluation Output Interpretation

`predictions.csv` contains an `all_ppl_json` column. Lower PPL is better. If accuracy is low, inspect:

- top-1 predicted label distribution
- true-label rank distribution
- top1-top2 PPL margin
- common confusion pairs

This helps distinguish single-class collapse, few-class collapse, and genuinely close candidate competition.
