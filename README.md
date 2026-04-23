# scData Q-Former Cell-Type Training

This repository contains the current CW ablation pipeline for connecting single-cell gene-token features to a Qwen2.5 language model through a trainable Q-Former bridge.

## Current Pipeline

The active model path is:

1. For each cell, read ranked gene ids from OKR-cell feature H5AD files.
2. Keep the top `gene_input_tokens` genes from the rank list, currently `400` in the main config.
3. Map those gene ids to static gene embeddings from `okrcell_top100_static_gene_embeddings.pt`.
4. Build a gene-token tensor with shape `[400, 768]` for each cell.
5. Use `PathwayCellFeatureQFormer` to compress the gene-token sequence into `cell_feature_tokens` query tokens.
6. Project Q-Former outputs from `768` to the LLM hidden size through the trainable cell bridge.
7. Replace the cell placeholder tokens in the dialogue prompt and train/evaluate with LM loss or forced-choice PPL.

The variable name `cell_features` is still used in parts of the code, but in the current pipeline it usually means gene-token embeddings, not a single `1 x 768` cell feature vector.

## Important Files

- `src_ablation_cw/config/config_cw_ablation_cell_only.json`: main training and evaluation config.
- `src_ablation_cw/config/accelerate_cw_ablation_deepspeed.yaml`: accelerate/deepspeed launch config.
- `src_ablation_cw/train/train_stage1_cw.sh`: stage-1 launch script.
- `src_ablation_cw/train/train_stage2_cw.sh`: stage-2 launch script.
- `src_ablation_cw/train/train_stage1_cw_cell_only.py`: stage-1 trainer.
- `src_ablation_cw/train/train_stage2_cw_cell_only_lora.py`: stage-2 LoRA trainer.
- `src_ablation_cw/models/modeling_cell_transformer_for_sft_cw.py`: LLM wrapper and Q-Former bridge integration.
- `src/scgeneqformer/models/gene_qformer.py`: Q-Former implementation.
- `src_ablation_cw/datasets/expression_h5ad_registry.py`: H5AD rank reading and static gene embedding lookup.
- `src_ablation_cw/eval/eval_tabsap_forced_choice.py`: Tabula Sapiens forced-choice PPL evaluation.
- `scripts/build_okrcell_static_gene_embeddings.py`: build static gene embeddings by top-100 cell average pooling.

## Active Data Inputs

The current config expects these files on the training machine:

- `data.gene_h5ad_paths`: OKR-cell feature H5AD files with `obsm['rank']`.
- `model.static_gene_embedding_ckpt_path`: static gene embedding checkpoint.
- `training.cw_ablation.stage1_json_paths`: stage-1 dialogue JSONs.
- `training.cw_ablation.stage2_json_paths`: stage-2 dialogue JSONs.

Optional pair-data fields exist but are disabled when set to `null` or max samples are `0`.

## Q-Former Size Controls

The Q-Former is configured by:

- `dataset.cell_feature_tokens`: number of output query tokens inserted into the LLM prompt.
- `model.qformer_num_layers`: number of cross-attention blocks.
- `model.qformer_num_heads`: number of attention heads.
- `model.qformer_ffn_mult`: FFN expansion ratio. FFN width is `768 * qformer_ffn_mult`.
- `model.train_pathway_cell_qformer`: whether Q-Former and pathway embeddings are trainable.

For example, with `hidden_dim=768`, `cell_feature_tokens=80`, `qformer_num_layers=4`, `qformer_num_heads=16`, and `qformer_ffn_mult=8`, the Q-Former has about `49.66M` parameters, or about `49.72M` including `pathway_embeddings`.

## Run Training

Stage 1:

```bash
bash src_ablation_cw/train/train_stage1_cw.sh
```

Foreground mode:

```bash
bash src_ablation_cw/train/train_stage1_cw.sh fg
```

Stage 2:

```bash
bash src_ablation_cw/train/train_stage2_cw.sh
```

Check that `CUDA_VISIBLE_DEVICES` and `num_processes` in the accelerate config match. For 4 visible GPUs, use `num_processes: 4`; for 8 visible GPUs, use `num_processes: 8`.

## Run Evaluation

```bash
python src_ablation_cw/eval/eval_tabsap_forced_choice.py \
  --config src_ablation_cw/config/config_cw_ablation_cell_only.json \
  --ckpt-path /path/to/checkpoint \
  --output-dir eval_results/ablation_cw/tabsap_forced_choice_plus
```

The forced-choice evaluator writes:

- `metrics.json`: aggregate accuracy and evaluation settings.
- `predictions.csv`: per-sample predicted label and all candidate PPL values.

## Cleanup Policy

Keep source code, configs, static embeddings, checkpoints, and evaluation CSV/JSON files. Safe cleanup targets are Python caches and temporary files such as `__pycache__/`, `*.pyc`, `*.tmp`, and editor backup files.
