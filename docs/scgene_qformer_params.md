# scGeneQFormer Formal Training Params

## Core Inputs
- `--topk-json`
  Path to the per-shard top-k file. It defines which cells are used to build static gene embeddings.
- `--lmdb-path`
  LMDB shard used both for loading the top-k cells and for loading training cells.
- `--pathway-json`
  Pathway gene set definition. This defines the 4366 target genes.
- `--lmdb-vocab`
  LMDB gene-token-to-gene-name mapping.
- `--encoder-vocab-path`
  scGPT vocab used in the second-stage mapping: gene name to scGPT token id.
- `--encoder-model-path`
  Pretrained cell encoder checkpoint used to produce 768-d cell features.

## Model Shape
- `--hidden-dim`
  Hidden size for static gene embeddings and Q-Former tokens. Current mainline uses `768`.
- `--num-queries`
  Number of learnable query tokens. Current setup uses `50`.
- `--num-heads`
  Attention heads in Q-Former cross-attention.
- `--num-layers`
  Number of Q-Former cross-attention blocks.

## Data Scale
- `--max-topk-cells`
  Limits how many top-k cells are used to build static gene embeddings. Larger is better but slower.
- `--max-train-cells`
  Number of training cells loaded from the shard for the current run.

## Runtime
- `--encoder-batch-size`
  Batch size for the pretrained cell encoder step.
- `--train-batch-size`
  Batch size for Q-Former training.
- `--num-epochs`
  Number of training epochs.
- `--learning-rate`
  AdamW learning rate.

## Recommended First Formal Run
- `hidden-dim=768`
- `num-queries=50`
- `num-heads=8`
- `num-layers=2`
- `max-topk-cells=2000`
- `max-train-cells=256`
- `encoder-batch-size=16`
- `train-batch-size=16`
- `num-epochs=10`
- `learning-rate=1e-4`

This is still a moderate single-shard formal run, not the final full-scale run.

- `--max-steps`
  Optional hard cap on optimization steps. If set, training stops once this many updates have run, even if `num-epochs` is larger.
