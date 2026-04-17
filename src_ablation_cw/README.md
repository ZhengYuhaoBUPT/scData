# CW Ablation (Cell-Only)

## New scripts
- `src_ablation_cw/train/train_stage1_cw_cell_only.py`
- `src_ablation_cw/train/train_stage2_cw_cell_only_lora.py`

## Core behavior
- Remove gene-token/gene-embedder/gene-loss path.
- Keep only `cell_features + dialogue text`.
- Stage1: freeze LLM backbone, train `cell_projector`.
- Stage2: freeze LLM backbone + train LoRA + `cell_projector`.

## Minimal config additions (`config/config.json`)

```json
{
  "training": {
    "cw_ablation": {
      "stage1_json_paths": [
        "/data/bgi/data/projects/multimodal/RNA_data/cellwhisper_data/sft_data/conversations/pretrain_texts.json"
      ],
      "stage2_json_paths": [
        "/data/bgi/data/projects/multimodal/RNA_data/cellwhisper_data/sft_data/conversations/finetune_conversations.json"
      ],
      "use_system_prompt": true,
      "append_image_tag": true,

      "stage1_epochs": 1,
      "stage1_lr": 1e-4,
      "stage1_max_steps": 0,
      "stage1_resume_from": null,

      "stage2_init_from_stage1": "/path/to/cw_ablation_stage1/checkpoint-step-XXXX",
      "stage2_use_lora": true,
      "lora_r": 8,
      "lora_alpha": 16,
      "lora_dropout": 0.05,
      "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
      "stage2_epochs": 1,
      "stage2_lora_lr": 2e-5,
      "stage2_cell_lr": 1e-4,
      "stage2_max_steps": 0,
      "stage2_resume_from": null
    }
  }
}
```

## Run

```bash
PYTHONPATH=/root/wanghaoran/zxy/project/sc_showo \
python /root/wanghaoran/zxy/project/sc_showo/src_ablation_cw/train/train_stage1_cw_cell_only.py \
  --config /root/wanghaoran/zxy/project/sc_showo/config/config.json

PYTHONPATH=/root/wanghaoran/zxy/project/sc_showo \
python /root/wanghaoran/zxy/project/sc_showo/src_ablation_cw/train/train_stage2_cw_cell_only_lora.py \
  --config /root/wanghaoran/zxy/project/sc_showo/config/config.json
```
