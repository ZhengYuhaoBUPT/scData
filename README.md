# scDataTest

单细胞转录组 + Qwen2 的多模态训练代码。

当前仓库已经按当前使用方式调整为：
- Stage1: understanding-only 自回归训练
- Stage2: 参考 `/data/bgi/data/projects/multimodal/zyh/tmp/src` 对齐后的课程学习 + LoRA / rankloss 训练
- 支持多目录 SFT LMDB、`sft_key_cache_dir`、whitelist key-based Stage1 混入

## 目录结构

- `models/`
  - 核心模型实现
  - `modeling_gene_transformer_rank_pe.py`: Stage1 主模型
  - `modeling_gene_transformer_for_sft_rank_pe.py`: Stage2 主模型
- `datasets/`
  - `bidirectional_stage1_dataset_rankloss.py`: Stage1 paired 数据 + whitelist 包装
  - `gene_sft_dataset_no_metadata_prompt_rankloss.py`: Stage2 SFT 数据
  - `cluster_feature_loader.py`: 多目录 LMDB loader
- `train/`
  - `train_stage1_bidirectional_rank_sin_rankloss.py`: Stage1 rankloss 入口
  - `train_stage2_curriculum_lora_rankloss.py`: Stage2 rankloss 入口
  - `train_stage1.sh`, `train_stage2.sh`: 启动脚本
- `config/config.json`
  - 主配置文件
- `run/logs/`
  - 训练日志

## 当前训练设定

### Stage1

当前 Stage1 不是原始的 bidirectional 训练，而是：
- 只保留 understanding 样本
- 只优化文本 NTP loss
- rankloss 仍然保留

也就是说现在的 Stage1 更接近：
- 输入 gene 条件
- 自回归生成 answer 文本

### Stage2

当前 Stage2：
- 参考旧目录 `tmp/src` 的课程学习、whitelist、SFT key cache 和多卡索引逻辑
- 混入的 Stage1 paired 数据改为 understanding-only
- 支持 rankloss

## 配置说明

主配置文件：
- `config/config.json`

关键字段：

### 数据

- `data.cluster_lmdb_dir`
  - Stage1 cluster LMDB 根目录
- `data.lmdb_base_dir`
  - Stage1 caption / metadata LMDB 根目录
- `data.codebook_dir`
  - gene codebook 目录
- `data.scgpt_gene_vocab`
  - scGPT vocab JSON
- `data.sft_json_paths`
  - Stage2 SFT 对话 JSON 列表
- `data.stage1_whitelist_json`
  - Stage2 混入 Stage1 paired 时使用的 whitelist
- `data.sft_cluster_lmdb_dirs`
  - Stage2 SFT cluster LMDB 目录列表
- `data.sft_key_cache_dir`
  - Stage2 SFT key cache 目录
- `data.sft_use_target_id_filter`
  - 是否启用 SFT target id 过滤
- `data.sft_target_id_json_paths`
  - target id JSON 列表
- `data.sft_db_glob_pattern`
  - Stage2 SFT DB 匹配模式，当前通常是 `*.db`
- `data.sft_celltype_sources`
  - Stage2 rankloss 用于构建 celltype map 的 h5ad 来源配置

### 模型

- `model.llm_model_path`
  - Qwen2 模型路径
- `model.disable_gene_position_ids`
  - 是否关闭 gene position ids

### 训练

- `training.batch_size`
- `training.num_workers`
- `training.epochs`
- `training.stage1`
  - Stage1 loss 权重配置
- `training.stage1_rank_loss`
  - Stage1 rankloss 配置
- `training.stage2`
  - Stage2 loss 权重配置
- `training.stage2_rank_loss`
  - Stage2 rankloss 配置

### 日志与保存

- `logging.log_interval`
- `logging.save_interval`
- `logging.mm_gap_enabled`
- `logging.mm_gap_interval`
- `logging.mm_gap_max_samples`

### Checkpoint

- `checkpoint.save_dir`
- `checkpoint.stage1_weights_path`
  - Stage2 初始化时挂载的 Stage1 权重
- `checkpoint.resume_from`
  - Stage2 断点恢复路径

## 启动方式

### Stage1

推荐入口：

```bash
bash train/train_stage1.sh
```

或直接：

```bash
python train/train_stage1_bidirectional_rank_sin_rankloss.py   --config /data/bgi/data/projects/multimodal/zyh/scDataTest/config/config.json
```

### Stage2

推荐入口：

```bash
bash train/train_stage2.sh
```

或直接：

```bash
python train/train_stage2_curriculum_lora_rankloss.py   --config /data/bgi/data/projects/multimodal/zyh/scDataTest/config/config.json
```

## 可移植用法

训练脚本支持：
- `--config /path/to/config.json`
- `SC_SHOWO_CONFIG=/path/to/config.json`
- `SC_SHOWO_ROOT=/path/to/project_root`

所以迁移到其他机器时，只要保证：
- 当前仓库代码完整
- `config.json` 中的绝对路径改成新机器可用路径

即可运行。

## 训练日志

默认日志可看：

```bash
tail -f run/logs/stage1.log
tail -f run/logs/stage2.log
```

Stage2 常见现象：
- `gradient_accumulation_steps` 较大时，`global_step` 增长会很慢
- `mm_gap` 会在特定 step 额外做 probe，某些步会更慢
- `rankloss` 间隔触发时也会额外做 gather / forward

## 常见问题

### 1. `ImportError: cannot import name 'IterableDataset' from 'datasets'`

原因：
- 本地 `datasets/` 包名遮蔽了 HuggingFace `datasets`

当前仓库已在 `datasets/__init__.py` 中做兼容导出。

### 2. `unexpected keyword argument 'skip_load_data'`

原因：
- `train_stage2` 脚本和 `bidirectional_stage1_dataset_rankloss.py` 版本不同步

需要保证当前仓库是完整同步后的版本。

### 3. SFT 初始化很慢或内存高

当前 rankloss SFT 数据集已按参考版对齐，使用：
- `sft_key_cache_dir`
- rank 0 建索引
- `_shared_cell_id_to_loc.pkl`
- 类级缓存 `_shared_cell_id_to_loc`

如果仍然慢，优先检查：
- `training.num_workers`
- `logging.mm_gap_enabled`
- `training.stage2_rank_loss.interval`
- `gradient_accumulation_steps`

### 4. 训练看起来“卡住在某一步”

优先检查：
- `logging.log_interval`
- `logging.save_interval`
- `logging.mm_gap_enabled`
- `training.stage2_rank_loss.interval`
- `training.num_workers`
- `gradient_accumulation_steps`

很多时候不是死锁，而是：
- 梯度累积过大
- 某些 step 触发额外 probe / rankloss
- DataLoader worker 负担太重

### 5. Stage2 的 `mm_gap` 是什么

`mm_gap` 是一个诊断指标，不参与反传。

它会比较：
- matched gene 条件下 answer 的 NTP loss
- mismatched gene 条件下 answer 的 NTP loss

gap 越大，说明模型越依赖正确的 gene 条件。

## 当前与参考目录的关系

参考目录：
- `/data/bgi/data/projects/multimodal/zyh/tmp/src`
- `/data/bgi/data/projects/multimodal/zyh/tmp/config.json`

当前仓库已经对齐了参考版的关键部分：
- Stage2 SFT key cache / shared index 逻辑
- whitelist key-based Stage1 混入
- 多目录 SFT LMDB 支持
- `.db` / `_cluster.db` 命名兼容
- Qwen2 text embedding 访问兼容
- mm_gap 聚合回到参考版写法

保留的主要定制差异：
- Stage1 understanding-only
- Stage2 混入的 Stage1 paired 也为 understanding-only

## 建议的排查顺序

如果 Stage2 很慢或看起来卡住，建议按这个顺序排查：

1. 先看 `run/logs/stage2.log`
2. 临时设置：
   - `logging.log_interval = 1`
   - `logging.mm_gap_enabled = false`
   - `training.num_workers = 0`
3. 如果只是验证流程，临时把 `training.stage2_rank_loss.lambda_rank = 0.0`
4. 确认 `checkpoint.stage1_weights_path` 路径有效
5. 确认 `stage1_whitelist_json` 和 Stage1 LMDB 是同一套数据版本

## 备注

这个仓库当前是为“当前机器、当前数据路径、当前训练链路”整理过的一版工程，不是通用 pip 包。
如果迁移到新环境，请优先检查 `config/config.json` 中所有绝对路径。
