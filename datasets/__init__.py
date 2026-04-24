# Compatibility: Accelerate may import `datasets.IterableDataset` to detect
# HuggingFace datasets. This local package is also named `datasets`, so expose
# the torch base class to avoid shadowing-related import failures.
from torch.utils.data import IterableDataset

# Datasets are imported directly from their respective modules
# to avoid circular import issues

# Available datasets:
# - from src.datasets.bidirectional_stage1_dataset import BidirectionalStage1Dataset
# - from src.datasets.gene_sft_dataset import SFTDataset
# - from src.datasets.mixed_dataloader import create_mixed_dataloader
