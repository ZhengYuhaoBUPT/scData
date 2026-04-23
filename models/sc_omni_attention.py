# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def full(b, h, q_idx, kv_idx):
    return q_idx >= 0


def modality(offset, length):
    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= offset) & (kv_idx < (offset + length))

    return mask_fn


# code is borrowed from https://github.com/lucidrains/transfusion-pytorch
def omni_attn_mask(modalities):
    modalities = modalities.long()

    def mask_mod(b, h, q_idx, kv_idx):
        mask = causal(b, h, q_idx, kv_idx)

        modality_batch = modalities[b]

        for offset, length in modality_batch:
            mask = mask | modality(offset, length)(b, h, q_idx, kv_idx)

        return mask

    return mask_mod


def omni_attn_mask_vectorized(B, LEN, modalities, device, inverted=False):
    """
    🚀 向量化版本：利用下三角掩码铺底 + 局部区块解锁，完美实现多模态注意力
    
    核心逻辑：
    1. 全局因果掩码（tril）铺底 → Gene 可以看到 Text 和 Cell
    2. 局部区块解锁 → Cell 和 Gene 内部变成双向注意力
    
    Args:
        B: Batch size
        LEN: Sequence length
        modalities: List[List[Tuple[start, len]]] 每个 batch 的模态列表
                   格式：[[(cell_start, cell_len), (gene_start, gene_len)], ...]
        device: torch device
        inverted: 是否返回 inverted mask（SDPA 需要）
    
    Returns:
        mask: (B, 1, LEN, LEN) 4D 张量
    """
    # ========== 1. 铺底：全局因果掩码（下三角） ==========
    # 这保证了：
    # - Text 区域是自回归的（Causal）
    # - Cell 区域可以看到所有 Text
    # - Gene 区域可以看到所有 Text + Cell
    tril_mask = torch.tril(
        torch.ones((LEN, LEN), dtype=torch.bool, device=device)
    )
    
    # 扩展到 Batch 维度：(B, LEN, LEN)
    mask = tril_mask.unsqueeze(0).expand(B, LEN, LEN).clone()
    
    # ========== 2. 局部解锁：将特定模态内部设为全向（Bidirectional） ==========
    # 遍历每个 batch 的每个模态区域
    for i in range(B):
        if len(modalities[i]) > 0:
            # 提取当前 batch 的所有模态起始和长度
            starts = torch.tensor([m[0] for m in modalities[i]], device=device)
            lengths = torch.tensor([m[1] for m in modalities[i]], device=device)
            
            # 批量处理：为每个模态生成全向掩码
            for start, length in zip(starts, lengths):
                if length > 0:
                    end = min(start + length, LEN)
                    # 将该模态内部的 upper-triangle 也设为 True
                    # 这样该区域就变成了完全的双向注意力
                    mask[i, start:end, start:end] = True
    
    # ========== 3. 转换为 SDPA 格式 ==========
    # (B, LEN, LEN) -> (B, 1, LEN, LEN)
    mask_4d = mask.unsqueeze(1)
    
    if inverted:
        # SDPA 格式：0.0 表示保留，极小值表示遮蔽
        dtype = torch.float32
        mask_final = torch.zeros((B, 1, LEN, LEN), dtype=dtype, device=device)
        mask_final.masked_fill_(~mask_4d, torch.finfo(dtype).min)
        return mask_final
    else:
        # 返回 0/1 bool 张量
        return mask_4d

def full_attn_mask_naive(B, LEN, device, inverted=True):
    attention_mask = torch.ones((B, 1, LEN, LEN), dtype=torch.long).to(device)
    if inverted:
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask

def causal_attn_mask_naive(B, LEN, device, inverted=True):
    attention_mask = torch.tril(torch.ones((B, 1, LEN, LEN), dtype=torch.long)).to(device)
    if inverted:
        inverted_attention_mask = 1 - attention_mask
        inverted_attention_mask = inverted_attention_mask.masked_fill(
            inverted_attention_mask.to(torch.bool), torch.iinfo(torch.long).min
        )
        return inverted_attention_mask
    else:
        return attention_mask

if __name__ == '__main__':
    device = 'cuda:0'
    # seq_len = 1024
    # modality_positions = torch.from_numpy(np.array(
    #     [[(200, 300), (0, 0), (0, 0)], [(0, 200), (800, 900), (900, 1000)], [(200, 500), (800, 1024), (0, 0)],
    #      [(200, 500), (800, 1024), (0, 0)]])).to(device)
    # omni_mask_fn = omni_attn_mask(modality_positions)
    #
    # import time
    #
    # for i in range(20):
    #     s = time.time()
    #     block_mask = create_block_mask(full, B=4, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
    #     print(block_mask)
    #     print(time.time() - s)
    #
    # print(type(block_mask) == BlockMask)

    seq_len = 20
    modality_positions = torch.from_numpy(np.array(
        [[(3, 8), (0, 0), (0, 0)], [(0, 5), (10, 15), (0, 0)]])).to(device)
    omni_mask = omni_attn_mask_naive(2, seq_len, modality_positions, device, inverted=False)
    print(omni_mask)
