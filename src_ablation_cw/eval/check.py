import torch

ckpt = "/mnt/c20250607/user/wanghaoran/zyh/scData/outputs/cw_ablation_stage2/checkpoint-step-1000/state.pt"
state = torch.load(ckpt, map_location="cpu", weights_only=False)
model_state = state.get("model", state)

keys = list(model_state.keys())
qformer_keys = [k for k in keys if "pathway_qformer" in k]
pathway_embed_keys = [k for k in keys if "pathway_embeddings" in k]

print("num_total_keys:", len(keys))
print("num_qformer_keys:", len(qformer_keys))
print("num_pathway_embedding_keys:", len(pathway_embed_keys))
print("\nfirst_qformer_keys:")
for k in qformer_keys[:20]:
    print(" ", k)

print("\npathway_embedding_keys:")
for k in pathway_embed_keys:
    print(" ", k)