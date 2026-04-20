#!/usr/bin/env python3
# coding: utf-8

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _build_topk_rank_targets(cell_expr: torch.Tensor, topk: int) -> torch.Tensor:
    topk = min(topk, cell_expr.size(1))
    target = torch.zeros_like(cell_expr)
    if topk <= 0:
        return target
    top_idx = torch.argsort(cell_expr, dim=1, descending=True)[:, :topk]
    target.scatter_(1, top_idx, 1.0)
    return target


def _train_loop(model, context_tensor, input_tensor, target_tensor, num_epochs, batch_size, learning_rate, device, max_steps=None):
    model = model.to(device)
    if context_tensor is not None:
        context_tensor = context_tensor.to(device)
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    epoch_history = []
    step_history = []
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        steps_in_epoch = 0
        for start in range(0, input_tensor.size(0), batch_size):
            batch_input = input_tensor[start:start + batch_size]
            batch_target = target_tensor[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            _queries, recon = model(context_tensor, batch_input)
            loss = F.mse_loss(recon, batch_target)
            loss.backward()
            optimizer.step()

            global_step += 1
            step_loss = float(loss.item())
            epoch_loss += step_loss
            steps_in_epoch += 1
            step_record = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "step_in_epoch": steps_in_epoch,
                "loss": step_loss,
            }
            step_history.append(step_record)
            print(f"step={global_step} epoch={epoch + 1} step_in_epoch={steps_in_epoch} loss={step_loss:.6f}", flush=True)

            if max_steps is not None and global_step >= max_steps:
                epoch_history.append({
                    "epoch": epoch + 1,
                    "loss": epoch_loss / max(steps_in_epoch, 1),
                    "global_step": global_step,
                })
                return {
                    "history": epoch_history,
                    "step_history": step_history,
                    "global_step": global_step,
                }

        epoch_history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss / max(steps_in_epoch, 1),
            "global_step": global_step,
        })
        print(f"epoch_end epoch={epoch + 1} mean_loss={epoch_loss / max(steps_in_epoch, 1):.6f} global_step={global_step}", flush=True)

    return {
        "history": epoch_history,
        "step_history": step_history,
        "global_step": global_step,
    }


def run_reconstruction_training(model, static_gene_embeddings, cell_expr, num_epochs, batch_size, learning_rate, device, max_steps=None) -> Dict:
    return _train_loop(
        model=model,
        context_tensor=static_gene_embeddings,
        input_tensor=cell_expr,
        target_tensor=cell_expr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        max_steps=max_steps,
    )


def run_cell_feature_training(model, pathway_embeddings, cell_features, num_epochs, batch_size, learning_rate, device, max_steps=None) -> Dict:
    return _train_loop(
        model=model,
        context_tensor=pathway_embeddings,
        input_tensor=cell_features,
        target_tensor=cell_features,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        max_steps=max_steps,
    )


def run_cell_feature_training_with_rank_aux(
    model,
    pathway_embeddings,
    cell_features,
    cell_expr,
    num_epochs,
    batch_size,
    learning_rate,
    device,
    max_steps=None,
    rank_topk: int = 256,
    rank_loss_weight: float = 0.2,
) -> Dict:
    model = model.to(device)
    pathway_embeddings = pathway_embeddings.to(device)
    cell_features = cell_features.to(device)
    cell_expr = cell_expr.to(device)

    aux_head = torch.nn.Linear(model.reconstruction_head[0].in_features, cell_expr.size(1)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(aux_head.parameters()), lr=learning_rate)

    epoch_history = []
    step_history = []
    global_step = 0

    for epoch in range(num_epochs):
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_rank = 0.0
        steps_in_epoch = 0
        for start in range(0, cell_features.size(0), batch_size):
            batch_feat = cell_features[start:start + batch_size]
            batch_expr = cell_expr[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            queries, recon = model(pathway_embeddings, batch_feat)
            recon_loss = F.mse_loss(recon, batch_feat)
            rank_logits = aux_head(queries.reshape(queries.size(0), -1))
            rank_targets = _build_topk_rank_targets(batch_expr, rank_topk)
            rank_loss = F.binary_cross_entropy_with_logits(rank_logits, rank_targets)
            loss = recon_loss + rank_loss_weight * rank_loss
            loss.backward()
            optimizer.step()

            global_step += 1
            step_total = float(loss.item())
            step_recon = float(recon_loss.item())
            step_rank = float(rank_loss.item())
            epoch_total += step_total
            epoch_recon += step_recon
            epoch_rank += step_rank
            steps_in_epoch += 1
            step_record = {
                "global_step": global_step,
                "epoch": epoch + 1,
                "step_in_epoch": steps_in_epoch,
                "loss": step_total,
                "recon_loss": step_recon,
                "rank_loss": step_rank,
            }
            step_history.append(step_record)
            print(
                f"step={global_step} epoch={epoch + 1} step_in_epoch={steps_in_epoch} "
                f"loss={step_total:.6f} recon_loss={step_recon:.6f} rank_loss={step_rank:.6f}",
                flush=True,
            )

            if max_steps is not None and global_step >= max_steps:
                epoch_history.append({
                    "epoch": epoch + 1,
                    "loss": epoch_total / max(steps_in_epoch, 1),
                    "recon_loss": epoch_recon / max(steps_in_epoch, 1),
                    "rank_loss": epoch_rank / max(steps_in_epoch, 1),
                    "global_step": global_step,
                })
                return {
                    "history": epoch_history,
                    "step_history": step_history,
                    "global_step": global_step,
                    "rank_aux_head_state_dict": aux_head.state_dict(),
                }

        epoch_history.append({
            "epoch": epoch + 1,
            "loss": epoch_total / max(steps_in_epoch, 1),
            "recon_loss": epoch_recon / max(steps_in_epoch, 1),
            "rank_loss": epoch_rank / max(steps_in_epoch, 1),
            "global_step": global_step,
        })
        print(
            f"epoch_end epoch={epoch + 1} mean_loss={epoch_total / max(steps_in_epoch, 1):.6f} "
            f"mean_recon_loss={epoch_recon / max(steps_in_epoch, 1):.6f} "
            f"mean_rank_loss={epoch_rank / max(steps_in_epoch, 1):.6f} global_step={global_step}",
            flush=True,
        )

    return {
        "history": epoch_history,
        "step_history": step_history,
        "global_step": global_step,
        "rank_aux_head_state_dict": aux_head.state_dict(),
    }
