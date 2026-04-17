#!/usr/bin/env python3
# coding: utf-8

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _train_loop(model, context_tensor, input_tensor, target_tensor, num_epochs, batch_size, learning_rate, device, max_steps=None, aux_tensor: Optional[torch.Tensor] = None):
    model = model.to(device)
    if context_tensor is not None:
        if isinstance(context_tensor, tuple):
            context_tensor = tuple(x.to(device) for x in context_tensor)
        else:
            context_tensor = context_tensor.to(device)
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    if aux_tensor is not None:
        aux_tensor = aux_tensor.to(device)
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
            batch_aux = aux_tensor[start:start + batch_size] if aux_tensor is not None else None
            optimizer.zero_grad(set_to_none=True)
            if batch_aux is None:
                _queries, recon = model(context_tensor, batch_input)
            else:
                _queries, recon = model(context_tensor[0], context_tensor[1], batch_input, batch_aux)
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


def run_ranked_cell_feature_training(model, pathway_embeddings, static_gene_embeddings, cell_features, cell_expr, num_epochs, batch_size, learning_rate, device, max_steps=None) -> Dict:
    return _train_loop(
        model=model,
        context_tensor=(pathway_embeddings, static_gene_embeddings),
        input_tensor=cell_features,
        target_tensor=cell_features,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        max_steps=max_steps,
        aux_tensor=cell_expr,
    )
