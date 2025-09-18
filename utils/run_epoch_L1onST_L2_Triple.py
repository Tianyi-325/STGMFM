# -*- coding: utf-8 -*-
"""Training / evaluation utilities (L1‑on‑Graphs + L2‑on‑All)
================================================================
**Updated for the new *triple‑branch* `STGENET`** (early‑fusion / late‑fusion
variants both compatible).

Key‑points
----------
* **I/O 不变** – 网络依旧返回 `logits, ccg_w_A, tsg_A, tsg_B, ccg_w_B`，原训练
  管线无须改动。
* **L2 正则** → 全参数平方和（unchanged）。
* **L1 正则** → 现包含所有 *graph 参数*：
  * `model.ccg.edge_weight`, `model.tsg_after.adj`, `model.tsg_before.adj`  (原)
  * `model.ccg_alpha.edge_weight`, `model.tsg_alpha.adj`,
    `model.ccg_beta.edge_weight`, `model.tsg_beta.adj`  (新增频带分支)
  在代码实现上使用 `hasattr` 保障向后兼容，若对应模块不存在则自动跳过。

其余代码保持不变，仅修订了文档与 `_compute_l1_graph`。
"""
from __future__ import annotations

import datetime
import time
import torch
import swanlab
from .tools import AverageMeter, accuracy

# -----------------------------------------------------------------------------
# Regularisation helpers
# -----------------------------------------------------------------------------


def _compute_l2(model: torch.nn.Module, l2_lambda: float = 1e-5) -> torch.Tensor:
    """L2 (weight‑decay style) on *all* parameters."""
    l2 = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        l2 += torch.sum(p.pow(2))
    return l2_lambda * l2


def _graph_terms(model: torch.nn.Module):
    """Yield all graph‑related parameters present in *any* STGENET variant."""
    for attr in [
        # 原双分支
        ("ccg", "edge_weight"),
        ("tsg_after", "adj"),
        ("tsg_before", "adj"),
        # 频带分支（若存在）
        ("ccg_alpha", "edge_weight"),
        ("ccg_beta", "edge_weight"),
        ("tsg_alpha", "adj"),
        ("tsg_beta", "adj"),
    ]:
        module_name, param_name = attr
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            if hasattr(module, param_name):
                yield getattr(module, param_name)


def _compute_l1_graph(model: torch.nn.Module, l1_lambda: float = 1e-5) -> torch.Tensor:
    """L1 sparsity penalty over *all* learnable graph parameters."""
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in _graph_terms(model):
        l1 += torch.sum(p.abs())
    return l1_lambda * l1


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train_one_epoch(
    epoch: int,
    iterator,
    data,
    model: torch.nn.Module,
    device,
    optimizer,
    criterion,
    tensorboard,
    start_time,
    args,
    rta,
):
    print(
        f"--------------------------Start training at epoch:{epoch+1}--------------------------"
    )

    model.to(device)
    criterion = criterion.to(device)

    meters = {"loss": AverageMeter(), "acc": AverageMeter()}
    model.train()

    total_steps = (
        data[0].shape[0] // args.batch_size + 1
        if data[0].shape[0] % args.batch_size
        else data[0].shape[0] // args.batch_size
    )

    for step, (features, labels) in enumerate(iterator):
        # ---- data aug ----
        features, labels = rta(features, labels)
        features, labels = features.to(device), labels.to(device)

        # ---- forward ----
        logits, ccg_w_A, tsg_A, *_ = model(features)

        # ---- losses ----
        ce_loss = criterion(logits, labels)
        l2_reg = _compute_l2(model)
        l1_reg = _compute_l1_graph(model)
        total_loss = ce_loss + l2_reg + l1_reg

        # ---- metrics ----
        acc = accuracy(logits.detach(), labels.detach())[0]
        meters["loss"].update(total_loss.item(), features.size(0))
        meters["acc"].update(acc.item(), features.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ---- console & tb ----
        global_step = epoch * total_steps + step + 1
        if global_step % args.print_freq == 0:
            elapsed = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
            lr = optimizer.param_groups[0]["lr"]
            info = (
                f"id:{args.id}  time:{elapsed}  epoch:{epoch+1}/{args.epochs}  lr:{lr:.2e}  "
                f"loss(val/avg):{meters['loss'].val:.3f}/{meters['loss'].avg:.3f}  "
                f"acc(val/avg):{meters['acc'].val:.3f}/{meters['acc'].avg:.3f}"
            )
            print(info)
            tensorboard.add_scalar("train/loss", meters["loss"].val, global_step)
            tensorboard.add_scalar("train/acc", meters["acc"].val, global_step)

    # ---- epoch summary ----
    swanlab.log(
        {"train/loss": meters["loss"].avg, "train/acc": meters["acc"].avg},
        step=epoch + 1,
    )
    print(
        f"--------------------------End training at epoch:{epoch+1}--------------------------"
    )

    # 兼容旧管线：返回 Branch‑A 的图权重
    node_weights = tsg_A.detach().cpu()
    space_node_weights = ccg_w_A.detach().cpu()
    return node_weights, space_node_weights


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------


def evaluate_one_epoch(
    epoch: int,
    iterator,
    data,
    model: torch.nn.Module,
    device,
    criterion,
    tensorboard,
    args,
    start_time,
    rta,
):
    print(
        f"--------------------------Start evaluating at epoch:{epoch+1}--------------------------"
    )

    model.to(device)
    criterion = criterion.to(device)

    meters = {"loss": AverageMeter(), "acc": AverageMeter()}
    model.eval()

    with torch.no_grad():
        for features, labels in iterator:
            features, labels = rta(features, labels)
            features, labels = features.to(device), labels.to(device)

            logits, *_ = model(features)  # predictions only
            loss = criterion(logits, labels)
            acc = accuracy(logits.detach(), labels.detach())[0]

            meters["loss"].update(loss.item(), features.size(0))
            meters["acc"].update(acc.item(), features.size(0))

    elapsed = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
    info = (
        f"time:{elapsed}  epoch:{epoch+1}/{args.epochs}  "
        f"loss(avg):{meters['loss'].avg:.3f}  acc(avg):{meters['acc'].avg:.3f}"
    )
    print(info)
    tensorboard.add_scalar("val/loss", meters["loss"].avg, epoch)
    tensorboard.add_scalar("val/acc", meters["acc"].avg, epoch)

    swanlab.log(
        {"val/loss": meters["loss"].avg, "val/acc": meters["acc"].avg}, step=epoch + 1
    )
    print(
        f"--------------------------End evaluating at epoch:{epoch+1}--------------------------"
    )

    return meters["acc"].avg, meters["loss"].avg
