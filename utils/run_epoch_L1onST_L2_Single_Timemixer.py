# -*- coding: utf-8 -*-
"""
Training / evaluation utilities for **single-branch TimeMixer**.
The outer interface (train_one_epoch / evaluate_one_epoch) is
identical to previous ST-GF helpers, so Train_LOSO_ST-GF_* scripts
only need to change the import path.

Save as:
    utils/run_epoch_L1onST_L2_TimeMixerSingle.py
"""

from __future__ import annotations
import datetime
import time
import torch
import swanlab
from .tools import AverageMeter, accuracy, build_tranforms2
from .RepeatedTrialAugmentation import RepeatedTrialAugmentation


# -----------------------------------------------------------------------------
# Regularisation helpers
# -----------------------------------------------------------------------------
def _compute_l2(model: torch.nn.Module, l2_lambda: float = 1e-5) -> torch.Tensor:
    l2 = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        l2 += torch.sum(p.pow(2))
    return l2_lambda * l2


def _compute_l1_graph(model: torch.nn.Module, l1_lambda: float = 1e-5) -> torch.Tensor:
    """
    For TimeMixer-Single there are **no graph parameters**.
    We check attributes defensively so the same helper also
    works for multi-branch models if re-used.
    """
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    for mod_name in (
        "ccg",
        "tsg_after",
        "tsg_before",
        "ccg_alpha",
        "ccg_beta",
        "tsg_alpha",
        "tsg_beta",
    ):
        if hasattr(model, mod_name):
            mod = getattr(model, mod_name)
            for field in ("edge_weight", "adj"):
                if hasattr(mod, field):
                    l1 += torch.sum(getattr(mod, field).abs())
    return l1_lambda * l1


# -----------------------------------------------------------------------------
# Training / Evaluation loops  (unchanged except for L1 helper)
# -----------------------------------------------------------------------------
def train_one_epoch(
    epoch,
    iterator,
    data,
    model,
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

    total_steps = data[0].shape[0] // args.batch_size + (
        1 if data[0].shape[0] % args.batch_size else 0
    )

    for step, (features, labels) in enumerate(iterator):
        features, labels = rta(features, labels)
        features, labels = features.to(device), labels.to(device)

        logits, *_ = model(features)  # 单分支只关心 logits
        ce_loss = criterion(logits, labels)
        total_loss = ce_loss + _compute_l2(model) + _compute_l1_graph(model)

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
            print(
                f"id:{args.id}  time:{elapsed}  epoch:{epoch+1}/{args.epochs}"
                f"  lr:{lr:.2e}  loss:{meters['loss'].val:.3f}/{meters['loss'].avg:.3f}"
                f"  acc:{meters['acc'].val:.3f}/{meters['acc'].avg:.3f}"
            )
            tensorboard.add_scalar("train/loss", meters["loss"].val, global_step)
            tensorboard.add_scalar("train/acc", meters["acc"].val, global_step)

    swanlab.log(
        {"train/loss": meters["loss"].avg, "train/acc": meters["acc"].avg},
        step=epoch + 1,
    )
    print(
        f"--------------------------End training at epoch:{epoch+1}--------------------------"
    )

    # 兼容旧返回：这里简单返回占位 tensor
    dummy = torch.zeros(1)
    return dummy, dummy


def evaluate_one_epoch(
    epoch, iterator, data, model, device, criterion, tensorboard, args, start_time, rta
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

            logits, *_ = model(features)
            loss = criterion(logits, labels)
            acc = accuracy(logits.detach(), labels.detach())[0]

            meters["loss"].update(loss.item(), features.size(0))
            meters["acc"].update(acc.item(), features.size(0))

    elapsed = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
    print(
        f"time:{elapsed}  epoch:{epoch+1}/{args.epochs}"
        f"  loss(avg):{meters['loss'].avg:.3f}  acc(avg):{meters['acc'].avg:.3f}"
    )
    tensorboard.add_scalar("val/loss", meters["loss"].avg, epoch)
    tensorboard.add_scalar("val/acc", meters["acc"].avg, epoch)

    swanlab.log(
        {"val/loss": meters["loss"].avg, "val/acc": meters["acc"].avg}, step=epoch + 1
    )
    print(
        f"--------------------------End evaluating at epoch:{epoch+1}--------------------------"
    )

    return meters["acc"].avg, meters["loss"].avg
