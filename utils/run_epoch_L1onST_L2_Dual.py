import torch
import datetime
import time
from .tools import AverageMeter, accuracy
import swanlab

"""
Updated training / evaluation loops for the **dual‑branch STGENet** (models.STGENet_Dual).

Key changes
------------
1. **Model outputs** – STGENet_Dual returns:
   ``logits, ccg_w, tsg_a, tsg_b, ccg_w_b``
   where ``ccg_w`` / ``ccg_w_b`` are the learnable spatial‑graph weights for the two
   passes through the shared ``SpatialGraph`` module, and ``tsg_a`` / ``tsg_b`` are the
   learnable time‑graph adjacency matrices for the two ``TimeGraph`` modules.

2. **Regularisation** –
   * **L2** (weight‑decay‑style) regularisation is applied to **all** trainable
     parameters (unchanged).
   * **L1** sparsity regularisation is now applied to **all graph parameters**:
       - ``model.ccg.edge_weight``
       - ``model.tsg_after.adj``
       - ``model.tsg_before.adj``

3. **Return values** – to stay compatible with the surrounding training script
   (`Train_LOSO_ST-GF_kfolder.py`) we still return two tensors:
   ``node_weights`` → the *time* graph weights (`tsg_a` from branch‑A)
   ``space_node_weights`` → the *spatial* graph weights (`ccg_w`)
"""


def _compute_l2(model, l2_lambda: float = 1e-5):
    """Sum of squared weights over all parameters."""
    l2 = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        l2 += torch.sum(p.pow(2))
    return l2_lambda * l2


def _compute_l1_graph(model, l1_lambda: float = 1e-5):
    """L1 on graph parameters only."""
    l1 = (
        torch.sum(model.ccg.edge_weight.abs())
        + torch.sum(model.tsg_after.adj.abs())
        + torch.sum(model.tsg_before.adj.abs())
    )
    return l1_lambda * l1


# -----------------------------------------------------------------------------
# Training
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
        "--------------------------Start training at epoch:{}--------------------------".format(
            epoch + 1
        )
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
        # Data augmentation (repeated‑trial, cut‑mix, etc.)
        features, labels = rta(features, labels)
        features, labels = features.to(device), labels.to(device)

        # ---------------- forward ----------------
        logits, ccg_w, tsg_a, tsg_b, ccg_w_b = model(features)

        # Cross‑entropy loss
        ce_loss = criterion(logits, labels)

        # Regularisation terms
        l2_reg = _compute_l2(model)
        l1_reg = _compute_l1_graph(model)
        total_loss = ce_loss + l2_reg + l1_reg

        # Metrics & logging
        acc = accuracy(logits.detach(), labels.detach())[0]
        meters["loss"].update(total_loss.item(), features.size(0))
        meters["acc"].update(acc.item(), features.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ---- console & tensorboard ----
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

    # swanlab logging
    swanlab.log(
        {"train/loss": meters["loss"].avg, "train/acc": meters["acc"].avg},
        step=epoch + 1,
    )
    print(
        "--------------------------End training at epoch:{}--------------------------".format(
            epoch + 1
        )
    )

    # For compatibility with existing pipeline
    node_weights = tsg_a.detach().cpu()
    space_node_weights = ccg_w.detach().cpu()
    return node_weights, space_node_weights


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate_one_epoch(
    epoch,
    iterator,
    data,
    model,
    device,
    criterion,
    tensorboard,
    args,
    start_time,
    rta,
):
    print(
        "--------------------------Start evaluating at epoch:{}--------------------------".format(
            epoch + 1
        )
    )
    model.to(device)
    criterion = criterion.to(device)

    meters = {"loss": AverageMeter(), "acc": AverageMeter()}
    model.eval()

    with torch.no_grad():
        for features, labels in iterator:
            features, labels = rta(features, labels)
            features, labels = features.to(device), labels.to(device)

            logits, *_ = model(features)  # only need predictions
            loss = criterion(logits, labels)
            acc = accuracy(logits.detach(), labels.detach())[0]

            meters["loss"].update(loss.item(), features.size(0))
            meters["acc"].update(acc.item(), features.size(0))

    # Console & tensorboard
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
        "--------------------------End evaluating at epoch:{}--------------------------".format(
            epoch + 1
        )
    )

    return meters["acc"].avg, meters["loss"].avg
