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


def _graph_terms(model: torch.nn.Module):
    """Yield all graph-related parameters present in *any* STGENET variant."""
    for attr in [
        # 原双分支
        ("tsg_after", "adj"),
        ("tsg_before", "adj"),
        # 频带分支（若存在）
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


def _compute_l2(model: torch.nn.Module, l2_lambda: float = 1e-5) -> torch.Tensor:
    """L2 (weight-decay style) on *all* parameters."""
    l2 = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        l2 += torch.sum(p.pow(2))
    return l2_lambda * l2


# -----------------------------------------------------------------------------
# Metrics helpers (Kappa, Recall, Precision, F1 - macro)
# -----------------------------------------------------------------------------

_EPS = 1e-12


def _init_confusion(n_classes: int) -> torch.Tensor:
    # Rows = ground-truth, Cols = predicted
    return torch.zeros((n_classes, n_classes), dtype=torch.long)


@torch.no_grad()
def _update_confusion(
    cm: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor, n_classes: int
):
    idx = y_true.to(torch.long) * n_classes + y_pred.to(torch.long)
    binc = torch.bincount(idx, minlength=n_classes * n_classes)
    cm += binc.reshape(n_classes, n_classes).cpu()


def _metrics_from_confusion(cm: torch.Tensor):
    """
    Returns: (kappa, recall_macro, precision_macro, f1_macro)
    """
    cm = cm.to(torch.float64)
    N = cm.sum().item()
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0

    trace = torch.trace(cm).item()
    p0 = trace / N
    row_sum = cm.sum(dim=1)
    col_sum = cm.sum(dim=0)
    pe = (row_sum * col_sum).sum().item() / (N * N)
    kappa = 0.0 if abs(1.0 - pe) < _EPS else (p0 - pe) / (1.0 - pe)

    tp = torch.diag(cm)
    fp = col_sum - tp
    fn = row_sum - tp
    support = row_sum

    precision_per = tp / torch.clamp(tp + fp, min=_EPS)
    recall_per = tp / torch.clamp(tp + fn, min=_EPS)
    f1_per = (
        2
        * precision_per
        * recall_per
        / torch.clamp(precision_per + recall_per, min=_EPS)
    )

    mask = support > 0
    if mask.sum() == 0:
        return kappa, 0.0, 0.0, 0.0

    precision_macro = precision_per[mask].mean().item()
    recall_macro = recall_per[mask].mean().item()
    f1_macro = f1_per[mask].mean().item()
    return kappa, recall_macro, precision_macro, f1_macro


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

    # metrics accumulators (epoch-level)
    cm_train = None
    n_classes = None

    total_steps = (
        data[0].shape[0] // args.batch_size + 1
        if data[0].shape[0] % args.batch_size
        else data[0].shape[0] // args.batch_size
    )

    # holders for last graphs (for plotting)
    ccg_w_A, ccg_w_B = None, None

    for step, (features, labels) in enumerate(iterator):
        # ---- data aug ----
        features, labels = rta(features, labels)
        features, labels = features.to(device), labels.to(device)

        # ---- forward ----
        logits, ccg_w_A, tsg_A, tsg_B, ccg_w_B = model(features)

        # ---- losses ----
        ce_loss = criterion(logits, labels)
        l2_reg = _compute_l2(model)
        l1_reg = _compute_l1_graph(model)
        total_loss = ce_loss + l2_reg + l1_reg
        # total_loss = ce_loss

        # ---- metrics ----
        with torch.no_grad():
            acc = accuracy(logits.detach(), labels.detach())[0]
            meters["loss"].update(total_loss.item(), features.size(0))
            meters["acc"].update(acc.item(), features.size(0))

            # confusion matrix update
            if n_classes is None:
                n_classes = logits.size(1)
                cm_train = _init_confusion(n_classes)
            y_pred = logits.detach().argmax(dim=1)
            _update_confusion(cm_train, y_pred, labels.detach(), n_classes)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ---- console & tb (keep your pretty style) ----
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

    # ---- epoch summary (compute macro metrics) ----
    kappa_tr, recall_tr, precision_tr, f1_tr = _metrics_from_confusion(cm_train)

    # epoch-level logs
    tensorboard.add_scalar("train/loss_epoch", meters["loss"].avg, epoch)
    tensorboard.add_scalar("train/acc_epoch", meters["acc"].avg, epoch)
    tensorboard.add_scalar("train/kappa", kappa_tr, epoch)
    tensorboard.add_scalar("train/recall_macro", recall_tr, epoch)
    tensorboard.add_scalar("train/precision_macro", precision_tr, epoch)
    tensorboard.add_scalar("train/f1_macro", f1_tr, epoch)

    swanlab.log(
        {
            "train/loss": meters["loss"].avg,
            "train/acc": meters["acc"].avg,
            "train/kappa": kappa_tr,
            "train/recall": recall_tr,
            "train/precision": precision_tr,
            "train/f1": f1_tr,
        },
        step=epoch + 1,
    )

    print(
        f"--------------------------End training at epoch:{epoch+1}--------------------------"
    )

    # Return the CCG matrices for plotting (keep original behavior)
    ccg_w_A = ccg_w_A.detach().cpu().numpy() if ccg_w_A is not None else None
    ccg_w_B = ccg_w_B.detach().cpu().numpy() if ccg_w_B is not None else None
    return ccg_w_A, ccg_w_B


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

    # metrics accumulators (epoch-level)
    cm_val = None
    n_classes = None

    with torch.no_grad():
        for features, labels in iterator:
            features, labels = rta(features, labels)
            features, labels = features.to(device), labels.to(device)

            logits, *_ = model(features)  # predictions only
            loss = criterion(logits, labels)
            acc = accuracy(logits.detach(), labels.detach())[0]

            meters["loss"].update(loss.item(), features.size(0))
            meters["acc"].update(acc.item(), features.size(0))

            # cm update
            if n_classes is None:
                n_classes = logits.size(1)
                cm_val = _init_confusion(n_classes)
            y_pred = logits.detach().argmax(dim=1)
            _update_confusion(cm_val, y_pred, labels.detach(), n_classes)

    # epoch macro metrics
    kappa_v, recall_v, precision_v, f1_v = _metrics_from_confusion(cm_val)

    elapsed = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
    info = (
        f"time:{elapsed}  epoch:{epoch+1}/{args.epochs}  "
        f"loss(avg):{meters['loss'].avg:.3f}  acc(avg):{meters['acc'].avg:.3f}  "
        f"kappa:{kappa_v:.3f}  R:{recall_v:.3f}  P:{precision_v:.3f}  F1:{f1_v:.3f}"
    )
    print(info)

    # epoch logs
    tensorboard.add_scalar("val/loss", meters["loss"].avg, epoch)
    tensorboard.add_scalar("val/acc", meters["acc"].avg, epoch)
    tensorboard.add_scalar("val/kappa", kappa_v, epoch)
    tensorboard.add_scalar("val/recall_macro", recall_v, epoch)
    tensorboard.add_scalar("val/precision_macro", precision_v, epoch)
    tensorboard.add_scalar("val/f1_macro", f1_v, epoch)

    swanlab.log(
        {
            "val/loss": meters["loss"].avg,
            "val/acc": meters["acc"].avg,
            "val/kappa": kappa_v,
            "val/recall": recall_v,
            "val/precision": precision_v,
            "val/f1": f1_v,
        },
        step=epoch + 1,
    )
    print(
        f"--------------------------End evaluating at epoch:{epoch+1}--------------------------"
    )

    return meters["acc"].avg, meters["loss"].avg
