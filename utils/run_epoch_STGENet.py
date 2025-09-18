# ===== utils/run_epoch_STGENet.py =====
import torch
import time
import datetime
from .tools import AverageMeter, accuracy
import swanlab


# ---------- RTA 适配 ----------
def _apply_rta(rta, features, labels, args):
    use_rta = getattr(args, "rta", True) and (rta is not None)
    if not use_rta:
        return features, labels
    try:
        out = rta(features, labels)
    except TypeError:
        out = rta.transform(features, labels)
    if isinstance(out, (tuple, list)):
        if len(out) < 2:
            raise RuntimeError("RTA must return at least (features, labels).")
        return out[0], out[1]
    try:
        f, y = out
        return f, y
    except Exception:
        raise RuntimeError("Unsupported RTA return type; expected (features, labels).")


# ---------- 多指标工具 ----------
def _ensure_cm(cm, y, pred):
    max_label = int(max(y.max().item(), pred.max().item()))
    if max_label + 1 > cm.size(0):
        newC = max_label + 1
        new_cm = torch.zeros(newC, newC, dtype=torch.long)
        new_cm[: cm.size(0), : cm.size(1)] = cm
        cm = new_cm
    for t, p in zip(y.view(-1).detach().cpu(), pred.view(-1).detach().cpu()):
        cm[t, p] += 1
    return cm


def _metrics_from_cm(cm: torch.Tensor):
    C = cm.size(0)
    tp = cm.diag().to(torch.float64)
    support = cm.sum(dim=1).clamp_min(1).to(torch.float64)
    pred_pos = cm.sum(dim=0).clamp_min(1).to(torch.float64)
    recall = tp / support
    precision = tp / pred_pos
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    macro_p = precision.mean().item()
    macro_r = recall.mean().item()
    macro_f1 = f1.mean().item()

    N = cm.sum().item()
    po = tp.sum().item() / (N + 1e-12)
    pe = (cm.sum(0) * cm.sum(1)).sum().item() / (N * N + 1e-12)
    kappa = (po - pe) / (1 - pe + 1e-12)
    return {
        "acc": po,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "kappa": kappa,
    }


def _fmt_metrics(tag, loss, m):
    return (
        f"[{tag}] loss {loss:.4f} | acc {m['acc']*100:5.2f}% | "
        f"P {m['macro_p']*100:5.2f}%  R {m['macro_r']*100:5.2f}%  "
        f"F1 {m['macro_f1']*100:5.2f}% | κ {m['kappa']:+.4f}"
    )


def _fmt_eta(seconds):
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------- Train ----------
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

    dict_log = {"loss": AverageMeter(), "acc": AverageMeter()}
    criterion = criterion.to(device)

    model.train()
    data, data_labels = data
    steps = (
        (data.shape[0] // args.batch_size + 1)
        if (data.shape[0] % args.batch_size)
        else (data.shape[0] // args.batch_size)
    )
    sstep = 0
    node_weights, space_node_weights = None, None

    cm = torch.zeros(
        getattr(args, "n_class", 2), getattr(args, "n_class", 2), dtype=torch.long
    )

    for features, labels in iterator:
        features, labels = _apply_rta(rta, features, labels, args)
        features = features.to(device)
        labels = labels.to(device)

        predicts, node_weights, space_node_weights = model(features)
        loss = criterion(predicts, labels)

        acc = accuracy(predicts.detach(), labels.detach())[0]
        dict_log["loss"].update(loss.item(), len(features))
        dict_log["acc"].update(acc.item(), len(features))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = predicts.argmax(1)
            cm = _ensure_cm(cm, labels, pred)

        all_steps = epoch * steps + sstep + 1
        if 0 == (all_steps % args.print_freq):
            lr = list(optimizer.param_groups)[0]["lr"]
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]
            msg = "id:{}   time consumption:{}    epoch:{}/{}  lr:{}    ".format(
                args.id, et, epoch + 1, args.epochs, lr
            )
            for key, value in dict_log.items():
                msg += "{}(val/avg):{:.3f}/{:.3f}  ".format(key, value.val, value.avg)
                if tensorboard is not None:
                    tensorboard.add_scalar(f"train/{key}", value.val, all_steps)
            print(msg)
        sstep += 1

    train_loss = dict_log["loss"].avg
    m = _metrics_from_cm(cm)
    print(_fmt_metrics("Train", train_loss, m))

    if tensorboard is not None:
        tensorboard.add_scalar("train/loss_epoch", train_loss, epoch + 1)
        tensorboard.add_scalar("train/acc_epoch", m["acc"], epoch + 1)
        tensorboard.add_scalar("train/macro_p", m["macro_p"], epoch + 1)
        tensorboard.add_scalar("train/macro_r", m["macro_r"], epoch + 1)
        tensorboard.add_scalar("train/macro_f1", m["macro_f1"], epoch + 1)
        tensorboard.add_scalar("train/kappa", m["kappa"], epoch + 1)

    try:
        swanlab.log(
            {
                "train/loss": train_loss,
                "train/acc": m["acc"],
                "train/macro_p": m["macro_p"],
                "train/macro_r": m["macro_r"],
                "train/macro_f1": m["macro_f1"],
                "train/kappa": m["kappa"],
            },
            step=epoch + 1,
        )
    except Exception:
        pass

    print(
        "--------------------------End training at epoch:{}--------------------------".format(
            epoch + 1
        )
    )
    return node_weights, space_node_weights


# ---------- Eval ----------
@torch.no_grad()
def evaluate_one_epoch(
    epoch, iterator, data, model, device, criterion, tensorboard, args, start_time, rta
):
    print(
        "--------------------------Start evaluating at epoch:{}--------------------------".format(
            epoch + 1
        )
    )
    model.to(device)
    dict_log = {"loss": AverageMeter(), "acc": AverageMeter()}
    model.eval()
    data, data_labels = data
    tic = time.perf_counter()

    cm = torch.zeros(
        getattr(args, "n_class", 2), getattr(args, "n_class", 2), dtype=torch.long
    )

    num_batches = len(iterator)
    pf_eval = getattr(args, "print_freq_eval", 0)

    for it, (features, labels) in enumerate(iterator, start=1):
        features, labels = _apply_rta(rta, features, labels, args)
        features = features.to(device)
        labels = labels.to(device)

        predicts, _, _ = model(features)
        loss = criterion(predicts, labels)

        acc = accuracy(predicts.detach(), labels.detach())[0]
        dict_log["acc"].update(acc.item(), len(features))
        dict_log["loss"].update(loss.item(), len(features))

        pred = predicts.argmax(1)
        cm = _ensure_cm(cm, labels, pred)

        # —— 按批次打印（仅当提供 print_freq_eval 时启用） ——
        if pf_eval and (it % pf_eval == 0 or it == num_batches):
            elapsed = time.perf_counter() - tic
            avg_loss = dict_log["loss"].avg
            avg_acc = dict_log["acc"].avg
            eta = (elapsed / it) * (num_batches - it)
            msg = (
                f"[Test ][{epoch+1:03d}] iter {it:04d}/{num_batches:04d} | "
                f"loss {loss.item():.4f} (avg {avg_loss:.4f}) | "
                f"acc {acc.item()*100:5.2f}% (avg {avg_acc*100:5.2f}%) | "
                f"elapsed {int(elapsed)}s | eta {_fmt_eta(eta)}"
            )
            print(msg, flush=True)

    eval_loss = dict_log["loss"].avg
    m = _metrics_from_cm(cm)
    et = str(datetime.timedelta(seconds=(time.perf_counter() - tic)))[:-7]
    print(
        f"time consumption:{et}    epoch:{epoch + 1}/{args.epochs}   loss(avg):{eval_loss:.3f} acc(avg):{m['acc']*100:5.2f}%"
    )
    print(_fmt_metrics("Test ", eval_loss, m))

    if tensorboard is not None:
        tensorboard.add_scalar("test/loss_epoch", eval_loss, epoch + 1)
        tensorboard.add_scalar("test/acc_epoch", m["acc"], epoch + 1)
        tensorboard.add_scalar("test/macro_p", m["macro_p"], epoch + 1)
        tensorboard.add_scalar("test/macro_r", m["macro_r"], epoch + 1)
        tensorboard.add_scalar("test/macro_f1", m["macro_f1"], epoch + 1)
        tensorboard.add_scalar("test/kappa", m["kappa"], epoch + 1)

    try:
        swanlab.log(
            {
                "val/loss": eval_loss,
                "val/acc": m["acc"],
                "val/macro_p": m["macro_p"],
                "val/macro_r": m["macro_r"],
                "val/macro_f1": m["macro_f1"],
                "val/kappa": m["kappa"],
            },
            step=epoch + 1,
        )
    except Exception:
        pass

    return m["acc"], eval_loss
