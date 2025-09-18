# ===== utils/run_epoch_single_EEGNet.py =====
import time
import torch

__all__ = ["train_one_epoch", "evaluate_one_epoch"]


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


def _fmt_summary(loss, m):
    return (
        f"loss {loss:.4f} | acc {m['acc']*100:5.2f}% | "
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


# ------------------------ Train ------------------------
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch,
    epochs,
    writer=None,
    swanlab=None,
    n_classes=None,
    print_freq=50,
):
    model.train()
    tic = time.perf_counter()

    total = len(dataloader.dataset)
    num_batches = len(dataloader)

    run_loss, run_correct, seen = 0.0, 0, 0
    C = n_classes if n_classes else 2
    cm = torch.zeros(C, C, dtype=torch.long)

    for it, (x, y) in enumerate(dataloader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # 累计
        bs = x.size(0)
        run_loss += loss.item() * bs
        seen += bs
        with torch.no_grad():
            pred = logits.argmax(1)
            run_correct += (pred == y).sum().item()
            # 动态扩容 cm（保险）
            max_label = int(max(y.max().item(), pred.max().item()))
            if max_label + 1 > cm.size(0):
                newC = max_label + 1
                new_cm = torch.zeros(newC, newC, dtype=torch.long)
                new_cm[: cm.size(0), : cm.size(1)] = cm
                cm = new_cm
            for t, p in zip(y.view(-1).detach().cpu(), pred.view(-1).detach().cpu()):
                cm[t, p] += 1

        # —— 批次打印（保留你“每个 epoch 多行”的风格） ——
        if print_freq > 0 and (it % print_freq == 0 or it == num_batches):
            elapsed = time.perf_counter() - tic
            avg_loss = run_loss / max(1, seen)
            avg_acc = run_correct / max(1, seen)
            lr = optimizer.param_groups[0]["lr"]
            eta = (elapsed / it) * (num_batches - it)
            print(
                f"[Train][{epoch:03d}/{epochs:03d}] "
                f"iter {it:04d}/{num_batches:04d} | "
                f"loss {loss.item():.4f} (avg {avg_loss:.4f}) | "
                f"acc {(pred == y).float().mean().item()*100:5.2f}% (avg {avg_acc*100:5.2f}%) | "
                f"lr {lr:.2e} | elapsed {int(elapsed)}s | eta {_fmt_eta(eta)}",
                flush=True,
            )

    # 末尾汇总一行（与原风格兼容）
    train_loss = run_loss / max(1, seen)
    m = _metrics_from_cm(cm)
    print(
        f"[Train][{epoch:03d}/{epochs:03d}] {_fmt_summary(train_loss, m)}", flush=True
    )

    # 日志
    if writer is not None:
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/acc", m["acc"], epoch)
        writer.add_scalar("train/macro_p", m["macro_p"], epoch)
        writer.add_scalar("train/macro_r", m["macro_r"], epoch)
        writer.add_scalar("train/macro_f1", m["macro_f1"], epoch)
        writer.add_scalar("train/kappa", m["kappa"], epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

    if swanlab is not None:
        try:
            swanlab.log(
                {
                    "train/loss": train_loss,
                    "train/acc": m["acc"],
                    "train/macro_p": m["macro_p"],
                    "train/macro_r": m["macro_r"],
                    "train/macro_f1": m["macro_f1"],
                    "train/kappa": m["kappa"],
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )
        except Exception:
            pass

    # 与三分支接口对齐：占位返回
    return None, None


# ------------------------ Eval ------------------------
@torch.no_grad()
def evaluate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    epoch,
    split="Val",
    writer=None,
    swanlab=None,
    n_classes=None,
    print_freq=0,
):
    model.eval()
    tic = time.perf_counter()

    num_batches = len(dataloader)
    run_loss, seen = 0.0, 0
    C = n_classes if n_classes else 2
    cm = torch.zeros(C, C, dtype=torch.long)

    for it, (x, y) in enumerate(dataloader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        run_loss += loss.item() * bs
        seen += bs

        pred = logits.argmax(1)
        # 动态扩容 cm（保险）
        max_label = int(max(y.max().item(), pred.max().item()))
        if max_label + 1 > cm.size(0):
            newC = max_label + 1
            new_cm = torch.zeros(newC, newC, dtype=torch.long)
            new_cm[: cm.size(0), : cm.size(1)] = cm
            cm = new_cm
        for t, p in zip(y.view(-1).detach().cpu(), pred.view(-1).detach().cpu()):
            cm[t, p] += 1

        # —— 测试阶段批次打印（可控频率） ——
        if print_freq and (it % print_freq == 0 or it == num_batches):
            elapsed = time.perf_counter() - tic
            avg_loss = run_loss / max(1, seen)
            avg_acc = cm.diag().sum().item() / max(1, cm.sum().item())
            eta = (elapsed / it) * (num_batches - it)
            tag = "Test" if split.lower().startswith("test") else "Val"
            print(
                f"[{tag} ][{epoch:03d}] "
                f"iter {it:04d}/{num_batches:04d} | "
                f"loss {loss.item():.4f} (avg {avg_loss:.4f}) | "
                f"acc {(pred == y).float().mean().item()*100:5.2f}% (avg {avg_acc*100:5.2f}%) | "
                f"elapsed {int(elapsed)}s | eta {_fmt_eta(eta)}",
                flush=True,
            )

    eval_loss = run_loss / max(1, seen)
    m = _metrics_from_cm(cm)
    tag = "Test" if split.lower().startswith("test") else "Val"
    print(f"[{tag} ][{epoch:03d}] {_fmt_summary(eval_loss, m)}", flush=True)

    # 日志
    if writer is not None:
        base = tag.lower()
        writer.add_scalar(f"{base}/loss", eval_loss, epoch)
        writer.add_scalar(f"{base}/acc", m["acc"], epoch)
        writer.add_scalar(f"{base}/macro_p", m["macro_p"], epoch)
        writer.add_scalar(f"{base}/macro_r", m["macro_r"], epoch)
        writer.add_scalar(f"{base}/macro_f1", m["macro_f1"], epoch)
        writer.add_scalar(f"{base}/kappa", m["kappa"], epoch)

    if swanlab is not None:
        try:
            swanlab.log(
                {
                    f"{tag.lower()}/loss": eval_loss,
                    f"{tag.lower()}/acc": m["acc"],
                    f"{tag.lower()}/macro_p": m["macro_p"],
                    f"{tag.lower()}/macro_r": m["macro_r"],
                    f"{tag.lower()}/macro_f1": m["macro_f1"],
                    f"{tag.lower()}/kappa": m["kappa"],
                },
                step=epoch,
            )
        except Exception:
            pass

    return {"loss": eval_loss, **m}
