import time
import torch
import numpy as np
import torch.nn.functional as F
import swanlab
from sklearn.metrics import accuracy_score, cohen_kappa_score

def train_one_epoch(epoch, train_loader, data, model, device, optimizer, criterion,
                    logger, start_time, _, rta):
    model.train()
    losses = []
    y_true, y_pred = [], []

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        X_aug = rta(X)
        out, _, _ = model(X_aug)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        losses.append(loss.item())

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    logger.log({"train/loss": np.mean(losses), "train/acc": acc, "train/kappa": kappa}, step=epoch)
    print(f"Epoch {epoch} | Train Loss: {np.mean(losses):.4f} | Acc: {acc:.4f} | Kappa: {kappa:.4f}")

def evaluate_one_epoch(epoch, val_loader, data, model, device, criterion,
                        logger, _, start_time, rta):
    model.eval()
    losses = []
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            X_aug = rta(X)
            out, _, _ = model(X_aug)
            loss = criterion(out, y)

            pred = out.argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            losses.append(loss.item())

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    logger.log({"val/loss": np.mean(losses), "val/acc": acc, "val/kappa": kappa}, step=epoch)
    print(f"Epoch {epoch} | Val Loss: {np.mean(losses):.4f} | Acc: {acc:.4f} | Kappa: {kappa:.4f}")

    return acc, kappa
