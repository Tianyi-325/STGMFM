from einops.layers.torch import Rearrange
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
import pytorch_lightning as pl


def linear_warmup_cosine_decay(warmup_steps: int, total_steps: int):
    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )

        # cosine decay
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return fn


def interaug(batch):
    x, y = batch
    new_samples = torch.zeros_like(x)
    new_labels = torch.zeros_like(y)
    current = 0
    n_chunks = 8 if new_samples.shape[-1] % 8 == 0 else 7  # special case for BCIC III
    for cls in torch.unique(y):
        x_cls = x[y == cls]
        chunks = torch.cat(torch.chunk(x_cls, chunks=n_chunks, dim=-1))
        indices = np.random.choice(
            len(x_cls), size=(len(x_cls), n_chunks), replace=True
        )
        for idx in indices:
            # add offset
            idx += np.arange(0, chunks.shape[0], len(x_cls))

            # create new sample
            new_sample = chunks[idx]
            new_sample = new_sample.permute(1, 0, 2).reshape(
                1, x_cls.shape[1], x_cls.shape[2]
            )
            new_samples[current] = new_sample
            new_labels[current] = cls
            current += 1

    combined_x = torch.cat((x, new_samples), dim=0)
    combined_y = torch.cat((y, new_labels), dim=0)

    # shuffle
    perm = torch.randperm(len(combined_x))
    combined_x = combined_x[perm]
    combined_y = combined_y[perm]

    return combined_x, combined_y


def glorot_weight_zero_bias(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            if "norm" not in module.__class__.__name__.lower():
                nn.init.xavier_uniform_(module.weight)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class ClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model,
        n_classes,
        lr=0.001,
        weight_decay=0.0,
        optimizer="adam",
        scheduler=False,
        max_epochs=1000,
        warmup_epochs=20,
        **kwargs,
    ):
        super(ClassificationModule, self).__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                betas=betas,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                betas=betas,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise NotImplementedError
        if self.hparams.scheduler:
            scheduler = LambdaLR(
                optimizer,
                linear_warmup_cosine_decay(
                    self.hparams.warmup_epochs, self.hparams.max_epochs
                ),
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="val")
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="test")
        return {"test_loss": loss, "test_acc": acc}

    def shared_step(self, batch, batch_idx, mode: str = "train"):
        if mode == "train" and self.hparams.get("interaug", False):
            x, y = interaug(batch)
        else:
            x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return torch.argmax(self.forward(x), dim=-1)


class ShallowNetModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        input_window_samples: int,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.5,
    ):
        super(ShallowNetModule, self).__init__()
        self.rearrange_input = Rearrange("b c t -> b 1 t c")
        self.conv_time = nn.Conv2d(
            1, n_filters_time, (filter_time_length, 1), bias=True
        )
        self.conv_spat = nn.Conv2d(
            n_filters_time, n_filters_time, (1, in_channels), bias=False
        )
        self.bnorm = nn.BatchNorm2d(n_filters_time)

        self.pool = nn.AvgPool2d((pool_time_length, 1), (pool_time_stride, 1))
        self.dropout = nn.Dropout(drop_prob)
        out = input_window_samples - filter_time_length + 1
        out = int((out - pool_time_length) / pool_time_stride + 1)

        self.classifier = nn.Sequential(
            nn.Conv2d(n_filters_time, n_classes, (out, 1)), Rearrange("b c 1 1 -> b c")
        )
        glorot_weight_zero_bias(self)

    def forward(self, x):
        x = self.rearrange_input(x)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bnorm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class ShallowNet(ClassificationModule):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        input_window_samples: int,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.5,
        **kwargs,
    ):
        model = ShallowNetModule(
            in_channels=in_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            drop_prob=drop_prob,
        )
        super(ShallowNet, self).__init__(model, n_classes, **kwargs)
