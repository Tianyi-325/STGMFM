import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F


nonlinearity_dict = dict(relu=nn.ReLU(), elu=nn.ELU())


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(F.pad(x, (self.__padding, 0)))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


def glorot_weight_zero_bias(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            if "norm" not in module.__class__.__name__.lower():
                nn.init.xavier_uniform_(module.weight)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class _TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: str = "relu",
    ):
        super(_TCNBlock, self).__init__()
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity1 = nonlinearity_dict[activation]
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity2 = nonlinearity_dict[activation]
        self.drop2 = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.project_channels = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.project_channels = nn.Identity()
        self.final_nonlinearity = nonlinearity_dict[activation]

    def forward(self, x):
        residual = self.project_channels(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinearity1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlinearity2(out)
        out = self.drop2(out)
        return self.final_nonlinearity(out + residual)


class EEGTCNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_channels: int = 22,
        layers: int = 2,
        kernel_s: int = 4,
        filt: int = 12,
        dropout: float = 0.3,
        activation: str = "relu",
        F1: int = 8,
        D: int = 2,
        kernLength: int = 32,
        dropout_eeg: float = 0.2,
    ):
        super().__init__()
        regRate = 0.25
        numFilters = F1
        F2 = numFilters * D

        self.eegnet = nn.Sequential(
            Rearrange("b c t -> b 1 c t"),
            nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, eps=0.001),
            Conv2dWithConstraint(
                F1, F2, (in_channels, 1), bias=False, groups=F1, max_norm=1
            ),
            nn.BatchNorm2d(F2, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_eeg),
            nn.Conv2d(F2, F2, (1, 16), padding="same", groups=F2, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_eeg),
            Rearrange("b c 1 t -> b c t"),
        )

        in_channels = [F2] + (layers - 1) * [filt]
        dilations = [2**i for i in range(layers)]
        self.tcn_blocks = nn.ModuleList(
            [
                _TCNBlock(
                    in_ch,
                    filt,
                    kernel_size=kernel_s,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                )
                for in_ch, dilation in zip(in_channels, dilations)
            ]
        )

        self.classifier = LinearWithConstraint(filt, n_classes, max_norm=regRate)
        glorot_weight_zero_bias(self.eegnet)
        glorot_weight_zero_bias(self.classifier)

    def forward(self, x):
        x = self.eegnet(x)
        for blk in self.tcn_blocks:
            x = blk(x)
        x = self.classifier(x[:, :, -1])
        return x


if __name__ == "__main__":
    inp = torch.rand(1, 22, 1000)
    model = EEGTCNet(4)
    out = model(inp)
    print(out.shape)
