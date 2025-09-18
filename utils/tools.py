import numpy as np
import torch
import random
import errno
import os
import sys
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from utils.cutmix import CutMix
from utils.random_crop import RandomCrop
from utils.random_erasing import RandomErasing


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_save_path(father_path, args):
    father_path = os.path.join(father_path, "{}".format(time.strftime("%m_%d_%H_%M")))
    mkdir(father_path)
    args.log_path = father_path
    args.model_path = father_path
    args.result_path = father_path
    args.spatial_adj_path = father_path
    args.time_adj_path = father_path
    args.tensorboard_path = father_path
    return args


def sliding_window_eeg(data, label, window_size, stride):
    # print(data.shape)
    trails = data.shape[0]
    num_channels = data.shape[1]
    num_samples = data.shape[2]
    num_segments = (num_samples - window_size) // stride + 1
    segments = np.zeros(
        (num_segments, trails, num_channels, window_size), dtype=np.float64
    )

    for i in range(trails):
        for j in range(num_segments):
            start = j * stride
            end = start + window_size
            segments[j][i] = data[i, :, start:end]

    return segments, label


EOS = 1e-10


def normalize(adj):
    adj = F.relu(adj)
    inv_sqrt_degree = 1.0 / (torch.sqrt(torch.sum(adj, dim=-1, keepdim=False)) + EOS)
    return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]


def save(checkpoints, save_path):
    torch.save(checkpoints, save_path)


def load_adj(dn="bciciv2a", norm=False):
    if "bciciv2a" == dn:
        num_node = 22
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [
            (1, 3),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 7),
            (2, 8),
            (2, 9),
            (3, 4),
            (3, 8),
            (3, 9),
            (3, 10),
            (4, 5),
            (4, 9),
            (4, 10),
            (4, 11),
            (5, 6),
            (5, 10),
            (5, 11),
            (5, 12),
            (6, 11),
            (6, 12),
            (6, 13),
            (7, 8),
            (7, 14),
            (8, 9),
            (8, 14),
            (8, 15),
            (9, 10),
            (9, 14),
            (9, 15),
            (9, 16),
            (10, 11),
            (10, 15),
            (10, 16),
            (10, 17),
            (11, 12),
            (11, 16),
            (11, 17),
            (11, 18),
            (12, 13),
            (12, 17),
            (12, 18),
            (13, 18),
            (14, 15),
            (14, 19),
            (15, 16),
            (15, 19),
            (15, 20),
            (16, 17),
            (16, 19),
            (16, 20),
            (16, 21),
            (17, 18),
            (17, 20),
            (17, 21),
            (18, 21),
            (19, 20),
            (19, 22),
            (20, 21),
            (20, 22),
            (21, 22),
        ]

    elif "oy" == dn:
        num_node = 23
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [
            (1, 2),
            (1, 6),
            (2, 3),
            (2, 8),
            (3, 4),
            (3, 8),
            (4, 5),
            (4, 8),
            (5, 9),
            (6, 7),
            (6, 8),
            (6, 7),
            (6, 8),
            (6, 10),
            (6, 12),
            (7, 8),
            (7, 9),
            (7, 8),
            (7, 9),
            (7, 10),
            (7, 12),
            (7, 13),
            (8, 9),
            (9, 12),
            (9, 13),
            (10, 11),
            (10, 12),
            (10, 14),
            (11, 12),
            (11, 13),
            (11, 15),
            (11, 16),
            (11, 17),
            (12, 13),
            (13, 23),
            (15, 16),
            (15, 18),
            (15, 21),
            (16, 17),
            (16, 20),
            (17, 19),
            (17, 22),
            (18, 21),
            (19, 20),
            (19, 22),
            (20, 21),
        ]

    else:
        raise ValueError("cant support {} dataset".format(dn))
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_link]
    edge = self_link + neighbor_link
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if "physionet" == dn:
        for i in range(64):
            for j in range(64):
                if A[i, j] == 0:
                    A[i, j] = 0.01
    elif "bciciv2a" == dn:
        for i in range(22):
            for j in range(22):
                if A[i, j] == 0:
                    A[i, j] = 0.1
    return A


def compute_plv_adj(train_X, tau=None, topk=None):
    """
    train_X: torch.Tensor [trials, channels, T] 或 numpy，返回 [C, C] 的对称图
    """
    if isinstance(train_X, np.ndarray):
        x = torch.from_numpy(train_X).float()
    else:
        x = train_X.float()
    # 去均值，计算解析信号 (Hilbert via FFT)
    x = x - x.mean(dim=-1, keepdim=True)
    N = x.size(-1)
    Xf = torch.fft.fft(x, dim=-1)  # [B,C,T] -> 复数
    h = torch.zeros(N, dtype=Xf.dtype, device=Xf.device)
    h[0] = 1.0
    if N % 2 == 0:
        h[N // 2] = 1.0
        h[1 : N // 2] = 2.0
    else:
        h[1 : (N + 1) // 2] = 2.0
    z = torch.fft.ifft(Xf * h, dim=-1)  # 解析信号
    phase = torch.angle(z)  # [B,C,T]
    V = torch.exp(1j * phase).permute(1, 0, 2).reshape(x.size(1), -1)  # [C, B*T]
    M = (V @ V.conj().T) / V.size(1)  # [C,C] 复相关
    A = torch.abs(M).float()  # PLV ∈ [0,1]
    A.fill_diagonal_(0.0)
    A = 0.5 * (A + A.T)  # 对称化
    if tau is not None:
        A = (A >= tau).float()  # 阈值二值化
    if topk is not None:
        # 每行保留 topk 边（不含自环）
        k = min(topk, A.size(0) - 1)
        vals, idx = torch.topk(A, k=k + 1, dim=1)  # +1 是包含对角元素
        mask = torch.zeros_like(A)
        rows = torch.arange(A.size(0))[:, None].expand(A.size(0), k + 1)
        mask[rows, idx] = 1.0
        mask.fill_diagonal_(0.0)
        A = torch.maximum(mask, mask.T).float()
    return A


def accuracy(output, target, topk=(1,)):
    shape = None
    if 2 == len(target.size()):
        shape = target.size()
        target = target.view(target.size(0))
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1.0 / batch_size))
    if shape:
        target = target.view(shape)
    return ret


class Logger(object):
    def __init__(self, fpath):
        self.console = sys.stdout
        self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, *args):
        for t in self.transforms:
            x = t(x, *args)
        return x


def build_tranforms():
    return Compose(
        [
            RandomCrop(1125),  ###########写死
            # RandomCrop(1375),
            CutMix(),
            # RandomErasing(),
        ]
    )


def build_tranforms2():
    return Compose(
        [
            RandomCrop(1125),  ###########写死
            # RandomCrop(1375),
            # CutMix(),
            # RandomErasing(),
        ]
    )


class EEGDataSet(Dataset):
    def __init__(self, data, label):
        self.label = label
        self.data = data

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, index):
        data = self.data[:, index]
        label = self.label[index]
        return data, label


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
