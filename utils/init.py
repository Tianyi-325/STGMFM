from torch.nn import init
import torch


def glorot_weight_zero_bias(model):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        # if isinstance(module, torch.nn.ModuleList):
        #     glorot_weight_zero_bias(module)
        # elif isinstance(module, torch.nn.Sequential):
        #     glorot_weight_zero_bias(module)
        # else:
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                print(module.__class__.__name__)
                init.xavier_uniform_(module.weight, gain=1)
            else:
                init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                init.constant_(module.bias, 0)


# from torch.nn import init
# import torch


# def glorot_weight_zero_bias(model):
#     """
#     Xavier-uniform 初始化所有可学习 weight (≥2D)，
#     1D 权重（LayerNorm / BatchNorm / …）保持默认或按需求特殊处理，
#     bias 统一置零。
#     """
#     for module in model.modules():

#         # ---------- 处理 weight ----------
#         if hasattr(module, "weight") and module.weight is not None:
#             w = module.weight
#             cls_name = module.__class__.__name__

#             if "BatchNorm" in cls_name:
#                 # BatchNorm 权重通常初始化为 1
#                 init.constant_(w, 1.0)

#             elif w.dim() >= 2:
#                 # 只有 ≥2D tensor 才能使用 Xavier/Gluorot
#                 init.xavier_uniform_(w, gain=1.0)

#             # 其余情况 (e.g. LayerNorm 1D weight) 保持 PyTorch 默认值

#         # ---------- 处理 bias ----------
#         if hasattr(module, "bias") and module.bias is not None:
#             init.constant_(module.bias, 0.0)
