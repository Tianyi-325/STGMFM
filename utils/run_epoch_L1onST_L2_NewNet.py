import torch
import math
import sys
import time
import datetime
from .tools import AverageMeter, accuracy
import swanlab

# from thop import profile


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
        data.shape[0] // args.batch_size + 1
        if data.shape[0] % args.batch_size
        else data.shape[0] // args.batch_size
    )
    sstep = 0
    for features, labels in iterator:
        features, labels = rta(features, labels)
        features = features.to(device)
        labels = labels.to(device)

        predicts, node_weights = model(features)

        # 计算原始损失
        loss = criterion(predicts, labels)

        # —— L2：所有参数 ——
        l2_lambda = 1e-5
        l2_reg = 0
        for p in model.parameters():
            l2_reg += torch.sum(p.pow(2))
        l2_reg = l2_lambda * l2_reg

        # —— L1：只对 SpatialGraph.edge_weight 和 TimeGraph.adj ——
        l1_lambda = 1e-5
        sg = model.spatial_module  # SpatialGraph 实例
        l1_reg = torch.sum(torch.abs(sg.edge_weight))
        l1_reg = l1_lambda * l1_reg

        # —— 总损失 ——
        total_loss = loss + l2_reg + l1_reg

        acc = accuracy(predicts.detach(), labels.detach())[0]
        dict_log["loss"].update(total_loss.item(), len(features))
        dict_log["acc"].update(acc.item(), len(features))
        res = dict(loss=None, acc=None)  # hhhh
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        all_steps = epoch * steps + sstep + 1
        if 0 == (all_steps % args.print_freq):
            lr = list(optimizer.param_groups)[0]["lr"]
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]
            print_information = (
                "id:{}   time consumption:{}    epoch:{}/{}  lr:{}    ".format(
                    args.id, et, epoch + 1, args.epochs, lr
                )
            )
            for key, value in dict_log.items():
                loss_info = "{}(val/avg):{:.3f}/{:.3f}  ".format(
                    key, value.val, value.avg
                )
                print_information = print_information + loss_info
                tensorboard.add_scalar(key, value.val, all_steps)
                res[key] = value.avg  # hhhh
            print(print_information)
        sstep = sstep + 1

    swanlab.log(
        {"train/loss": res["loss"], "train/acc": res["acc"]}, step=epoch + 1
    )  # hhhh
    print(
        "--------------------------End training at epoch:{}--------------------------".format(
            epoch + 1
        )
    )
    return node_weights


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
    res = dict(loss=None, acc=None)  # hhhh
    model.eval()
    data, data_labels = data
    # step = 0
    start_time = time.time()
    for features, labels in iterator:
        features, labels = rta(features, labels)
        # x = x.permute(0,2,1,3)
        # x = x.contiguous().view(x.shape[0],x.shape[1],-1)
        # features = features.permute(0,2,1,3)
        # features = features.contiguous().view(features.shape[0],features.shape[1],-1)

        features = features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            predicts, _ = model(features)
            loss = criterion(predicts, labels)
        acc = accuracy(predicts.detach(), labels.detach())[0]
        dict_log["acc"].update(acc.item(), len(features))
        dict_log["loss"].update(loss.item(), len(features))
    end_time = time.time()
    now_time = time.time() - start_time
    et = str(datetime.timedelta(seconds=now_time))[:-7]
    print_information = "time consumption:{}    epoch:{}/{}   ".format(
        et, epoch + 1, args.epochs, len(data)
    )

    for key, value in dict_log.items():
        loss_info = "{}(avg):{:.3f} ".format(key, value.avg)
        print_information = print_information + loss_info
        tensorboard.add_scalar(key, value.val, epoch)
        res[key] = value.avg  # hhhh

    duration_time = "    " + str(end_time - start_time)
    print(print_information + duration_time)
    swanlab.log(
        {"val/loss": res["loss"], "val/acc": res["acc"]}, step=epoch + 1
    )  # hhhh
    print(
        "--------------------------Ending evaluating at epoch:{}--------------------------".format(
            epoch + 1
        )
    )
    return dict_log["acc"].avg, dict_log["loss"].avg
