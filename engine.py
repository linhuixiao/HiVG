# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
from utils.box_utils import xywh2xyxy
import numpy as np


# TODO: 训练核心代码
def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, max_norm: float = 0):
    # 设置模型在训练模式
    model.train()
    # TODO: metric_logger 是个啥东西？
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        # TODO: 这是从 data_loader.py 的 TransVGdataset 的 __getitem__()中获取的
        img_data, text_data, target, obj_mask = batch

        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        obj_mask = obj_mask.to(device)

        # model forward core-computer
        output, text_eos, img_cls, visu_sim, seg_mask = model(img_data, text_data)

        # The `loss_dict` is a dictionary that contains `l1_smooth` and `giou`.
        loss_dict = loss_utils.trans_vg_loss(args, output, target, obj_mask, text_eos, img_cls, visu_sim, seg_mask)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:  # The default value of max_norm is 0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target, tgt_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        tgt_mask = tgt_mask.to(device)

        pred_boxes, _, _, _, seg_mask = model(img_data, text_data)
        miou, accu, mask_iou_list = eval_utils.trans_vg_eval_val(args, pred_boxes, target, seg_mask, tgt_mask)

        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)
        metric_logger.update_v2('mask seg miou', torch.mean(mask_iou_list), batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    text_list = []

    pred_mask_list = []
    gt_mask_list = []

    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, tgt_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        tgt_mask = tgt_mask.to(device)
        output, _, _, token_sim, seg_mask = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

        pred_mask_list.append(seg_mask.cpu())
        gt_mask_list.append(tgt_mask.cpu())

        for text_i in text_data:
            text_list.append(text_i)

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)

    pred_masks = torch.cat(pred_mask_list, dim=0)
    gt_masks = torch.cat(gt_mask_list, dim=0)

    total_num = gt_boxes.shape[0]
    accu_num, iou, mask_iou_list = eval_utils.trans_vg_eval_test(args, pred_boxes, gt_boxes, pred_masks, gt_masks)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    if args.use_mask_loss:
        # It is work only used for referring image segmentation task and enable use args.use_seg_mask
        acc_mask_iou = torch.sum(mask_iou_list, dim=0)
        mask_result_tensor = torch.tensor([acc_mask_iou, total_num]).to(device)

    statistic_diff_length_acc = False
    if statistic_diff_length_acc:  # only can be used in one GPU, used for result comparison.
        # calculate text length, statistics
        assert len(text_list) == iou.shape[0]
        count_for_len_in_1_to_5 = [0, 0]
        count_for_len_in_6_to_7 = [0, 0]
        count_for_len_in_8_to_10 = [0, 0]
        count_for_len_in_11_plus = [0, 0]
        for i in range(len(text_list)):
            len_i = len(text_list[i].split(" "))
            iou_i = iou[i]
            if (len_i >= 1) and (len_i <= 5):
                count_for_len_in_1_to_5[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_1_to_5[0] += 1
            elif (len_i >= 6) and (len_i <= 7):
                count_for_len_in_6_to_7[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_6_to_7[0] += 1
            elif (len_i >= 8) and (len_i <= 10):
                count_for_len_in_8_to_10[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_8_to_10[0] += 1
            elif (len_i >= 11):
                count_for_len_in_11_plus[1] += 1
                if iou_i >= 0.5:
                    count_for_len_in_11_plus[0] += 1

        print("acc in length  1-5: ", count_for_len_in_1_to_5, ", ",
              count_for_len_in_1_to_5[0] / count_for_len_in_1_to_5[1])
        print("acc in length  6-7: ", count_for_len_in_6_to_7, ", ",
              count_for_len_in_6_to_7[0] / count_for_len_in_6_to_7[1])
        print("acc in length 8-10: ", count_for_len_in_8_to_10, ", ",
              count_for_len_in_8_to_10[0] / count_for_len_in_8_to_10[1])
        print("acc in length  11+: ", count_for_len_in_11_plus, ", ",
              count_for_len_in_11_plus[0] / count_for_len_in_11_plus[1])

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)
    if args.use_mask_loss:
        dist.all_reduce(mask_result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    print("accuracy2: ", accuracy)
    if args.use_mask_loss:
        # It is work only used for referring image segmentation task and enable use args.use_seg_mask
        miou = float(mask_result_tensor[0]) / float(mask_result_tensor[1])
        # print("segmentation miou: ", miou)

    return accuracy


@torch.no_grad()
def evaluate_ori(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    pred_box_list = []
    gt_box_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, obj_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        output, _, _, _, seg_mask = model(img_data, text_data)

        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)
    total_num = gt_boxes.shape[0]
    accu_num = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)

    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])

    return accuracy

