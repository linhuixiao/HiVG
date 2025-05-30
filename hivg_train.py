import os
import time
import math
import json
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, validate


def get_args_parser():
    parser = argparse.ArgumentParser('HiVG Training Args', add_help=False)
    parser.add_argument('--sup_type', default='full', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--lr_exponential', default=0.9, type=float, help='lr exponential')
    parser.add_argument('--clip_max_norm', default=0., type=float, help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_drop', default=60, type=int)

    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true', help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true', help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true', help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true', help="If true, use random translate augmentation")
    # only support ViT-B/16 and ViT-L/14
    parser.add_argument('--model', type=str, default='ViT-B/16', help="Name of model to be exploited.")
    # Model parameters
    parser.add_argument('--model_name', type=str, default='HiVG', help="Name of model to be exploited.")
    parser.add_argument('--extract_layer', default=0, type=int)
    parser.add_argument('--warmup', action='store_true', help="If true, vision adapt layer is null")

    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--imsize', default=224, type=int, help='image size')
    """ embedding size"""
    parser.add_argument('--emb_size', default=512, type=int, help='fusion module embedding dimensions')
    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=512, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--vl_dec_layers', default=6, type=int,
                        help='Number of decoders in the vision-language transformer')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./data/image_data/', help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='./data/pseudo_samples/',  help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str, help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument('--max_query_len', default=77, type=int, help='maximum time steps (lang length) per batch')

    # Prompt Engineering
    parser.add_argument('--prompt', type=str, default='', help="Prompt template")
    parser.add_argument('--use_cot_prompt', action='store_true', help="If true, using COT prompt")
    parser.add_argument('--cot_length', type=int, default=0, help="Prompt template")
    parser.add_argument('--use_contrastive_loss', action='store_true', help="If true, use contrastive loss")
    parser.add_argument('--use_rtcc_constrain_loss', action='store_true', help="If true, use contrastive loss")
    parser.add_argument('--use_mask_loss', action='store_true', help="If true, use segmentation loss")
    parser.add_argument('--use_seg_mask', action='store_true',
                        help="If true, use segmentation mask in the segmentation task, otherwise use box mask.")
    parser.add_argument('--retrain', default='', help='retrain from checkpoint')
    parser.add_argument('--adapt_mlp', action='store_true', help="If true, use contrastive loss")
    parser.add_argument('--normalize_before', action='store_true', help="If true, use normalize_before")
    parser.add_argument('--save_hilora_clip', action='store_true', help="If true, save hilora clip model")
    parser.add_argument('--hi_lora_stage', default=0, type=int, help='lora stage')
    parser.add_argument('--hi_lora_retrain', default='', help='lora retrain from checkpoint')
    parser.add_argument('--hi_lora_clip', default='', type=str, help='clip model')
    parser.add_argument('--mixup_pretrain', action='store_true', help="If true, use mixup pretraining data")
    parser.add_argument('--enable_adaptive_weights', action='store_true', help="If true, enable adaptive weight")

    # Cross module structure
    parser.add_argument('--cross_num_attention_heads', default=1, type=int, help='cross module attention head number')
    # parser.add_argument('--cross_vis_hidden_size', default=256, type=int, help='cross module hidden size')
    parser.add_argument('--cross_vis_hidden_size', default=512, type=int, help='cross module hidden size')
    # parser.add_argument('--cross_text_hidden_size', default=768, type=int, help='cross module hidden size')
    parser.add_argument('--cross_text_hidden_size', default=512, type=int, help='cross module hidden size')
    parser.add_argument('--cross_hidden_dropout_prob', default=0.1, type=float, help='cross module hidden dropout probability')
    parser.add_argument('--cross_attention_probs_dropout_prob', default=0.1, type=float)

    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--clip_model', default='', type=str, help='clip model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if (args.model == "ViT-L/14" or args.model == "ViT-L/14@336px"):
        args.vl_hidden_dim = 768

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('### INFO ### torch.backends.cudnn.benchmark = {}'.format(torch.backends.cudnn.benchmark))

    # build model
    model = build_model(args)
    model.to(device)

    close_clip_param_update = False
    if close_clip_param_update:
        for name, param in model.clip.named_parameters():
            param.requires_grad_(False)

    n_parameters_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of requires_grad params: ', n_parameters_grad)
    print('number of all params: ', n_parameters)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    rest_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]

    param_list = [{"params": rest_param, "lr": args.lr}]

    # using RMSProp or AdamW
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError('Lr scheduler type not supported ')

    # using polynomial lr scheduler or half decay every 10 epochs or step
    if args.lr_scheduler == 'poly':
        lr_func = lambda epoch: (1 - epoch / args.epochs) ** args.lr_power
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'halfdecay':
        lr_func = lambda epoch: 0.5 ** (epoch // (args.epochs // 10))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'cosine':
        lr_func = lambda epoch: 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'exponential':
        lr_func = lambda epoch: args.lr_exponential ** epoch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    # build dataset
    print('build dataset...')
    if (args.sup_type == 'full'):
        print("perform fullly supervised setting.")
        dataset_train = build_dataset('train', args)
    else:  # unsupervised
        dataset_train = build_dataset('train_pseudo', args)

    dataset_val = build_dataset('val', args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.clip_model != "":
        checkpoint = torch.load(args.clip_model, map_location='cpu')
        print("\nmodel structures: \n", model_without_ddp.clip)
        missing_keys, unexpected_keys = model_without_ddp.clip.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys when loading lora fine-tuned clip model:')
        print(missing_keys)
        print('Unexpected additional keys when loading lora fine-tuned clip model:')
        print(unexpected_keys)

    best_accu = 0
    if args.hi_lora_stage and not args.resume:
        print("hi_lora_stage, load last stage model")
        checkpoint = torch.load(args.hi_lora_retrain, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        # print('Missing keys when loading stage model: \n', missing_keys)
        # print('Unexpected additional keys when loading stage model: \n', unexpected_keys)
        print("hi_lora_stage, load clip model")
        checkpoint = torch.load(args.hi_lora_clip, map_location='cpu')
        # TODO: In the new HiLoRA stage, the CLIP model has changed, and the CLIP parameters from the previous stage
        #  can no longer be loaded. They need to be loaded separately
        missing_keys, unexpected_keys = model_without_ddp.clip.model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys when loading lora fine-tuned clip model:')
        print(missing_keys)
        print('Unexpected additional keys when loading lora fine-tuned clip model:')
        print(unexpected_keys)
        val_stats = validate(args, model, data_loader_val, device)
        best_accu = val_stats['accu']
        print("best_accu: ", best_accu)
        # Prevent negative optimization
        checkpoint_path = os.path.join(args.output_dir, 'best_checkpoint.pth')
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': -1,
            'args': args,
            'val_accu': val_stats['accu']
        }, checkpoint_path)
        clip_checkpoint_path = os.path.join(args.output_dir, "clip_lora_stage_with_bridge.pth")
        utils.save_on_master({"model": model_without_ddp.clip.state_dict()}, clip_checkpoint_path)
        print("HiLoRA CLIP checkpoint saved!")

    if args.retrain:
        checkpoint = torch.load(args.retrain, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        val_stats = validate(args, model, data_loader_val, device)
        best_accu = val_stats['accu']
        print("best_accu: ", best_accu)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            # args.start_epoch = 0  # 微调训练
        val_stats = validate(args, model, data_loader_val, device)
        best_accu = val_stats['accu']
        print("best_accu: {}".format(best_accu))

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(str(args) + "\n")

    print("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        start_ep_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(args, model, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        val_stats = validate(args, model, data_loader_val, device)

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'validation_{k}': v for k, v in val_stats.items()},
                     'n_parameters': n_parameters}
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [os.path.join(args.output_dir, 'checkpoint.pth')]
            # extra checkpoint before LR drop and every 10 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                # checkpoint_paths.append(os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            if val_stats['accu'] > best_accu:
                checkpoint_paths.append(os.path.join(args.output_dir, 'best_checkpoint.pth'))
                best_accu = val_stats['accu']
                if args.save_hilora_clip:
                    clip_checkpoint_path = os.path.join(args.output_dir, "clip_lora_stage_with_bridge.pth")
                    utils.save_on_master({"model": model_without_ddp.clip.state_dict()}, clip_checkpoint_path)
                    print("HiLoRA CLIP checkpoint saved!")

            for checkpoint_path in checkpoint_paths:
                print('Checkpoint is saving to: ', str(checkpoint_path))
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu']
                }, checkpoint_path)
            print('Checkpoints have been saved!')

        end_ep_time = time.time()
        total_time = end_ep_time - start_ep_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Current epoch training time {}'.format(total_time_str))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HiVG training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
