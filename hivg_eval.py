import time
import json
import random
import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import evaluate

import os


os.environ["CUDA_VISIBLE_DEVICES"] = '7'


def get_args_parser():
    parser = argparse.ArgumentParser('HiVG Evaluate args', add_help=False)
    parser.add_argument('--sup_type', default='full', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)  # 融合模块学习率
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float, help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--lr_drop', default=80, type=int)

    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true', help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true', help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true', help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true', help="If true, use random translate augmentation")
    # only support ViT-B/16 and ViT-L/14
    parser.add_argument('--model', type=str, default='ViT-B/16', help="Name of model to be exploited.")
    # Model parameters
    parser.add_argument('--model_name', type=str, default='CLIP-VG', help="Name of model to be exploited.")
    parser.add_argument('--extract_layer', default=0, type=int)
    parser.add_argument('--warmup', action='store_true', help="If true, vision adapt layer is null")

    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--imsize', default=224, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int, help='fusion module embedding dimensions')
    # Vision-Language Transformer
    parser.add_argument('--use_vl_type_embed', action='store_true', help="If true, use vl_type embedding")
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=512, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int, help='Number of encoders in the vision-language transformer')
    parser.add_argument('--vl_dec_layers', default=6, type=int,
                        help='Number of decoders in the vision-language transformer')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./data/image_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='./data/pseudo_samples/',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str, help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--max_query_len', default=77, type=int,
                        help='maximum time steps (lang length) per batch')

    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default='./saved_models/detr-r50.pth', type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # evalutaion options
    parser.add_argument('--eval_set', default='test', type=str)  # 'testA', 'testB', 'val'
    parser.add_argument('--eval_model', default='', type=str)

    parser.add_argument('--prompt', type=str, default='{pseudo_query}', help="Prompt template")
    parser.add_argument('--adapt_mlp', action='store_true', help="If true, use contrastive loss")
    parser.add_argument('--use_loss_coef', action='store_true', help="If true, use contrastive loss")
    parser.add_argument('--normalize_before', action='store_true', help="If true, use normalize_before")
    parser.add_argument('--save_hilora_clip', action='store_true', help="If true, save hilora clip model")
    parser.add_argument('--hi_lora_stage', default=0, type=int, help='lora stage')
    parser.add_argument('--use_seg_mask', action='store_true',
                        help="If true, use segmentation mask in segmentation task, otherwise use box mask.")
    parser.add_argument('--use_mask_loss', action='store_true', help="If true, use segmentation loss")
    parser.add_argument('--mixup_pretrain', action='store_true', help="If true, use mixup pretraining data")
    parser.add_argument('--enable_adaptive_weights', action='store_true', help="If true, enable adaptive weight")

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

    # build model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of all params: ', n_parameters)

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)

    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, args.batch_size, drop_last=False)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    # print('Missing keys when loading stage model: \n', missing_keys)
    # print('Unexpected additional keys when loading stage model: \n', unexpected_keys)

    if checkpoint['epoch']:
        print("Current model training epoch is: ", checkpoint['epoch'])

    # output log
    eval_model = args.eval_model
    eval_model = eval_model.split('/')[-1].split('.')[0]
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_{}_{}_{}_log.txt".format(args.dataset, args.eval_set, eval_model)).open("a") as f:
            f.write(str(args) + "\n")
            f.flush()

    start_time = time.time()

    # perform evaluation
    accuracy = evaluate(args, model, data_loader_test, device)

    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Testing time {}'.format(total_time_str))

        log_stats = {'test_model:': args.eval_model,
                     '%s_set_accuracy'%args.eval_set: accuracy,
                     }
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "eval_{}_{}_{}_log.txt".format(args.dataset, args.eval_set, eval_model)).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    if args.save_hilora_clip:
        clip_checkpoint_path = os.path.join(args.output_dir, "clip_lora_stage_with_bridge.pth")
        torch.save({"model": model_without_ddp.clip.state_dict()}, clip_checkpoint_path)
        print("HiLoRA CLIP checkpoint saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HiVG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
