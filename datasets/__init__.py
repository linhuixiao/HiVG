# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torchvision.transforms import Compose, ToTensor, Normalize

import datasets.transforms as T
from .data_loader import TransVGDataset

""""CLIP's default transform"""
# def _transform(n_px):
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])


def make_transforms(args, image_set, is_onestage=False):
    if is_onestage:
        normalize = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return normalize

    imsize = args.imsize

    if image_set in ['train', 'train_pseudo']:
        scales = []
        if args.aug_scale:
            for i in range(7):
                scales.append(imsize - 32 * i)
        else:
            scales = [imsize]

        if args.aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.

        # By default, RandomResize sets with_long_side = True. The difference lies in whether the resizing is based on
        # the long side or the short side. Ultimately, the entire image needs to be compressed.
        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600], with_long_side=False),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.GaussianBlur(aug_blur=args.aug_blur),
            # T.RandomHorizontalFlip(),  # It has a certain impact on grounding performance and needs to be turned off
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize, aug_translate=args.aug_translate)
        ])

    if image_set in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize),
        ])

    raise ValueError(f'unknown {image_set}')


# args.data_root default='./ln_data/', args.split_root default='data', '--dataset', default='referit'
# split = test, testA, val, args.max_query_len = 20
def build_dataset(split, args):
    return TransVGDataset(args,
                          data_root=args.data_root,
                          split_root=args.split_root,
                          dataset=args.dataset,
                          split=split,
                          transform=make_transforms(args, split),
                          max_query_len=args.max_query_len,
                          prompt_template=args.prompt)
