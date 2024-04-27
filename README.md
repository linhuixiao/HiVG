# HiVG: Hierarchical Multimodal Fine-grained Modulation for Visual Grounding
<p align="center"> <img src='docs/model.jpg' align="center" width="55%"> </p>


This repository is the official Pytorch implementation for the paper [**HiVG: Hierarchical Multimodal Fine-grained 
Modulation for Visual Grounding**](https://arxiv.org/abs/2404.13400), by
[Linhui Xiao](https://scholar.google.com.hk/citations?user=4rTE4ogAAAAJ&hl=zh-CN&oi=sra), 
[Xiaoshan Yang](https://yangxs.ac.cn/home), 
[Fang Peng](Fang Peng), [Yaowei Wang](https://scholar.google.com.hk/citations?user=o_DllmIAAAAJ&hl=zh-CN), 
and [Changsheng Xu](https://scholar.google.com.hk/citations?user=hI9NRDkAAAAJ&hl=zh-CN), which is an advanced version
of our preliminary work **CLIP-VG** ([github](https://github.com/linhuixiao/CLIP-VG), [publication](
https://ieeexplore.ieee.org/abstract/document/10269126), [Arxiv](https://arxiv.org/abs/2305.08685)). 

If you have any questions, please feel free to open an issue or contact me with emails: <xiaolinhui16@mails.ucas.ac.cn>.
Any kind discussions are welcomed!


<h3 align="left">
Links: 
<a href="https://arxiv.org/abs/2404.13400">ArXiv</a>
</h3>

**Please leave a <font color='orange'>STAR ⭐</font> if you like this project!**

## News

- **All of the code and models will be released soon!**
- **Update on 2024/04/20: Release the project repository.**


## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@article{xiao2024hivg,
      title={HiVG: Hierarchical Multimodal Fine-grained Modulation for Visual Grounding}, 
      author={Linhui Xiao and Xiaoshan Yang and Fang Peng and Yaowei Wang and Changsheng Xu},
      year={2024},
      eprint={2404.13400},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)
5. [Acknowledgments](#acknowledgments)


## Highlight
-  **A concise hierarchical multimodal modulation framework**, which utilizes the hierarchical structure to gradually adapt CLIP to grounding. HiVG achieves fine-grained interaction between multi-level visual representations and language semantics, and significantly alleviates the task gap between CLIP and grounding.
- **The first to propose the hierarchical multimodal low-rank adaptation paradigm.** Hi LoRA is a basic and concise hierarchical adaptation paradigm, which is task-agnostic.
- **Extensive experiments are conducted to verify the effectiveness of HiVG approaches.** Results show that our method achieves promising results, surpassing the SOTA methods under the same setting by a significant margin. Besides, our model offers significant computing efficiency advantages.


## TODO
- [ ] Release all the checkpoints.
- [ ] Release the full model code, training and inference code.




## Introduction

Visual grounding, which aims to ground a visual region via natural language, is a task that heavily relies on cross-modal 
alignment. Existing works utilized uni-modal pre-trained models to transfer visual/linguistic knowledge separately while 
ignoring the multimodal corresponding information. Motivated by recent advancements in contrastive language-image 
pre-training and low-rank adaptation (LoRA) methods, we aim to solve the grounding task based on multimodal pre-training.
However, there exists significant task gaps between pre-training and grounding. Therefore, to address these gaps, we 
propose **a concise and efficient hierarchical multimodal fine-grained modulation framework**, namely **HiVG**. Specifically,
HiVG consists of a multi-layer adaptive cross-modal bridge and a hierarchical multimodal low-rank adaptation (Hi LoRA) 
paradigm. The cross-modal bridge can address the inconsistency between visual features and those required for grounding,
and establish a connection between multi-level visual and text features. Hi LoRA prevents the accumulation of perceptual 
errors by adapting the cross-modal features from shallow to deep layers in a hierarchical manner. Experimental results 
on five datasets demonstrate the effectiveness of our approach and showcase the significant grounding capabilities as well 
as promising energy efficiency advantages.

For more details, please refer to [our paper](https://arxiv.org/abs/2404.13400).

## Usage
### Dependencies
- Python 3.9.10
- PyTorch 1.9.0 + cu111 + cp39
- Check [requirements.txt](requirements.txt) for other dependencies. 

Our model is **easy to deploy** in a variety of environments and **has been successfully tested** on multiple pytorch versions.


### Image Data Preparation
1.You can download the images from the original source and place them in your disk folder, such as `$/path_to_image_data`:
- [MS COCO 2014](download_mscoco2014.sh) (for RefCOCO, RefCOCO+, RefCOCOg dataset, almost 13.0GB) 
- [ReferItGame](https://drive.google.com/drive/folders/1D4shieeoKly6FswpdjSpaOrxJQNKTyTv)
- [Flickr30K Entities](http://shannon.cs.illinois.edu/DenotationGraph/#:~:text=make%20face-,Downloads,-Please%20fill%20in)

   We provide a script to download the mscoco2014 dataset, you just need to run the script in terminal with the following command:
   ```
   bash download_mscoco2014.sh
   ```
   Or you can also follow the data preparation of TransVG, which can be found in [GETTING_STARTED.md](https://github.com/djiajunustc/TransVG/blob/main/docs/GETTING_STARTED.md).

Only the image data in these datasets is used, and these image data is easily find in similar repositories of visual grounding work, such as [TransVG](https://github.com/linhuixiao/TransVG) etc. 
Finally, the `$/path_to_image_data` folder will have the following structure:

```angular2html
|-- image_data
   |-- Flickr30k
      |-- flickr30k-images
   |-- other
      |-- images
        |-- mscoco
            |-- images
                |-- train2014
   |-- referit
      |-- images
```
- ```$/path_to_image_data/image_data/Flickr30k/flickr30k-images/```: Image data for the Flickr30K dataset, please download from this [link](http://shannon.cs.illinois.edu/DenotationGraph/#:~:text=make%20face-,Downloads,-Please%20fill%20in). Fill the form and download the images.
- ```$/path_to_image_data/image_data/other/images/```: Image data for RefCOCO/RefCOCO+/RefCOCOg, i.e., mscoco2014. 
- ```$/path_to_image_data/image_data/referit/images/```: Image data for ReferItGame.

## Text-Box Anotations 
The labels in the fully supervised scenario is consistent with previous works such as [TransVG](https://github.com/linhuixiao/TransVG).


### Fully supervised setting
<table>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Datasets </th>
    <th style="text-align:center" > RefCOCO </th>
    <th style="text-align:center" > RefCOCO+ </th>
    <th style="text-align:center" > RefCOCOg-g </th>
    <th style="text-align:center" > RefCOCOg-u </th>
    <th style="text-align:center" > ReferIt </th>
    <th style="text-align:center" > Flickr </th>
    </tr>
    <tr> <!-- line 2 -->
        <th style="text-align:center" rowspan="1"> url, size </th> <!-- table head -->
        <th style="text-align:center" colspan="6"> <a href="https://drive.google.com/file/d/1ituKSxWU5aXsGnXePd7twv7ImJoFiATc/view?usp=drive_link">All of six datasets</a>,  89.0MB </th>  <!-- table head -->
    </tr>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > with curriculum selecting </th>
    <th style="text-align:center" > - </th>
    <th style="text-align:center" > - </th>
    <th style="text-align:center" > - </th>
    <th style="text-align:center" > <a href="https://drive.google.com/file/d/1eSGr-sTqZ6z_Jy7APnJXNxegt2Q-pbqE/view?usp=drive_link">dataset</a> </th>
    <th style="text-align:center" > - </th>
    <th style="text-align:center" > - </th>
    </tr>
</table>

\* Since we observed a relatively clear performance increase on the RefCOCOg-u dataset in the fully supervised setting, 
we provide data for this dataset after applying our SSA algorithm for curriculum selecting. Typically, by using this 
filtered data, there is an approximate ~1.0 increase in performance on both val-u and test-u.

Download the above annotations to a disk directory such as `$/path_to_split`; then will have the following similar directory structure:

```angular2html
|-- /full_sup_data
    ├── flickr
    │   ├── flickr_test.pth
    │   ├── flickr_train.pth
    │   └── flickr_val.pth
    ├── gref
    │   ├── gref_train.pth
    │   └── gref_val.pth
    ├── gref_umd
    │   ├── gref_umd_test.pth
    │   ├── gref_umd_train.pth
    │   └── gref_umd_val.pth
    ├── referit
    │   ├── referit_test.pth
    │   ├── referit_train.pth
    │   └── referit_val.pth
    ├── unc
    │   ├── unc_testA.pth
    │   ├── unc_testB.pth
    │   ├── unc_train.pth
    │   └── unc_val.pth
    └── unc+
        ├── unc+_testA.pth
        ├── unc+_testB.pth
        ├── unc+_train.pth
        └── unc+_val.pth
```


## Pre-trained Checkpoints

### Fully supervised setting

<table>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > Datasets </th>
    <th style="text-align:center" > RefCOCO </th>
    <th style="text-align:center" > RefCOCO+ </th>
    <th style="text-align:center" > RefCOCOg-g </th>
    <th style="text-align:center" > RefCOCOg-u </th>
    <th style="text-align:center" > ReferIt </th>
    <th style="text-align:center" > Flickr </th>
    </tr>
    <tr> <!-- line 3 -->
    <th style="text-align:center" > separate </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    <th style="text-align:center" > <a href="todo">model</a> </th>
    </tr>
    <tr> <!-- line 2 -->
        <th style="text-align:center" rowspan="1"> url, size </th> <!-- table head -->
        <th style="text-align:center" colspan="6"> <a href="todo">All of six models (All have not ready)</a>  </th>  <!-- table head -->
    </tr>
</table>

The checkpoints include the Base model and Large mode under the fine-tuning setting and dataset-mixed pretraining setting. 


## Training and Evaluation

You just only need to change ```$/path_to_split```, ``` $/path_to_image_data```, ``` $/path_to_output``` to your own file directory to execute the following command.
The first time we run the command below, it will take some time for the repository to download the CLIP model.

1. Training on RefCOCO with fully supervised setting. 
    The only difference is an additional control flag: ```--sup_type full```
    ```
    CUDA_VISIBLE_DEVICES=3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=5 --master_port 28887 --use_env train_clip_vg.py --num_workers 32 --epochs 120 --batch_size 64 --lr 0.00025  --lr_scheduler cosine --aug_crop --aug_scale --aug_translate    --imsize 224 --max_query_len 77  --sup_type full --dataset unc      --data_root $/path_to_image_data --split_root $/path_to_split --output_dir $/path_to_output/output_v01/unc;
    ```
    Please refer to [train_and_eval_script/train_and_eval_full_sup.sh](train_and_eval_script/train_and_eval_full_sup.sh) for training commands on other datasets.

2. Evaluation on RefCOCO. The instructions are the same for the fully supervised Settings.
    ```
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port 28888 --use_env eval.py --num_workers 2 --batch_size 128    --dataset unc      --imsize 224 --max_query_len 77 --data_root $/path_to_image_data --split_root $/path_to_split --eval_model $/path_to_output/output_v01/unc/best_checkpoint.pth      --eval_set val    --output_dir $/path_to_output/output_v01/unc;
    ```
    Please refer to [train_and_eval_script/train_and_eval_unsup.sh](train_and_eval_script/train_and_eval_unsup.sh) for evaluation commands on other splits or datasets.
    
3. We strongly recommend to use the following commands to training or testing with different datasets and splits, 
    which will significant reduce the training workforce.
    ```
    bash train_and_eval_script/train_and_eval_full_sup.sh
    ```



## Results

### 1. RefCOCO, RefCOCO+, RefCOCOg, ReferIt, Flickr, datasets
<details open>
<summary><font size="4">
SOTA Result Table
</font></summary>
<img src="docs/sota.jpg" alt="COCO" width="100%">
</details>

**(1) When compared to the CLIP-based fine-tuning SOTA work**, i.e., Dynamic-MDETR, our approach consistently 
outperforms it by achieving an increase of 3.15%(testB), 3.11%(testA), 4.30%(test), 5.55%(test), 
0.22%(test) on all five datasets. 

**(2) When compared to the detector-based fine-tuning SOTA work**, i.e., 
TransVG++, our approach demonstrates superior performance (improved by 2.30%(testB), 4.36%(testA), 2.49%(test), 
1.22%(test), 0.62%(test)) across all five datasets. The improvement of our results on the RefCOCO+/g datasets is 
considerably more significant, indicating our model exhibits a stronger capacity for semantic comprehension in complex 
sentences. 

**(3) When compared with the dataset-mixed pre-training works**, the base model of our work outperforms 
Grounding-DINO by 1.24%(testB), 1.81%(testA), and 1.68%(testA) on the RefCOCO/+/g 
datasets, and it also outperforms OFA by 3.93%(testB), 2.06%(testA), and 4.31%(testA). 
After dataset-mixed pre-training, our performance has significantly improved, further demonstrating the effectiveness 
of our method.

### 2. Our model also has significant energy efficiency advantages.

<details open>
<summary><font size="4">
Illustration
</font></summary>
<div align=center>
<img src="docs/result_performance.jpg" alt="COCO" width="100%"></div>
</details>

**Comparison between HiVG (base) and SOTA models, as well as the ablation study of HiVG on the main modules.** (a) HiVG 
achieves significant energy efficiency advantages, **8.2x** faster than TransVG++ while
outperforming it on RefCOCO-val. (b) The computational complexity of HiVG is **only 13.0%** compared with 
TransVG++. (c) HiVG outperforms SOTA models in different expression lengths on RefCOCOg-test. (d) Hi LoRA method brings
significant performance gains to HiVG model.


## Methods 

<p align="center"> <img src='docs/motivation.jpg' align="center" width="60%"> </p>

**Visual attentions and grounding results of CLIP and the proposed HiVG.** The attentions are perceived by the 
[CLS] token over vision tokens.

<p align="center"> <img src='docs/hilora.jpg' align="center" width="60%"> </p>

**Hi LoRA and vanilla LoRA.** (a) The vanilla LoRA learns the global low-rank matrix utilizing the entire set of 
pre-trained weights in a single round. (b) The proposed Hi LoRA employs a hierarchical approach to adapt the pre-trained 
model in a progressive manner, thereby finely reducing the task gap between pre-training and transfer tasks.

## Visualization
<p align="center"> <img src='docs/visualization.jpg' align="center" width="70%"> </p>

 **Qualitative results of our HiVG framework on the RefCOCOg-val split.** The CLIP-VG model is compared. We present the
 prediction box with IoU (in cyan) and the ground truth box (in green) in a unified  image to visually display the 
 grounding accuracy. We show the [REG] token’s attention over vision tokens from the last 
 grounding block of each framework. The examples exhibit the relatively more challenging instances for grounding, thereby 
 showcasing HiVG's robust semantic comprehension capabilities.

## Contacts
Email: <xiaolinhui16@mails.ucas.ac.cn>.
Any kind discussions are welcomed!

## Acknowledgement

Our model is related to [CLIP](https://github.com/openai/CLIP), [CLIP-VG](https://github.com/linhuixiao/CLIP-VG). Thanks for their great work!

We also thank the great previous work including [TransVG++](https://github.com/linhuixiao/TransVG), 
[DETR](https://github.com/facebookresearch/detr), [QRNet](https://github.com/LukeForeverYoung/QRNet), etc. 

Thanks [OpenAI](https://github.com/openai) for their awesome models.









