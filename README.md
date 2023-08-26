<div align="center">
  
# 【ICCV'2023 🔥】DiffusionRet: Generative Text-Video Retrieval with Diffusion Model
[![Conference](http://img.shields.io/badge/ICCV-2023-FFD93D.svg)](https://iccv2023.thecvf.com/)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2303.09867-FF6B6B.svg)](https://arxiv.org/abs/2303.09867)
</div>

The implementation of the paper [DiffusionRet: Generative Text-Video Retrieval with Diffusion Model](https://arxiv.org/abs/2303.09867).

In this paper, we propose a novel diffusion-based text-video retrieval framework, called DiffusionRet, which addresses the limitations of current discriminative solutions
from a generative perspective.

## 📌 Citation
If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:
```
@article{jin2023diffusionret,
  title={Diffusionret: Generative text-video retrieval with diffusion model},
  author={Jin, Peng and Li, Hao and Cheng, Zesen and Li, Kehan and Ji, Xiangyang and Liu, Chang and Yuan, Li and Chen, Jie},
  journal={arXiv preprint arXiv:2303.09867},
  year={2023}
}
```

## 📣 Updates
* Aug 27 2023: We release the training code.
* Jul 14 2023: Our **DiffusionRet** has been accepted by ICCV 2023! We will release the training code asap. 
* Jun 28 2023: We release the inference code.
* May 31 2023: Our paper is under review. After our paper is published, we will release the code as soon as possible.

## 📕 Overview
Existing text-video retrieval solutions are, in essence, discriminant models focused on maximizing the conditional likelihood, i.e., p(candidates|query). While straightforward, this de facto paradigm overlooks the underlying data distribution p(query), which makes it challenging to identify out-of-distribution data. To address this limitation, we creatively tackle this task from a generative viewpoint and model the correlation between the text and the video as their joint probability p(candidates,query). This is accomplished through a diffusion-based text-video retrieval framework (DiffusionRet), which models the retrieval task as a process of gradually generating joint distribution from noise.

<div align="center">
<img src="pictures/fig.png" width="600px">
</div>

## 🚀 Quick Start
### Setup

#### Setup code environment
```shell
conda create -n DiffusionRet python=3.9
conda activate DiffusionRet
pip install -r requirements.txt
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Download CLIP Model
```shell
cd DiffusionRet/models
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
# wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
# wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

#### Download Datasets
<div align=center>

|Datasets|Google Cloud|Baidu Yun|Peking University Yun|
|:--------:|:--------------:|:-----------:|:-----------:|
| MSR-VTT | [Download](https://drive.google.com/drive/folders/1LYVUCPRxpKMRjCSfB_Gz-ugQa88FqDu_?usp=sharing) | [Download](https://pan.baidu.com/s/1Gdf6ivybZkpua5z1HsCWRA?pwd=enav) | [Download](https://disk.pku.edu.cn:443/link/BE39AF93BE1882FF987BAC900202B266) |
| MSVD | [Download](https://drive.google.com/drive/folders/18EXLWvCCQMRBd7-n6uznBUHdP4uC6Q15?usp=sharing) | [Download](https://pan.baidu.com/s/1hApFdxgV3TV2TCcnM_yBiA?pwd=kbfi) | [Download](https://disk.pku.edu.cn:443/link/CC02BD15907BFFF63E5AAE4BF353A202) |
| ActivityNet | TODO | [Download](https://pan.baidu.com/s/1tI441VGvN3In7pcvss0grg?pwd=2ddy) | [Download](https://disk.pku.edu.cn:443/link/83351ABDAEA4A17A5A139B799BB524AC) |
| DiDeMo | TODO | [Download](https://pan.baidu.com/s/1Tsy9nb1hWzeXaZ4xr7qoTg?pwd=c842) | [Download](https://disk.pku.edu.cn:443/link/BBF9F5990FC4D7FD5EA9777C32901E62) |

</div>

### Model Zoo
<div align=center>

|Checkpoint|Google Cloud|Baidu Yun|Peking University Yun|
|:--------:|:--------------:|:-----------:|:-----------:|
| MSR-VTT | [Download](https://drive.google.com/file/d/16eTeXS9EZnBWP8EcO00Jxi6ZwsIUUHW_/view?usp=sharing) | [Download](https://pan.baidu.com/s/1JVxwh5SxnE0rGcAe9dCP_g?pwd=3xzi) | [Download](https://disk.pku.edu.cn:443/link/989708CFB90C80B93F7297A5260F5582) |
| ActivityNet | [Download](https://drive.google.com/file/d/1KcajRwDJMNxSWrlgLGHJ4nFtwgv0UWdc/view?usp=drive_link) | [Download](https://pan.baidu.com/s/1Nn-jUCJcydXhB01CNRvsfg?pwd=qsbd) | [Download](https://disk.pku.edu.cn:443/link/0E4384D13EA6E1693EF400FC27053033) |

</div>

### Evaluate
#### Eval on MSR-VTT
```shell
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=1 \
eval.py \
--workers 8 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--diffusion_steps 50 \
--noise_schedule cosine \
--init_model ${CHECKPOINT_PATH} \
--output_dir ${OUTPUT_PATH}
```

#### Eval on ActivityNet Captions
```shell
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=2 \
eval.py \
--workers 8 \
--batch_size_val 128 \
--anno_path ${DATA_PATH}/ActivityNet \
--video_path ${DATA_PATH}/ActivityNet/Activity_Videos \
--datatype activity \
--max_words 64 \
--max_frames 64 \
--video_framerate 1 \
--diffusion_steps 50 \
--noise_schedule cosine \
--init_model ${CHECKPOINT_PATH} \
--output_dir ${OUTPUT_PATH}
```

### Train
#### Discrimination Pretrain 
Train the feature extractor from the discrimination perspective.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=4 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--stage discrimination \
--output_dir ${OUTPUT_PATH}
```

#### Generation Finetune
Optimize the generator from the generation perspective.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=4 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path data/MSR-VTT/anns \
--video_path ${DATA_PATH}/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--stage generation \
--diffusion_steps 50 \
--noise_schedule cosine \
--init_model ${CHECKPOINT_PATH} \
--output_dir ${OUTPUT_PATH}
```


## 🎗️ Acknowledgments
Our code is based on [EMCL](https://github.com/jpthu17/EMCL), [CLIP](https://github.com/openai/CLIP), [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/) and [DRL](https://github.com/foolwood/DRL). We sincerely appreciate for their contributions.

