# 【ICCV 2023】 Random Sub-Samples Generation for Self-Supervised Real Image Denoising

### Yizhong Pan, Xiao Liu, Xiangyu Liao, Yuanzhouhan Cao, Chao Ren

[![paper](https://img.shields.io/badge/arXiv-Paper-green_yellow)](https://arxiv.org/abs/2307.16825) [![iccv_paper](https://img.shields.io/badge/ICCV-Paper-blue)](https://openaccess.thecvf.com/content/ICCV2023/html/Pan_Random_Sub-Samples_Generation_for_Self-Supervised_Real_Image_Denoising_ICCV_2023_paper.html)

This is the official code of SDAP: Random Sub-Samples Generation for Self-Supervised Real Image Denoising.

![main_fig](./figs/main.png)


## Abstract
With sufficient paired training samples, the supervised deep learning methods have attracted much attention in image denoising because of their superior performance. However, it is still very challenging to widely utilize the supervised methods in real cases due to the lack of paired noisy-clean images. Meanwhile, most self-supervised denoising methods are ineffective as well when applied to the real-world denoising tasks because of their strict assumptions in applications. For example, as a typical method for self-supervised denoising, the original blind spot network (BSN) assumes that the noise is pixel-wise independent, which is much different from the real cases. To solve this problem, we propose a novel self-supervised real image denoising framework named Sampling Difference As Perturbation (SDAP) based on Random Sub-samples Generation (RSG) with a cyclic sample difference loss. Specifically, we dig deeper into the properties of BSN to make it more suitable for real noise. Surprisingly, we find that adding an appropriate perturbation to the training images can effectively improve the performance of BSN. Further, we propose that the sampling difference can be considered as perturbation to achieve better results. Finally we propose a new BSN framework in combination with our RSG strategy. The results show that it significantly outperforms other state-of-the-art self-supervised denoising methods on real-world datasets.

## Requirements
Our experiments are done with:

- Python 3.7.9
- PyTorch 1.7.1
- numpy 1.19.5
- opencv 4.5.1
- scikit-image 0.17.2

## Pre-trained Models

|   Training Dataset   |  Model  |
| :-------------------------------------: | :-----------------------------: |
|        [SIDD Medium](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)      | SDAP.pth |
|        [SIDD Medium](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) + [SIDD Benchmark](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php)    | SDAP_S_for_SIDD.pth |
|        [SIDD Medium](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php) + [DND Benchmark](https://noise.visinf.tu-darmstadt.de/downloads/)      | SDAP_S_for_DND.pth |

## Test
You can get the complete SIDD validation dataset from https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php.

'.mat' files need to be converted to images ('.png'). 

run `test.py`to output the denoising results of our proposed method.

## Train
'.mat' files in training datasets need to be converted to images ('.png').

You have to add the noisy images ('.png') for training in './dataset/train_data'.

Tip: Reducing the size of the training image speeds up the reading of the dataset thus speeding up training. So you can crop the image to a fixed size (e.g. 512*512) in advance.

run `train.py`.

## Citation

    @inproceedings{Pan_2023_ICCV,
    author    = {Pan, Yizhong and Liu, Xiao and Liao, Xiangyu and Cao, Yuanzhouhan and Ren, Chao},
    title     = {Random Sub-Samples Generation for Self-Supervised Real Image Denoising},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {12150-12159}
    }

## Contact
If you have any questions, please contact p1y2z3@163.com.


## Acknowledgment
The codes are based on [AP-BSN](https://github.com/wooseoklee4/AP-BSN) and [Neighbor2Neighbor](https://github.com/TaoHuang2018/Neighbor2Neighbor). Thanks for their awesome works.
