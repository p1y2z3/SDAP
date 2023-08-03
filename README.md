# 【ICCV 2023】 Random Sub-Samples Generation for Self-Supervised Real Image Denoising

### Yizhong Pan, Xiao Liu, Xiangyu Liao, Yuanzhouhan Cao, Chao Ren

[![paper](https://img.shields.io/badge/arXiv-Paper-green_yellow)](https://arxiv.org/abs/2307.16825)

This is the official code of SDAP: Random Sub-Samples Generation for Self-Supervised Real Image Denoising.

![main_fig](./figs/main.png)

# More details coming soon.

## Abstract
With sufficient paired training samples, the supervised deep learning methods have attracted much attention in image denoising because of their superior performance. However, it is still very challenging to widely utilize the supervised methods in real cases due to the lack of paired noisy-clean images. Meanwhile, most self-supervised denoising methods are ineffective as well when applied to the real-world denoising tasks because of their strict assumptions in applications. For example, as a typical method for self-supervised denoising, the original blind spot network (BSN) assumes that the noise is pixel-wise independent, which is much different from the real cases. To solve this problem, we propose a novel self-supervised real image denoising framework named Sampling Difference As Perturbation (SDAP) based on Random Sub-samples Generation (RSG) with a cyclic sample difference loss. Specifically, we dig deeper into the properties of BSN to make it more suitable for real noise. Surprisingly, we find that adding an appropriate perturbation to the training images can effectively improve the performance of BSN. Further, we propose that the sampling difference can be considered as perturbation to achieve better results. Finally we propose a new BSN framework in combination with our RSG strategy. The results show that it significantly outperforms other state-of-the-art self-supervised denoising methods on real-world datasets.

## Requirements
Our experiments are done with:

- Python 3.7.9
- PyTorch 1.7.1
- numpy 1.19.5
- opencv 4.5.1
- scikit-image 0.17.2

## Test
You can get the complete dataset from https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php.

run `test.py`

## Citation

    @inproceedings{pan2023random,
      title={Random Sub-Samples Generation for Self-Supervised Real Image Denoising}, 
      author={Yizhong Pan and Xiao Liu and Xiangyu Liao and Yuanzhouhan Cao and Chao Ren},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      year={2023}
    }

## Contact
If you have any questions, please contact p1y2z3@163.com.


## Acknowledgment
The codes are based on [AP-BSN](https://github.com/wooseoklee4/AP-BSN) and [Neighbor2Neighbor](https://github.com/TaoHuang2018/Neighbor2Neighbor). Thanks for their awesome works
