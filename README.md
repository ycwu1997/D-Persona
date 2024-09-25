# Diversified and Personalized Multi-rater Medical Image Segmentation
by Yicheng Wu*+, Xiangde Luo+, Zhe Xu, Xiaoqing Guo, Lie Ju, Zongyuan Ge, Wenjun Liao and Jianfei Cai.

### News
```
<18.06.2024> We release the NPC170 dataset on the MMIS-2024 Grand Challenge;
```
```
<04.05.2024> The paper is selected as a Poster (Highlight, top 15%) in CVPR 2024;
```
```
<19.03.2024> We released the codes;
```
```
<27.02.2024> The paper is accepted by CVPR 2024;
```
### Introduction

![](assets/poster.png)

This repository is for our paper: "[Diversified and Personalized Multi-rater Medical Image Segmentation](https://doi.org/10.1109/CVPR52733.2024.01090)", the video introduction can be found at [YouTube](https://www.youtube.com/watch?v=5sFQVk_AkpE) platform.

Here, we study the inherent annotation ambiguity problem in medical image segmentation and use two datasets for the model evaluation (the public [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) and our in-house NPC-170 datasets). 

For the LIDC-IDRI dataset, we use its pre-processed version as [MedicalMatting](https://doi.org/10.1007/978-3-030-87199-4_54). For the NPC-170 dataset, we now open the [MMIS-2024](https://mmis2024.com/) challenge in ACM MM 2024. Please check it for more details.

### Requirements
This repository is based on PyTorch 2.0.1+cu118 and Python 3.11.4; All experiments in our paper were conducted on a single NVIDIA GeForce 3090 GPU.

### Usage
1. Clone this repo.;
```
git clone https://github.com/ycwu1997/D-Persona.git
```
2. Put the data into "./dataset";

3. First-stage training;
```
cd ./D-Persona/code
# e.g., the LIDC-IDRI dataset
python train_dp.py --stage 1 --val_num 10 --gpu 0
```
4. Put the first-stage weights into the "../code/";
```
cp ../models/[YOUR_MODEL_PATH]/DPersona1_LIDC_[IDX]_best.pth ../code/
```

6. Second-stage training;
```
python train_dp.py --stage 2 --val_num 100 --gpu 0
```
6. Test the model;
```
# e.g., first-stage performance on the LIDC-IDRI dataset
Python evaluate_dp.py --stage 1 --save_path ../models/[YOUR_MODEL_PATH] --test_num 50
# e.g., second-stage performance
Python evaluate_dp.py --stage 2 --save_path ../models/[YOUR_MODEL_PATH] --test_num 500
```

### Citation
If our D-Persona model is useful for your research, please consider citing:

        @InProceedings{Wu_2024_CVPR,
            author    = {Wu, Yicheng and Luo, Xiangde and Xu, Zhe and Guo, Xiaoqing and Ju, Lie and Ge, Zongyuan and Liao, Wenjun and Cai, Jianfei},
            title     = {Diversified and Personalized Multi-rater Medical Image Segmentation},
            booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
            month     = {June},
            year      = {2024},
            pages     = {11470-11479}
        }

### Acknowledgements:
Our code is adapted from [Pionono](https://github.com/arneschmidt/pionono_segmentation), [MedicalMatting](https://github.com/wangsssky/MedicalMatting), and [Prob. U-Net](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

### Questions
If any questions, feel free to contact me at 'ycwueli@gmail.com'
