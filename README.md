# Diversified and Personalized Multi-rater Medical Image Segmentation
by Yicheng Wu*+, Xiangde Luo+, Zhe Xu, Xiaoqing Guo, Lie Ju, Zongyuan Ge, Wenjun Liao and Jianfei Cai.

### News
```
<19.03.2024> We released the codes;
```
```
<27.02.2024> The paper is accepted by CVPR 2024;
```
### Introduction
This repository is for our paper: Diversified and Personalized Multi-rater Medical Image Segmentation. We studied the inherent annotation ambiguity problem in medical image segmentation.

### Requirements
This repository is based on PyTorch 2.0.1+cu118 and Python 3.11.4; All experiments in our paper were conducted on a single NVIDIA GeForce 3090 GPU.

### Usage
1. Clone the repo.;
```
git clone https://github.com/ycwu1997/D-Persona.git
```
2. Put the data in './DPersona/data';

3. Train the model;
```
cd code
# e.g., First stage on the LIDC-IDRI dataset
Python train_dp.py --stage 1
# put the weights DPersona1_LIDC_[fold]_best.pth from ../models/[YOUR_LOCAL_PATH]/ to ./
# Training in the second stage
Python train_dp.py --stage 2
```

4. Test the model;
```
# e.g., Evaluate the first stage performance on the LIDC-IDRI dataset
Python evaluate_dp.py --stage 1 --save_path ../models/[YOUR_LOCAL_PATH] --test_num 50
# e.g., Evaluate the second stage performance
Python evaluate_dp.py --stage 2 --save_path ../models/[YOUR_LOCAL_PATH] --test_num 500
```

### Citation
If our D-Persona model is useful for your research, please consider citing:

      @inproceedings{wu2024dpersona,
        title={Diversified and Personalized Multi-rater Medical Image Segmentation},
        author={Wu, Yicheng and Luo, Xiangde and Xu, Zhe, and Guo, Xiaoqing and Ju, Lie and Ge, Zongyuan and Liao, Wenjuan and Cai, Jianfei},
        booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
        year={2024},
        organization={IEEE}
        }

### Acknowledgements:
Our code is adapted from [Pionono](https://github.com/arneschmidt/pionono_segmentation), [MedicalMatting](https://github.com/wangsssky/MedicalMatting), [DTC](https://github.com/HiLab-git/DTC), and [Prob. U-Net](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

### Questions
If any questions, feel free to contact me at 'ycwueli@gmail.com'
