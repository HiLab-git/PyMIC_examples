# Semi-supervised demo using PyMIC

In this example, we show semi-supervised learning methods implemented in PyMIC.
Currently, the following semi-supervised methods are implemented:
|PyMIC Method|Reference|Remarks|
|---|---|---|
|SSLEntropyMinimization|[Grandvalet et al.][em_paper], NeurIPS 2005| Oringinally proposed for classification|
|SSLMeanTeacher| [Tarvainen et al.][mt_paper], NeurIPS 2017| Oringinally proposed for classification|
|SSLUAMT| [Yu et al.][uamt_paper], MICCAI 2019| Uncertainty-aware mean teacher|
|SSLURPC| [Luo et al.][urpc_paper], MedIA 2022| Uncertainty rectified pyramid consistency|
|SSLCCT| [Ouali et al.][cct_paper], CVPR 2020| Cross-pseudo supervision|
|SSLCPS| [Chen et al.][cps_paper], CVPR 2021| Cross-consistency training|

[em_paper]:https://papers.nips.cc/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf
[mt_paper]:https://arxiv.org/abs/1703.01780
[uamt_paper]:https://arxiv.org/abs/1907.07034 
[urpc_paper]:https://doi.org/10.1016/j.media.2022.102517
[cct_paper]:https://arxiv.org/abs/2003.09005 
[cps_paper]:https://arxiv.org/abs/2106.01226 


## Data 
The [ACDC][ACDC_link] (Automatic Cardiac Diagnosis Challenge) dataset is used in this demo. It contains 200 short-axis cardiac cine MR images of 100 patients, and the classes for segmentation are: Right Ventricle (RV), Myocardiym (Myo) and Left Ventricle (LV). The images are available in `PyMIC_data/ACDC/preprocess`, where we have normalized the intensity to [0, 1]. You can download `PyMIC_data` from .... The images are split at patient level into 70%, 10% and 20% for training, validation  and testing, respectively (see `config/datas` for details).

In the training set, we randomly select 14 images of 7 patients as annotated images and the other 126 images as unannotated images. See `random_split_train.py`. 

[ACDC_link]:https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

## Training
In this demo, we experiment with five methods: EM, UAMT, UPRC, CCT and CPS, and they are compared with the baseline of learning from annotated images. All these methods use UNet2D as the backbone network.

### Baseline Method
The baseline method uses the 14 annotated cases for training. The batch size is 4, and the patch size is 6x192x192. Therefore, indeed there are 2D 16 slices in each batch. See `config/unet2d_baseline.cfg` for details about the configuration.You need to set `root_dir` to your own `PyMIC_data/ACDC/preprocess`. The dataset configration is:

```bash
tensor_type = float
task_type = seg
root_dir  = /home/disk2t/projects/PyMIC_project/PyMIC_data/ACDC/preprocess/
train_csv = config/data/image_train_r10_lab.csv
valid_csv = config/data/image_valid.csv
test_csv  = config/data/image_test.csv
train_batch_size = 4
```

For data augmentation, we use random rotate, random crop, random flip, gamma correction and gaussian noise. The images cropped images are also normaized with mean and std. The details for data transforms are:

```bash
train_transform = [Pad, RandomRotate, RandomCrop, RandomFlip, NormalizeWithMeanStd, GammaCorrection, GaussianNoise, LabelToProbability]
valid_transform       = [NormalizeWithMeanStd, Pad, LabelToProbability]
test_transform        = [NormalizeWithMeanStd, Pad]

Pad_output_size = [8, 256, 256]
Pad_ceil_mode   = False

RandomRotate_angle_range_d = [-90, 90]
RandomRotate_angle_range_h = None
RandomRotate_angle_range_w = None

RandomCrop_output_size = [6, 192, 192]
RandomCrop_foreground_focus = False
RandomCrop_foreground_ratio = None
Randomcrop_mask_label       = None

RandomFlip_flip_depth  = False
RandomFlip_flip_height = True
RandomFlip_flip_width  = True

NormalizeWithMeanStd_channels = [0]

GammaCorrection_channels  = [0]
GammaCorrection_gamma_min = 0.7
GammaCorrection_gamma_max = 1.5

GaussianNoise_channels = [0]
GaussianNoise_mean     = 0
GaussianNoise_std      = 0.05
GaussianNoise_probability = 0.5
```

The configuration of 2D UNet is:
```bash
net_type = UNet2D
class_num     = 4
in_chns       = 1
feature_chns  = [16, 32, 64, 128, 256]
dropout       = [0.0, 0.0, 0.0, 0.5, 0.5]
bilinear      = True
deep_supervise= False
```

For training, we use a combinatin of DiceLoss and CrossEntropyLoss, and train the network by the   `Adam` optimizer. The maximal iteration is 30k, and the training is early stopped if there is not performance improvement on the validation set for 10k iteratins. The learning rate scheduler is `ReduceLROnPlateau`. The corresponding configuration is:
```bash
gpus       = [0]
loss_type     = [DiceLoss, CrossEntropyLoss]
loss_weight   = [0.5, 0.5]

optimizer     = Adam
learning_rate = 1e-3
momentum      = 0.9
weight_decay  = 1e-5


lr_scheduler  = ReduceLROnPlateau
lr_gamma      = 0.5
ReduceLROnPlateau_patience = 4000
early_stop_patience = 10000

ckpt_save_dir    = model/unet2d_baseline

iter_start = 0
iter_max   = 30000
iter_valid = 100
iter_save  = [30000]
```

During inference, we use a sliding window of 6x192x192, and post process the results by `KeepLargestComponent`. The configuration is:
```bash
# checkpoint mode can be [0-latest, 1-best, 2-specified]
ckpt_mode         = 1
output_dir        = result/unet2d_baseline
post_process      = KeepLargestComponent

sliding_window_enable = True
sliding_window_size   = [6, 192, 192]
sliding_window_stride = [6, 192, 192]
```

To train the mode, run `pymic_run train config/unet2d_baseline.cfg`.
For inference, run `pymic_run test config/unet2d_baseline.cfg`.

1. Edit `config/unet.cfg` by setting the value of `root_dir` as your `HC_root`. Then start to train by running:
 
```bash
pymic_run train config/unet.cfg
```

2. During training or after training, run `tensorboard --logdir model/unet` and you will see a link in the output, such as `http://your-computer:6006`. Open the link in the browser and you can observe the average Dice score and loss during the training stage, such as shown in the following images, where red and blue curves are for training set and validation set respectively. 

![avg_dice](./picture/train_avg_dice.png)
![avg_loss](./picture/train_avg_loss.png)

## Testing and evaluation
1. Run the following command to obtain segmentation results of testing images based on the best-performing checkpoint on the validation set. By default we use sliding window inference to get better results. You can also edit the `testing` section of `config/unet.cfg` to use other inference strategies.

```bash
pymic_run test config/unet.cfg
```

2. Then edit `config/evaluation.cfg` by setting `ground_truth_folder_root` as your `HC_root`, and run the following command to obtain quantitative evaluation results in terms of Dice. 

```bash
pymic_eval_seg config/evaluation.cfg
```

The obtained average Dice score by default setting should be close to 97.07%. You can set `metric = assd` in `config/evaluation.cfg` and run the evaluation command again to get Average Symmetric Surface Distance (ASSD) evaluation results. 

3. Set `tta_mode = 1` in `config/unet.cfg` to enable test time augmentation, and run the testing and evaluation code again, we find that the average Dice will be increased to around 97.22%.

[PyMIC][PyMIC_link] is an Pytorch-based medical image computing toolkit with deep learning. Here we provide a set of examples to show how it can be used for image classification and segmentation tasks. For beginners, you can follow the examples by just editting the configure files for model training, testing and evaluation. For advanced users, you can develop your own modules, such as customized networks and loss functions.  

## Install PyMIC
The latest released version of PyMIC can be installed by:

```bash
pip install PYMIC==0.2.4
```

To use the latest development version, you can download the source code [here][PyMIC_link], and install it by:

```bash
python setup.py install
``` 
