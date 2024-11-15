# Semi-Supervised Left Atrial Segmentation using 3D CNNs

In this example, we show the following four semi-supervised learning methods for segmentation of left atrial from 3D medical images. All these methods use UNet3D as the backbone network.
|PyMIC Method|Reference|Remarks|
|---|---|---|
|SSLEntropyMinimization|[Grandvalet et al.][em_paper], NeurIPS 2005| Oringinally proposed for classification|
|SSLUAMT| [Yu et al.][uamt_paper], MICCAI 2019| Uncertainty-aware mean teacher|
|SSLURPC| [Luo et al.][urpc_paper], MedIA 2022| Uncertainty rectified pyramid consistency|
|SSLCPS| [Chen et al.][cps_paper], CVPR 2021| Cross-consistency training|

For a full list of available semi-supervised methods, see the [document][all_ssl_link].

[em_paper]:https://papers.nips.cc/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf
[mt_paper]:https://arxiv.org/abs/1703.01780
[uamt_paper]:https://arxiv.org/abs/1907.07034 
[urpc_paper]:https://doi.org/10.1016/j.media.2022.102517
[cct_paper]:https://arxiv.org/abs/2003.09005 
[cps_paper]:https://arxiv.org/abs/2106.01226 
[all_ssl_link]:https://pymic.readthedocs.io/en/latest/usage.ssl.html


## 1. Data 
The [Left Atrial][atrial_link] dataset is used in this demo, which is from the 2018 Atrial Segmentation Challenge. The original dataset contains 100 cases for training and 54 cases for testing.  As the official testing data are not publicly available, here we split the other 100 cases into 72 for training, 8 for validation and 20 for testing. The images are available in `PyMIC_data/AtriaSeg`. We have preprocessed the images by cropping the volume with an output size of  88 x 160 x 256. For semi-supervised learning, we set the annotation ratio to 10%, i.e., 7 images are annotated  and the other 65 images are unannotated in the training set. 

[atrial_link]:http://atriaseg2018.cardiacatlas.org/

## 2. Baseline Method
The baseline method uses the 7 annotated cases for training. The batch size is 2, and the patch size is 72x96x112. See `config/unet3d_r10_baseline.cfg` for details about the configuration. The dataset configuration is:

```bash
tensor_type    = float
task_type      = seg
supervise_type = fully_sup

train_dir = ../../PyMIC_data/AtriaSeg/TrainingSet_crop/
train_csv = config/data/image_train_r10_lab.csv
valid_csv = config/data/image_valid.csv
test_csv  = config/data/image_test.csv
train_batch_size = 2
```

For data augmentation, we use random crop, random flip, gamma correction and gaussian noise. The cropped images are also normaized with mean and std. The details for data transforms are:

```bash
train_transform = [RandomCrop, RandomFlip, NormalizeWithMeanStd, GammaCorrection, GaussianNoise, LabelToProbability]
valid_transform = [NormalizeWithMeanStd, LabelToProbability]
test_transform  = [NormalizeWithMeanStd]

RandomCrop_output_size = [72, 96, 112]
RandomCrop_foreground_focus = False
RandomCrop_foreground_ratio = None
Randomcrop_mask_label       = None

RandomFlip_flip_depth  = True
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

The configuration of 3D UNet is:

```bash
net_type = UNet3D
class_num     = 2
in_chns       = 1
feature_chns  = [32, 64, 128, 256]
dropout       = [0.0, 0.0, 0.5, 0.5]
up_mode       = 2
multiscale_pred = False
```

For training, we use a combinatin of DiceLoss and CrossEntropyLoss, and train the network by the `Adam` optimizer. The maximal iteration is 20k, and the training is early stopped if there is not performance improvement on the validation set for 5k iteratins. The learning rate scheduler is `ReduceLROnPlateau`. The corresponding configuration is:
```bash
gpus          = [0]
loss_type     = [DiceLoss, CrossEntropyLoss]
loss_weight   = [0.5, 0.5]

optimizer     = Adam
learning_rate = 1e-3
momentum      = 0.9
weight_decay  = 1e-5

lr_scheduler  = ReduceLROnPlateau
lr_gamma      = 0.5
ReduceLROnPlateau_patience = 3000
early_stop_patience = 5000

ckpt_dir    = model/unet3d_baseline

iter_max   = 20000
iter_valid = 100
iter_save  = 20000
```

During inference, we send the entire input volume to the network, and do not use postprocess. The configuration is:
```bash
# checkpoint mode can be [0-latest, 1-best, 2-specified]
ckpt_mode         = 1
output_dir        = result/unet3d_baseline
post_process      = None
sliding_window_enable = False
```

The following commands are used for training and inference with this method, respectively:

```bash
pymic_train config/unet3d_r10_baseline.cfg
pymic_test config/unet3d_r10_baseline.cfg
```

## 3. Data configuration for semi-supervised learning
For semi-supervised learning, we set the batch size as 4, where 2 are annotated images and the other 2 are unannotated images. 

```bash
tensor_type    = float
task_type      = seg
supervise_type = semi_sup

train_dir = ../../PyMIC_data/AtriaSeg/TrainingSet_crop/
train_csv = config/data/image_train_r10_lab.csv
train_csv_unlab = config/data/image_train_r10_unlab.csv
valid_csv = config/data/image_valid.csv
test_csv  = config/data/image_test.csv

train_batch_size = 2
train_batch_size_unlab = 2
```

### 3.1 Entropy Minimization
The configuration file for Entropy Minimization is `config/unet3d_r10_em.cfg`.  The data configuration has been described above, and the settings for data augmentation, network, optmizer, learning rate scheduler and inference are the same as those in the baseline method. Specific setting for Entropy Minimization is:

```bash
[semi_supervised_learning]
method_name    = EntropyMinimization
regularize_w   = 0.1
rampup_start   = 1000
rampup_end     = 15000
```

where the weight of the regularization loss is 0.1, and rampup is used to gradually increase it from 0 to 0.1.
The following commands are used for training and inference with this method, respectively:

```bash
pymic_train config/unet3d_r10_em.cfg
pymic_test config/unet3d_r10_em.cfg
```

### 3.2 UAMT
The configuration file for UAMT is `config/unet3d_r10_uamt.cfg`. The corresponding setting is:

```bash
[semi_supervised_learning]
method_name    = UAMT
regularize_w   = 0.1
ema_decay      = 0.99
rampup_start   = 1000
rampup_end     = 15000
```

The following commands are used for training and inference with this method, respectively:
```bash
pymic_train config/unet3d_r10_uamt.cfg
pymic_test config/unet3d_r10_uamt.cfg
```

### 3.3 UPRC
The configuration file for UPRC is `config/unet3d_r10_urpc.cfg`. This method requires deep supervision and pyramid prediction of a network. The network setting is:

```bash 
[network]
net_type = UNet3D
class_num     = 2
in_chns       = 1
feature_chns  = [32, 64, 128, 256]
dropout       = [0.0, 0.0, 0.5, 0.5]
up_mode       = 2
multiscale_pred  = True

[training]
deep_supervise = True
```

The setting for URPC training is:

```bash 
[semi_supervised_learning]
method_name    = URPC
regularize_w   = 0.1
rampup_start   = 1000
rampup_end     = 15000
```

The following commands are used for training and inference with this method, respectively:
```bash
pymic_train config/unet3d_r10_urpc.cfg
pymic_test config/unet3d_r10_urpc.cfg
```

### 3.4 CPS
The configuration file for CPS is `config/unet3d_r10_cps.cfg`, and the corresponding setting is:

```bash 
[semi_supervised_learning]
method_name    = CPS
regularize_w   = 0.1
rampup_start   = 1000
rampup_end     = 15000
```

The training and inference commands are:

```bash
pymic_train config/unet3d_r10_cps.cfg
pymic_test config/unet3d_r10_cps.cfg
```

## 4. Evaluation
Use `pymic_eval_seg -cfg config/evaluation.cfg` for quantitative evaluation of the segmentation results. You need to edit `config/evaluation.cfg` first, for example:

```bash
metric = [dice, assd]
label_list = [1]
organ_name = atrial
ground_truth_folder  = ../../PyMIC_data/AtriaSeg/TrainingSet_crop/
segmentation_folder  = result/unet3d_r10_baseline
evaluation_image_pair     = config/data/image_test_gt_seg.csv
```