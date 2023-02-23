# Semi-supervised segmentation using PyMIC

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
The [ACDC][ACDC_link] (Automatic Cardiac Diagnosis Challenge) dataset is used in this demo. It contains 200 short-axis cardiac cine MR images of 100 patients, and the classes for segmentation are: Right Ventricle (RV), Myocardiym (Myo) and Left Ventricle (LV). The images are available in `PyMIC_data/ACDC/preprocess`, where we have normalized the intensity to [0, 1]. The images are split at patient level into 70%, 10% and 20% for training, validation  and testing, respectively (see `config/data` for details).

In the training set, we have randomly selected 14 images of 7 patients as annotated images and the other 126 images as unannotated images. See `random_split_train.py`. 

[ACDC_link]:https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

## Training
In this demo, we experiment with five methods: EM, UAMT, UPRC, CCT and CPS, and they are compared with the baseline of learning from annotated images. All these methods use UNet2D as the backbone network.

### Baseline Method
The baseline method uses the 14 annotated cases for training. The batch size is 4, and the patch size is 6x192x192. Therefore, indeed there are 16 2D slices in each batch. See `config/unet2d_baseline.cfg` for details about the configuration. The dataset configuration is:

```bash
tensor_type    = float
task_type      = seg
supervise_type = fully_sup
root_dir  = ../../PyMIC_data/ACDC/preprocess/
train_csv = config/data/image_train_r10_lab.csv
valid_csv = config/data/image_valid.csv
test_csv  = config/data/image_test.csv
train_batch_size = 4
```

For data augmentation, we use random rotate, random crop, random flip, gamma correction and gaussian noise. The cropped images are also normaized with mean and std. The details for data transforms are:

```bash
train_transform = [Pad, RandomRotate, RandomCrop, RandomFlip, NormalizeWithMeanStd, GammaCorrection, GaussianNoise, LabelToProbability]
valid_transform = [NormalizeWithMeanStd, Pad, LabelToProbability]
test_transform  = [NormalizeWithMeanStd, Pad]

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
net_type      = UNet2D
class_num     = 4
in_chns       = 1
feature_chns  = [16, 32, 64, 128, 256]
dropout       = [0.0, 0.0, 0.0, 0.5, 0.5]
bilinear      = True
multiscale_pred = False
```

For training, we use a combinatin of DiceLoss and CrossEntropyLoss, and train the network by the   `Adam` optimizer. The maximal iteration is 30k, and the training is early stopped if there is not performance improvement on the validation set for 10k iteratins. The learning rate scheduler is `ReduceLROnPlateau`. The corresponding configuration is:
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
ReduceLROnPlateau_patience = 4000
early_stop_patience = 10000

ckpt_save_dir    = model/unet2d_baseline

iter_start = 0
iter_max   = 30000
iter_valid = 100
iter_save  = [30000]
```

During inference, we use a sliding window of 6x192x192, and postprocess the results by `KeepLargestComponent`. The configuration is:
```bash
# checkpoint mode can be [0-latest, 1-best, 2-specified]
ckpt_mode         = 1
output_dir        = result/unet2d_baseline
post_process      = KeepLargestComponent

sliding_window_enable = True
sliding_window_size   = [6, 192, 192]
sliding_window_stride = [6, 192, 192]
```

The following commands are used for training and inference with this method, respectively:

```bash
pymic_train config/unet2d_baseline.cfg
pymic_test  config/unet2d_baseline.cfg
```

### Data configuration for semi-supervised learning
For semi-supervised learning, we set the batch size as 8, where 4 are annotated images and the other 4 are unannotated images. 

```bash
tensor_type    = float
task_type      = seg
supervise_type = semi_sup

root_dir  = ../../PyMIC_data/ACDC/preprocess/
train_csv = config/data/image_train_r10_lab.csv
train_csv_unlab = config/data/image_train_r10_unlab.csv
valid_csv = config/data/image_valid.csv
test_csv  = config/data/image_test.csv

train_batch_size = 4
train_batch_size_unlab = 4
```

### Entropy Minimization
The configuration file for Entropy Minimization is `config/unet2d_em.cfg`.  The data configuration has been described above, and the settings for data augmentation, network, optmizer, learning rate scheduler and inference are the same as those in the baseline method. Specific setting for Entropy Minimization is:

```bash
[semi_supervised_learning]
method_name    = EntropyMinimization
regularize_w   = 0.1
rampup_start   = 1000
rampup_end     = 20000
```

where the weight of the regularization loss is 0.1, and rampup is used to gradually increase it from 0 to 0.1.
The following commands are used for training and inference with this method, respectively:

```bash
pymic_train config/unet2d_em.cfg
pymic_test config/unet2d_em.cfg
```

### UAMT
The configuration file for UAMT is `config/unet2d_uamt.cfg`. The corresponding setting is:

```bash
[semi_supervised_learning]
method_name    = UAMT
regularize_w   = 0.1
ema_decay      = 0.99
rampup_start   = 1000
rampup_end     = 20000
```

The following commands are used for training and inference with this method, respectively:
```bash
pymic_train config/unet2d_uamt.cfg
pymic_test config/unet2d_uamt.cfg
```

### UPRC
The configuration file for UPRC is `config/unet2d_urpc.cfg`. This method requires deep supervision and pyramid prediction of a network. The network setting is:

```bash 
[network]
net_type      = UNet2D
class_num     = 4
in_chns       = 1
feature_chns  = [16, 32, 64, 128, 256]
dropout       = [0.0, 0.0, 0.0, 0.5, 0.5]
bilinear      = True
multiscale_pred = True
[training]
deep_supervise = True
```

The setting for URPC training is:

```bash 
[semi_supervised_learning]
method_name    = URPC
regularize_w   = 0.1
rampup_start   = 1000
rampup_end     = 20000
```

The following commands are used for training and inference with this method, respectively:
```bash
pymic_train config/unet2d_urpc.cfg
pymic_test config/unet2d_urpc.cfg
```

### CCT
The orginal [CCT][cct_paper] uses multiple auxiliary deocders in the network. Due to the memory constraint and efficiency consideration, we only use 4 auxiliary decoders based on DropOut, FeatureDrop, FeatureNoise and VAT, respectively. The configuration file of CCT is `config/unet2d_cct.cfg`, and the network setting is:

```bash 
net_type      = UNet2D_CCT
class_num     = 4
in_chns       = 1
feature_chns  = [16, 32, 64, 128, 256]
dropout       = [0.0, 0.0, 0.0, 0.5, 0.5]
bilinear      = True

# parameters specific to CCT
VAT_it = 2
VAT_xi = 1e-6
VAT_eps= 2
Uniform_range = 0.3
```

The setting for CCT training is:

```bash 
[semi_supervised_learning]
method_name    = CCT
regularize_w   = 0.1
rampup_start   = 1000
rampup_end     = 20000
unsupervised_loss = MSE
```

The following commands are used for training and inference with this method, respectively:

```bash
pymic_train config/unet2d_cct.cfg
pymic_test config/unet2d_cct.cfg
```

### CPS
The configuration file for CPS is `config/unet2d_cps.cfg`, and the corresponding setting is:

```bash 
[semi_supervised_learning]
method_name    = CPS
regularize_w   = 0.1
rampup_start   = 1000
rampup_end     = 20000
```

The training and inference commands are:

```bash
pymic_train config/unet2d_cps.cfg
pymic_test config/unet2d_cps.cfg
```

## Evaluation
Use `pymic_eval_seg config/evaluation.cfg` for quantitative evaluation of the segmentation results. You need to edit `config/evaluation.cfg` first, for example:

```bash
metric_list = [dice, hd95]
label_list = [1,2,3]
organ_name = heart
ground_truth_folder_root  = ../../PyMIC_data/ACDC/preprocess
segmentation_folder_root  = ./result/unet2d_baseline
evaluation_image_pair     = ./config/data/image_test_gt_seg.csv
```