# Weakly supervised segmentation demo using PyMIC

In this example, we show scribble-supervised learning methods implemented in PyMIC.
Currently, the following methods are available in PyMIC:
|PyMIC Method|Reference|Remarks|
|---|---|---|
|WSLEntropyMinimization|[Grandvalet et al.][em_paper], NeurIPS 2005| Entropy minimization for regularization|
|WSLTotalVariation| [Luo et al.][tv_paper], arXiv 2022| Tobal variation for regularization|
|WSLMumfordShah| [Kim et al.][mumford_paper], TIP 2020| Mumford-Shah loss for regularization|
|WSLGatedCRF| [Lbukhov et al.][gcrf_paper], arXiv 2019| Gated CRF for regularization|
|WSLUSTM| [Liu et al.][ustm_paper], PR 2022| Adapt USTM with transform-consistency|
|WSLDMPLS| [Luo et al.][dmpls_paper], MICCAI 2022| Dynamically mixed pseudo label supervision|

[em_paper]:https://papers.nips.cc/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf
[tv_paper]:https://arxiv.org/abs/2111.02403
[mumford_paper]:https://doi.org/10.1109/TIP.2019.2941265  
[gcrf_paper]:http://arxiv.org/abs/1906.04651
[ustm_paper]:https://doi.org/10.1016/j.patcog.2021.108341 
[dmpls_paper]:https://arxiv.org/abs/2203.02106 


## Data 
The [ACDC][ACDC_link] (Automatic Cardiac Diagnosis Challenge) dataset is used in this demo. It contains 200 short-axis cardiac cine MR images of 100 patients, and the classes for segmentation are: Right Ventricle (RV), Myocardiym (Myo) and Left Ventricle (LV). [Valvano et al.][scribble_link] provided scribble annotations of this dataset. The images and scribble annotations are available in `PyMIC_data/ACDC/preprocess`, where we have normalized the intensity to [0, 1]. The images are split at patient level into 70%, 10% and 20% for training, validation  and testing, respectively (see `config/data` for details).

[ACDC_link]:https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
[scribble_link]:https://gvalvano.github.io/wss-multiscale-adversarial-attention-gates/data

## Training
In this demo, we experiment with five methods: EM, TV, GatedCRF, USTM and DMPLS, and they are compared with the baseline of learning from annotated pixels with partial CE loss. All these methods use UNet2D as the backbone network.

### Baseline Method
The dataset setting is similar to that in the `seg_ssl/ACDC` demo. Here we use a slightly different setting of data transform:

```bash
tensor_type    = float
task_type      = seg
supervise_type = fully_sup
root_dir  = ../../PyMIC_data/ACDC/preprocess
train_csv = config/data/image_train.csv
valid_csv = config/data/image_valid.csv
test_csv  = config/data/image_test.csv
train_batch_size = 4

# data transforms
train_transform = [Pad, RandomCrop, RandomFlip, NormalizeWithMeanStd, PartialLabelToProbability]
valid_transform = [NormalizeWithMeanStd, Pad, LabelToProbability]
test_transform  = [NormalizeWithMeanStd, Pad]

Pad_output_size = [4, 224, 224]
Pad_ceil_mode   = False

RandomCrop_output_size = [3, 224, 224]
RandomCrop_foreground_focus = False
RandomCrop_foreground_ratio = None
Randomcrop_mask_label       = None

RandomFlip_flip_depth  = False
RandomFlip_flip_height = True
RandomFlip_flip_width  = True

NormalizeWithMeanStd_channels = [0]
```

Please note that we use a `PartialLabelToProbability` class to convert the partial labels into a one-hot segmentation map and a mask for annotated pixels. The mask is used as a pixel weighting map in `CrossEntropyLoss`, so that parial CE loss is calculated as a weighted CE loss, i.e., the weight for unannotated pixels is 0.


The configuration of 2D UNet is:

```bash
net_type = UNet2D
class_num     = 4
in_chns       = 1
feature_chns  = [16, 32, 64, 128, 256]
dropout       = [0.0, 0.0, 0.0, 0.5, 0.5]
bilinear      = True
multiscale_pred = False
```

For training, we use the CrossEntropyLoss with pixel weighting (i.e., partial CE loss), and train the network by the  `Adam` optimizer. The maximal iteration is 20k, and the training is early stopped if there is not performance improvement on the validation set for 8k iteratins. The learning rate scheduler is `ReduceLROnPlateau`. The corresponding configuration is:

```bash
gpus       = [0]
loss_type     = CrossEntropyLoss

# for optimizers
optimizer     = Adam
learning_rate = 1e-3
momentum      = 0.9
weight_decay  = 1e-5

# for lr schedular 
lr_scheduler  = ReduceLROnPlateau
lr_gamma      = 0.5
ReduceLROnPlateau_patience = 2000
early_stop_patience = 8000
ckpt_save_dir    = model/unet2d_baseline

# start iter
iter_start = 0
iter_max   = 20000
iter_valid = 100
iter_save  = [2000, 20000]
```

During inference, we use a sliding window of 3x224x224, and post process the results by `KeepLargestComponent`. The configuration is:
```bash
# checkpoint mode can be [0-latest, 1-best, 2-specified]
ckpt_mode         = 1
output_dir        = result/unet2d_baseline
post_process      = KeepLargestComponent

sliding_window_enable = True
sliding_window_size   = [3, 224, 224]
sliding_window_stride = [3, 224, 224]
```

The following commands are used for training and inference with this method, respectively:

```bash
pymic_train config/unet2d_baseline.cfg
pymic_test config/unet2d_baseline.cfg
```

### Data configuration for other weakly supervised learning
For other weakly supervised learning methods, please set `supervise_type = weak_sup` in the configuration.

```bash
tensor_type    = float
task_type      = seg
supervise_type = weak_sup

root_dir  = ../../PyMIC_data/ACDC/preprocess
train_csv = config/data/image_train.csv
valid_csv = config/data/image_valid.csv
test_csv  = config/data/image_test.csv
...
```

### Entropy Minimization
The configuration file for Entropy Minimization is `config/unet2d_em.cfg`.  The data configuration has been described above, and the settings for data augmentation, network, optmizer, learning rate scheduler and inference are the same as those in the baseline method. Specific setting for Entropy Minimization is:

```bash
[weakly_supervised_learning]
method_name    = EntropyMinimization
regularize_w   = 0.1
rampup_start   = 2000
rampup_end     = 15000
```

where wet the weight of the regularization loss as 0.1, rampup is used to gradually increase it from 0 t 0.1.

The following commands are used for training and inference with this method, respectively:

```bash
pymic_train config/unet2d_em.cfg
pymic_test config/unet2d_em.cfg
```

### TV
The configuration file for TV is `config/unet2d_tv.cfg`. The corresponding setting is:

```bash
[weakly_supervised_learning]
method_name    = TotalVariation
regularize_w   = 0.1
rampup_start   = 2000
rampup_end     = 15000
```

The following commands are used for training and inference with this method, respectively:
```bash
pymic_train config/unet2d_tv.cfg
pymic_test config/unet2d_tv.cfg
```

### Gated CRF
The configuration file for Gated CRF is `config/unet2d_gcrf.cfg`. The corresponding setting is:

```bash 
[weakly_supervised_learning]
method_name    = GatedCRF
regularize_w   = 0.1
rampup_start   = 2000
rampup_end     = 15000
GatedCRFLoss_W0     = 1.0
GatedCRFLoss_XY0    = 5
GatedCRFLoss_rgb    = 0.1
GatedCRFLoss_W1     = 1.0
GatedCRFLoss_XY1    = 3
GatedCRFLoss_Radius = 5
```

The following commands are used for training and inference with this method, respectively:

```bash
pymic_train config/unet2d_gcrf.cfg
pymic_test config/unet2d_gcrf.cfg
```

### USTM
The configuration file for USTM is `config/unet2d_ustm.cfg`. The corresponding setting is:

```bash
[weakly_supervised_learning]
method_name    = USTM
regularize_w   = 0.1
rampup_start   = 2000
rampup_end     = 15000
```

The commands for training and inference are:

```bash
pymic_train config/unet2d_ustm.cfg
pymic_test config/unet2d_ustm.cfg
```

### DMPLS
The configuration file for DMPLS is `config/unet2d_dmpls.cfg`, and the corresponding setting is:

```bash 
[weakly_supervised_learning]
method_name    = DMPLS
regularize_w   = 0.1
rampup_start   = 2000
rampup_end     = 15000
```

The training and inference commands are:

```bash
pymic_train config/unet2d_dmpls.cfg
pymic_test config/unet2d_dmpls.cfg
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

