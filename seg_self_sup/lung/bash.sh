
### 1, Preprocess luna dataset for pretraining, and split the data into training and validation
# python luna_preprocess.py
# python get_luna_csv.py

### 2, Split the LCTSC2017 dataset into training, validaiton and testing
# python write_csv.py

### 3, For volume fusion, the following three commens are used to create the validation images, 
# do pretraining, and applying the pretrained model to the downstream segmentation task, respectively. 
# pymic_preprocess  config/luna_data/preprocess_volumefusion.cfg
# pymic_train config/luna_pretrain/unet3d_volumefusion.cfg
# pymic_train config/lctsc_train/unet3d_volumefusion.cfg

### 4, Train from scratch
# pymic_train config/lctsc_train/unet3d_scratch.cfg

### 5, Inference with testing images and do the evaluation. 
# pymic_test config/lctsc_train/unet3d_volumefusion.cfg
# pymic_test config/lctsc_train/unet3d_scratch.cfg
# pymic_eval_seg -cfg config/evaluation.cfg

### 6, Pretraining with Model Genesis. The following three commens are used to create the validation images, 
# do pretraining, and applying the pretrained model to the downstream segmentation task, respectively. 
# pymic_preprocess config/luna_data/preprocess_genesis.cfg
# pymic_train config/luna_pretrain/unet3d_genesis.cfg
# pymic_train config/lctsc_train/unet3d_genesis.cfg

### 7, Pretraining with Patch Swap. The following three commens are used to create the validation images, 
# do pretraining, and applying the pretrained model to the downstream segmentation task, respectively. 
# pymic_preprocess config/luna_data/preprocess_patchswap.cfg
# pymic_train config/luna_pretrain/unet3d_patchswap.cfg
# pymic_train config/lctsc_train/unet3d_patchswap.cfg