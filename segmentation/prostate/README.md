# Prostate segmentation from 3D MRI
<img src="./picture/seg_example.png" width="796" height="682"/> 

In this example, we use 3D U-Net with deep supervision to segment the prostate from 3D MRI. 

## Data 
1. We use the [Promise12][promise12_link] dataset for this example. The preprocessed images are available in `PyMIC_data/Promise12`. We have resampled the original images into a uniform resolution and cropped them to a smaller size. The code for preprocessing is in  `preprocess.py`.
2. Run `python write_csv_files.py` to randomly split the dataset into our own training (35 images), validation (5 images) and testing (10 images) sets. The output csv files are saved in `config/data`.

[promise12_link]:https://promise12.grand-challenge.org/

## Training
1. Start to train by running:
 
```bash
pymic_run train config/unet3d.cfg
```

Note that we set `multiscale_pred = True`, `deep_supervise = True` and `loss_type = [DiceLoss, CrossEntropyLoss]` in the configure file. We also use Mixup for data
augmentation by setting `mixup_probability=0.5`.

2. During training or after training, run `tensorboard --logdir model/unet3d` and you will see a link in the output, such as `http://your-computer:6006`. Open the link in the browser and you can observe the average Dice score and loss during the training stage, such as shown in the following images, where blue and red curves are for training set and validation set respectively. 

![avg_dice](./picture/train_avg_dice.png)
![avg_loss](./picture/train_avg_loss.png)

## Testing and evaluation
1. Run the following command to obtain segmentation results of testing images. By default we set `ckpt_mode` to 1, which means using the best performing checkpoint based on the validation set.

```bash
pymic_run test config/unet3d.cfg
```

2. Run the following command to obtain quantitative evaluation results in terms of Dice. 

```bash
pymic_eval_seg config/evaluation.cfg
```

The obtained average Dice score by default setting should be close to 88.04%, and the Average Symmetric Surface Distance (ASSD) is 1.41 mm. You can try your efforts to improve the performance with different networks or training strategies by changing the configuration file `config/unet3d.cfg`.

