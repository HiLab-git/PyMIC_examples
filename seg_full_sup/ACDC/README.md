# Heart Structure Segmentation using 2D Networks
<img src="./picture/seg_example.png" width="796" height="682"/> 

In this example, we show heart structure segmentaiton from the ACDC dataset using 2D networks. The following is a list of networks and their performance measured by Dice (%) by default setting. 

|Network  |Reference | RV| Myo | LV| Average|
|---|---|---|---|---|
|UNet2D | [Ronneberger et al., MICCAI 2015][unet_paper]| | | | | 
|UNet2D_scse |[Roy et al., TMI 2019][scse_paper]| | | | | 
|CANet| [Gu et al., TMI 2021][canet_paper]|  | | | | 
|COPLENet | [Wang et al., TMI 2020][coplenet]| | | | | 
|UNet++ | [Zhou et al., MICCAI Workshop 2018][unet]| | | | | 
|TransUNet | [Chen et al., Arxiv 2021][transunet]| | | | | 
|SwinUNet|  [Cao et al., ECCV Workshop 2022][swinunet]| | | | | 

[unet_paper]:https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28
[scse_paper]:https://ieeexplore.ieee.org/document/8447284
[canet_paper]:https://ieeexplore.ieee.org/abstract/document/9246575
[coplenet]:https://ieeexplore.ieee.org/document/9109297
[unet++]:https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1
[transunet]:https://arxiv.org/abs/2102.04306
[swinunet]:https://link.springer.com/chapter/10.1007/978-3-031-25066-8_9


## Data 
1. We use the [Promise12][promise12_link] dataset for this example. The preprocessed images are available in `PyMIC_data/Promise12`. We have resampled the original images into a uniform resolution and cropped them to a smaller size. The code for preprocessing is in  `preprocess.py`.
2. Run `python write_csv_files.py` to randomly split the dataset into our own training (35 images), validation (5 images) and testing (10 images) sets. The output csv files are saved in `config/data`.

[promise12_link]:https://promise12.grand-challenge.org/

## Training
1. Start to train by running:
 
```bash
pymic_train config/unet3d.cfg
```

Note that we set `multiscale_pred = True`, `deep_supervise = True` and `loss_type = [DiceLoss, CrossEntropyLoss]` in the configure file. We also use Mixup for data
augmentation by setting `mixup_probability=0.5`.

2. During training or after training, run `tensorboard --logdir model/unet3d` and you will see a link in the output, such as `http://your-computer:6006`. Open the link in the browser and you can observe the average Dice score and loss during the training stage, such as shown in the following images, where blue and red curves are for training set and validation set respectively. 

![avg_dice](./picture/train_avg_dice.png)
![avg_loss](./picture/train_avg_loss.png)

## Testing and evaluation
1. Run the following command to obtain segmentation results of testing images. By default we set `ckpt_mode` to 1, which means using the best performing checkpoint based on the validation set.

```bash
pymic_test config/unet3d.cfg
```

2. Run the following command to obtain quantitative evaluation results in terms of Dice. 

```bash
pymic_eval_seg -cfg config/evaluation.cfg
```

The obtained average Dice score by default setting should be close to 88.04%, and the Average Symmetric Surface Distance (ASSD) is 1.41 mm. You can try your efforts to improve the performance with different networks or training strategies by changing the configuration file `config/unet3d.cfg`.

