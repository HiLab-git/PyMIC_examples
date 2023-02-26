# Fetal brain segmentation from ultrasound images

<img src="./picture/001_HC.png" width="256" height="256"/> <img src="./picture/001_HC_seg.png" width="256" height="256"/>

In this example, we use 2D U-Net to segment the fetal brain from ultrasound images. During training, we use tensorboard to observe the performance of the network at different iterations. We then apply the trained model to testing images and obtain quantitative evaluation results.


## Data and preprocessing
1. We use the [HC18][hc18_link] dataset for this example. The images are ready to use in `PyMIC_data/Fetal_HC`. There are 999 2D ultrasound images and their annotations in the training set.  The original annotations are contours, and we have converted them into binary masks for segmentation. (The conversion code is in `get_ground_truth.py`)
2. Run `python write_csv_files.py` to randomly split the official HC18 training set into our own training (780 images), validation (70 images) and testing (149 images) sets. The output csv files are saved in `config`.

[hc18_link]:https://hc18.grand-challenge.org/

## Training
1. Start to train by running:
 
```bash
pymic_train config/unet.cfg
```

2. During training or after training, run `tensorboard --logdir model/unet` and you will see a link in the output, such as `http://your-computer:6006`. Open the link in the browser and you can observe the average Dice score and loss during the training stage, such as shown in the following images, where red and blue curves are for training set and validation set respectively. 

![avg_dice](./picture/train_avg_dice.png)
![avg_loss](./picture/train_avg_loss.png)

## Testing and evaluation
1. Run the following command to obtain segmentation results of testing images based on the best-performing checkpoint on the validation set. By default we use sliding window inference to get better results. You can also edit the `testing` section of `config/unet.cfg` to use other inference strategies.

```bash
pymic_test config/unet.cfg
```

2. Use the following command to obtain quantitative evaluation results in terms of Dice. 

```bash
pymic_eval_seg config/evaluation.cfg
```

The obtained average Dice score by default setting should be close to 97.07%. The Average Symmetric Surface Distance (ASSD) is also calculated. 

3. Set `tta_mode = 1` in `config/unet.cfg` to enable test time augmentation, and run the testing and evaluation code again, we find that the average Dice will be increased to around 97.22%.
