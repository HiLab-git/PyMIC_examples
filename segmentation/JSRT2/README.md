# Lung segmentation from 2D X-ray images using customized CNN

![image_example](../JSRT/picture/JPCLN001.png)
![label_example](../JSRT/picture/JPCLN001_seg.png)

In this example, we show how to use a customized CNN and a customized loss function to segment the lung from X-Ray images. The configurations are the same as those in the `JSRT` example except the network structure and loss function. 

The customized CNN is detailed in `my_net2d.py`, which is a modification of the 2D UNet. In this new network, we use a residual connection in each block. The customized loss is detailed in `my_loss.py`, where we define a focal dice loss named as MyFocalDiceLoss. We use `MyFocalDiceLoss + CrossEntropyLoss` to train the customized network.

We also write a customized main function in `jsrt_net_run.py` so that we can combine SegmentationAgent from PyMIC with our customized CNN and loss function.

## Data 
1. We use the same dataset as in the the `JSRT` example. 

## Training
1. Edit `config/mynet.cfg` by setting the value of `root_dir` as your `JSRT_root`, and start to train by running:
 
```bash
python net_run_jsrt.py train config/mynet.cfg
```

2. During training or after training, run `tensorboard --logdir model/mynet` and you will see a link in the output, such as `http://your-computer:6006`. Open the link in the browser and you can observe the average Dice score and loss during the training stage, such as shown in the following images, where red and blue curves are for training set and validation set respectively. 

![avg_dice](./picture/jsrt2_avg_dice.png)
![avg_loss](./picture/jsrt2_avg_loss.png)

## Testing and evaluation
1. Edit the `testing` section in `config/mynet.cfg`, and run the following command for testing:
 
```bash
python net_run_jsrt.py test config/mynet.cfg
```

2. Edit `config/evaluation.cfg` by setting `ground_truth_folder_root` as your `JSRT_root`, and run the following command to obtain quantitative evaluation results in terms of dice.

```
pymic_eval_seg config/evaluation.cfg
```

The obtained dice score by default setting should be close to 97.999%. 
