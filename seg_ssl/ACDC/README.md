# Semi-supervised demo using PyMIC

In this example, we show semi-supervised learning methods implemented in PyMIC.
Currently, the following semi-supervised methods are implemented:
|PyMIC Method|Reference|Remarks|
|---|---|---|
|SSLEntropyMinimization|[Grandvalet et al.][em_paper], 2005| Oringinally proposed for classification|
|SSLMeanTeacher| [Tarvainen et al.][mt_paper], 2017| Oringinally proposed for classification|


[em_paper]:https://papers.nips.cc/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf
[mt_paper]:https://arxiv.org/abs/1703.01780


## Data 
## Training
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
