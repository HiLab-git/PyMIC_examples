# PyMIC_examples
[PyMIC][PyMIC_link] is an Pytorch-based medical image computing toolkit with deep learning. Here we provide a set of examples to show how it can be used for image classification and segmentation tasks. For beginners, you can follow the examples by just editting the configure files for model training, testing and evaluation. For advanced users, you can develop your own modules, such as customized networks and loss functions.  

## Install PyMIC
The latest released version of PyMIC can be installed by:

```bash
pip install PYMIC==0.2.3
```

To use the latest development version, you can download the source code [here][PyMIC_link], and install it by:

```bash
python setup.py install
``` 

## List of Examples
Currently we provide two examples for image classification, and four examples for 2D/3D image segmentation. These examples include:

1, [classification/AntBee][AntBee_link]: finetuning a resnet18 for Ant and Bee classification.

2, [classification/CHNCXR][CHNCXR_link]: finetuning restnet18 and vgg16 for normal/tuberculosis X-ray image classification.

3, [segmentation/JSRT][JSRT_link]: using a 2D UNet for heart segmentation from chest X-ray images.

4, [segmentation/JSRT2][JSRT2_link]: defining a customized network for heart segmentation from chest X-ray images.

5, [segmentation/fetal_hc][fetal_hc_link]: using a 2D UNet for fetal head segmentation from 2D ultrasound images.

6, [segmentation/prostate][prostate_link]: using a 3D UNet for prostate segmentation from 3D MRI.

[PyMIC_link]: https://github.com/HiLab-git/PyMIC
[AntBee_link]:classification/AntBee
[CHNCXR_link]:classification/CHNCXR
[JSRT_link]:segmentation/JSRT
[JSRT2_link]:segmentation/JSRT2
[fetal_hc_link]:segmentation/fetal_hc
[prostate_link]:segmentation/prostate

