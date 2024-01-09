# PyMIC_examples
[PyMIC][PyMIC_link] is a PyTorch-based toolkit for medical image computing with annotation-efficient deep learning. Here we provide a set of examples to show how it can be used for image classification and segmentation tasks. For annotation efficient learning, we show examples of Semi-Supervised Learning (SSL), Self-Supervised Learning (self-SL), Weakly Supervised Learning (WSL) and Noisy Label Learning (NLL), respectively.  For beginners, you can follow the examples by just editting the configuration files for model training, testing and evaluation. For advanced users, you can easily develop your own modules, such as customized networks and loss functions.  

## Install PyMIC
The released version of PyMIC (v0.4.0) is required for these examples, and it can be installed by:

```bash
pip install PYMIC==0.4.0
```

To use the latest development version, you can download the source code [here][PyMIC_link], and install it by:

```bash
python setup.py install
``` 

## Data
The datasets for the examples can be downloaded from [Google Drive][google_link] or [Baidu Disk][baidu_link] (extraction code: xlwg). Extract the files to `PyMIC_data` after downloading. 


## List of Examples

Currently we provide the following examples in this repository:
|Catetory|Example|Remarks|
|---|---|---|
|Classification|[AntBee][AntBee_link]|Finetuning a resnet18 for Ant and Bee classification|
|Classification|[CHNCXR][CHNCXR_link]|Finetuning restnet18 and vgg16 for normal/tuberculosis X-ray image classification|
|Fully supervised segmentation|[JSRT][JSRT_link]|Using five 2D Networks for lung segmentation from chest X-ray images|
|Fully supervised segmentation|[JSRT2][JSRT2_link]|Using a customized network and loss function for the JSRT dataset|
|Fully supervised segmentation|[Fetal_HC][fetal_hc_link]|Using a 2D UNet for fetal head segmentation from 2D ultrasound images|
|Fully supervised segmentation|[Prostate][prostate_link]|Using a 3D UNet for prostate segmentation from 3D MRI|
|Semi-supervised segmentation|[seg_semi_sup/ACDC][ssl_acdc_link]|Semi-supervised methods for heart structure segmentation using 2D CNNs|
|Semi-supervised segmentation|[seg_semi_sup/AtriaSeg][ssl_atrial_link]|Semi-supervised methods for left atrial segmentation using 3D CNNs|
|Weakly-supervised segmentation|[seg_weak_sup/ACDC][wsl_acdc_link]|Segmentation of heart structure with scrible annotations|
|Noisy label learning|[seg_noisy_label/JSRT][nll_jsrt_link]|Comparing different NLL methods for learning from noisy labels|
|Self-Supervised learning|[seg_self_sup/lung][self_lung_link]|Self-Supervised learning methods for pretraining a segmentation model|

[PyMIC_link]: https://github.com/HiLab-git/PyMIC
[google_link]:https://drive.google.com/file/d/1eZakSEBr_zfIHFTAc96OFJix8cUBf-KR/view?usp=sharing
[baidu_link]:https://pan.baidu.com/s/1tN0inIrVYtSxTVRfErD9Bw 
[AntBee_link]:classification/AntBee
[CHNCXR_link]:classification/CHNCXR
[JSRT_link]:seg_full_sup/JSRT
[JSRT2_link]:seg_full_sup/JSRT2
[fetal_hc_link]:seg_full_sup/fetal_hc
[prostate_link]:seg_full_sup/prostate
[ssl_acdc_link]:seg_semi_sup/ACDC
[ssl_atrial_link]:seg_semi_sup/AtriaSeg/
[wsl_acdc_link]:seg_weak_sup/ACDC 
[nll_jsrt_link]:seg_noisy_label/JSRT
[self_lung_link]:seg_self_sup/lung

## Useful links
* PyMIC on Github: https://github.com/HiLab-git/PyMIC
* Usage of PyMIC: https://pymic.readthedocs.io/en/latest/usage.html 