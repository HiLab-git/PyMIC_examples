# PyMIC with Customized Network and Loss

In this example, we show how to use a customized network and loss function to segment the lung from X-Ray images. The configurations are the same as those in the `2d_JSRT` example except for the network structure and loss function. 

## 1. Define your own network and loss
The customized network is detailed in `my_net2d.py`, which is a modification of the 2D UNet, and names as `MyUNet2D`. In this new network, we use a residual connection in each block. The customized loss is detailed in `my_loss.py`, where we define a focal dice loss named as `MyFocalDiceLoss`. We use `MyFocalDiceLoss + CrossEntropyLoss` to train the customized network.

We also write a customized main function in `custom_run.py` so that we can combine `SegmentationAgent` from PyMIC with our customized network and loss function. The corresponding code is like:

```python
from pymic.net_run.agent_seg import  SegmentationAgent
from pymic.loss.loss_dict_seg import SegLossDict
from my_net2d import MyUNet2D 
from my_loss  import MyFocalDiceLoss

loss_dict = {'MyFocalDiceLoss':MyFocalDiceLoss}
loss_dict.update(SegLossDict)

def main():
    ...
    agent  = SegmentationAgent(config, args.stage)
    # use custormized network and loss function
    mynet  = MyUNet2D(config['network'])
    agent.set_network(mynet)
    agent.set_loss_dict(loss_dict)
    agent.run()
```

## 2. Training
We use the same dataset as in the the `2d_JSRT` example, and specify data, network and hyper-parameter configurations in `config/mynet.cfg`:

```bash
[dataset]
...
train_dir = ../../PyMIC_data/JSRT
train_csv = ../2d_JSRT/config/jsrt_train.csv
valid_csv = ../2d_JSRT/config/jsrt_valid.csv
test_csv  = ../2d_JSRT/config/jsrt_test.csv

train_batch_size = 4

# data transforms
train_transform = [NormalizeWithMeanStd, RandomCrop, LabelConvert, LabelToProbability]
valid_transform = [NormalizeWithMeanStd, LabelConvert, LabelToProbability]
test_transform  = [NormalizeWithMeanStd]

NormalizeWithMeanStd_channels = [0]
RandomCrop_output_size = [240, 240]
LabelConvert_source_list = [0, 255]
LabelConvert_target_list = [0, 1]

[network]
net_type      = MyUNet2D
class_num     = 2
in_chns       = 1
feature_chns  = [16, 32, 64, 128, 256]
dropout       = [0.0, 0.0, 0.3, 0.4, 0.5]

[training]
...
loss_type     = [MyFocalDiceLoss, CrossEntropyLoss]
loss_weight   = [1.0, 1.0]
MyFocalDiceLoss_beta = 1.5

optimizer     = Adam
learning_rate = 1e-3
momentum      = 0.9
weight_decay  = 1e-5

lr_scheduler  = MultiStepLR
lr_gamma      = 0.5
lr_milestones = [1500, 3000, 4500]
ckpt_dir    = model/mynet

# start iter
iter_max   = 6000
iter_valid = 250
iter_save  = 6000
...
```

where we use our customized networks and loss by setting `net_type = MyUNet2D` and `loss_type = [MyFocalDiceLoss, CrossEntropyLoss]`. Start to train by running:
 
```bash
python custom_run.py train config/mynet.cfg
```

During training or after training, run `tensorboard --logdir model/mynet` and you will see a link in the output, such as `http://your-computer:6006`. Open the link in the browser and you can observe the average Dice score and loss during the training stage.

### 3. Testing and evaluation
The configuration for testing is similar to that in the `2d_JSRT` example:

```bash
[testing]
gpus       = [0]
ckpt_mode  = 1
output_dir = result/mynet

label_source = [0, 1]
label_target = [0, 255]
```

Run the following command for testing:
 
```bash
python custom_run.py test config/mynet.cfg
```

Finally, use the following command to obtain quantitative evaluation results in terms of Dice and ASSD.

```bash
pymic_eval_seg -cfg config/evaluation.cfg
```

The obtained Dice score by default setting should be close to 97.95%, with an ASSD value of 1.04 pixel.
