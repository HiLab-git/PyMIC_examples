# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from pymic.loss.seg.util import reshape_tensor_to_2D, get_classwise_dice

class MyFocalDiceLoss(nn.Module):
    """
    Focal Dice loss proposed in the following paper:
       P. Wang et al. Focal dice loss and image dilatin for brain tumor segmentation.
       in Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical
       Decision Support, 2018.
    """
    def __init__(self, params):
        super(MyFocalDiceLoss, self).__init__()
        self.beta = params['MyFocalDiceLoss_beta'.lower()]
        self.softmax = params.get('loss_softmax', True)
        assert(self.beta >= 1.0)

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']

        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 

        dice_score = get_classwise_dice(predict, soft_y, None)
        dice_score = 0.005 + dice_score * 0.99
        dice_loss  = 1.0 - torch.pow(dice_score, 1.0 / self.beta)

        avg_loss = torch.mean(dice_loss)   
        return avg_loss
