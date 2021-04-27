# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import csv 
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt 


def show_roc(gt_csv, prob_csv):
    gt_items  = pd.read_csv(gt_csv)
    prob_items = pd.read_csv(prob_csv)
    assert(len(gt_items) == len(prob_items))
    for i in range(len(gt_items)):
        assert(gt_items.iloc[i, 0] == prob_items.iloc[i, 0])
    
    gt_data  = np.asarray(gt_items.iloc[:, 1])
    prob_data = np.asarray(prob_items.iloc[:, -1])
    fpr, tpr, thre = metrics.roc_curve(gt_data, prob_data)
    auc = metrics.auc(fpr, tpr)
    plt.title("ROC")
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.show() 

gt_csv = "config/valid_data.csv"
prob_csv   = "result/resnet18_ce1_prob.csv"
show_roc(gt_csv, prob_csv)
