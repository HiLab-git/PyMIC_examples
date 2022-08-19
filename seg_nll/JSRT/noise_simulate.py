# -*- coding: utf-8 -*-
"""
Generate noisy 2D segmentation labels by random dilation, erosion
or edge distortion. 
"""
import os
import sys
from PIL import Image
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import copy
import numpy as np
import pandas as pd
import random

def create_circle_mask_on_edge(label, r_range, sample_ratio):
    H, W = label.shape
    edge = ndimage.binary_dilation(label) - label
    y, x = np.where(edge > 0)
    edge_length = len(y)
    idx  = random.sample(range(edge_length), int(edge_length * sample_ratio))
    ys, xs = y[idx], x[idx]
    
    # create mask with circle
    mask = np.zeros_like(label)
    num  = len(xs)
    for i in range(num):
        yi, xi = ys[i], xs[i]
        r = random.randint(r_range[0], r_range[1])
        for h in range(-r, r):
            for w in range(-r, r):
                yt, xt = yi + h, xi + w 
                if(yt < 0 or yt >= H or xt <  0 or xt >= W):
                    continue
                if((xt - xi)* (xt - xi) + (yt - yi)* (yt - yi) < r*r):
                    mask[yt, xt] = 1
    return mask

def random_edge_distort(label, r_range, sample_ratio):
    mask1 = create_circle_mask_on_edge(label, r_range, sample_ratio)
    out   = np.maximum(mask1, label)
    mask2 = create_circle_mask_on_edge(out, r_range, sample_ratio)
    out = (1 - mask2) * out
    return out

def add_random_nosie_to_label(label, r_range):
    p = random.random()
    if(p < 0.35):
        r = random.randint(r_range[0], r_range[1])
        label = ndimage.binary_dilation(label, iterations = r)
    elif(p < 0.7):
        r = random.randint(r_range[0], r_range[1])
        label = ndimage.binary_erosion(label, iterations = r)
    else:
        label = random_edge_distort(label, r_range, sample_ratio = 0.1)
    return label 

def debug():
    # for debug
    image_name = "/home/disk2t/projects/PyMIC_project/PyMIC_data/JSRT/label/JPCLN003.png"
    img = Image.open(image_name)
    lab = np.asarray(img)
    lab_max = lab.max()
    lab  = np.asarray(lab > 0, np.uint8)
    r_range = (6, 9)
    lab_noise = add_random_nosie_to_label(lab, r_range) * lab_max

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(lab_noise)
    plt.show()

def select_clean_samples(clean_ratio = 0.2):
    train_csv_name = "config/data/jsrt_train.csv"
    dframe = pd.read_csv(train_csv_name)
    dframe = dframe.sample(frac=1)
    H, W = dframe.shape
    n_clean = int(H * clean_ratio)
    df_clean = dframe.iloc[:n_clean, :]
    df_noise = dframe.iloc[n_clean:, :]
    df_clean = df_clean.sort_values(by=["image"])
    df_noise = df_noise.sort_values(by=["image"])
    df_clean.to_csv("config/data/jsrt_train_clean.csv", index = False)

    for i in range(H - n_clean):
        df_noise.iloc[i, 1] = df_noise.iloc[i, 1].replace(
            "label", "label_noise1")

    df_noise.to_csv("config/data/jsrt_train_noise.csv", index = False)
    df_mix = pd.concat([df_clean, df_noise], axis=0)
    df_mix.to_csv("config/data/jsrt_train_mix.csv", index = False)
    
def generate_noise(data_root, radius_range):
    output_dir = data_root + '/label_noise1'
    if(not os.path.isdir(output_dir)):
        os.mkdir(output_dir)
    noise_csv_name = "config/data/jsrt_train_noise.csv"
    df_noise = pd.read_csv(noise_csv_name)
    for i in range(df_noise.shape[0]):
        lab_name = df_noise.iloc[i, 1]
        input_name  = data_root + '/label/' + lab_name.split('/')[-1]
        output_name = data_root + '/' + lab_name
        print(input_name)
        lab = np.asarray(Image.open(input_name))
        lab_max = lab.max()
        lab = np.asarray(lab > 0, np.uint8)
        lab_noise = add_random_nosie_to_label(lab, radius_range) * lab_max
        lab_noise = Image.fromarray(lab_noise)
        lab_noise.save(output_name)

if __name__ == "__main__":
    random.seed(2022)
    np.random.seed(2022)
    # for exp1
    clean_ratio = 0.05
    radius_range = (6, 12)
    select_clean_samples(clean_ratio)
    generate_noise("./data", radius_range)
