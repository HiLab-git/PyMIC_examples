"""Script for writing cvs files
"""

import os
import csv
import pandas as pd
import random
import numpy as np 
from random import shuffle

def create_csv_file(image_dir, output_csv):
    """
    create a csv file to store the paths of files for each patient
    """
    img_names = os.listdir(image_dir)
    img_names = [item for item in img_names if ".png" in item]
    img_names = sorted(img_names)
    name_lab_list = []
    for img_name in img_names:
        lab = 0 if("0.png" in img_name) else 1
        name_lab_list.append([img_name, lab])
        
    with open(output_csv, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['image', 'label'])
        for item in name_lab_list:
            csv_writer.writerow(item)

def random_split_dataset():
    random.seed(2021)
    input_file = 'config/cxr_all.csv'
    train_names_file = 'config/cxr_train.csv'
    valid_names_file = 'config/cxr_valid.csv'
    test_names_file  = 'config/cxr_test.csv'
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    img_num = len(data_lines)
    idx = list(range(img_num))
    shuffle(idx)
    num1 = int(img_num * 0.7)
    num2 = int(img_num * 0.8)
    train_idx = sorted(idx[:num1])
    valid_idx = sorted(idx[num1:num2])
    test_idx  = sorted(idx[num2:])

    train_lines  = [data_lines[i] for i in train_idx]
    valid_lines  = [data_lines[i] for i in valid_idx]
    test_lines   = [data_lines[i] for i in test_idx]
    with open(train_names_file, 'w') as f:
        f.writelines(lines[:1] + train_lines)
    with open(valid_names_file, 'w') as f:
        f.writelines(lines[:1] + valid_lines)
    with open(test_names_file, 'w') as f:
        f.writelines(lines[:1] + test_lines)

  
if __name__ == "__main__":
    # create cvs file for ISIC dataset
    image_dir   = '../../PyMIC_data/CHNCXR/CXR_png'
    output_csv  = 'config/cxr_all.csv'
    create_csv_file(image_dir, output_csv)

    # split the dataset in to training, validation and testing
    random_split_dataset()
