"""Script for writing cvs files
"""

import os
import csv
import random
from random import shuffle

def create_csv_file(data_root, output_file):
    """
    create a csv file to store the paths of files for each patient
    """
    filenames = []
    image_names = os.listdir(data_root + "/image")
    image_names.sort()
    print('total number of images {0:}'.format(len(image_names)))
    for item in image_names:
        image_name = "image/"+item
        label_name = "label/"+item
        filenames.append([image_name, label_name])

    fields = ["image", "label"]
    with open(output_file, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(fields)
        for item in filenames:
            csv_writer.writerow(item)

def random_split_dataset():
    random.seed(1224)
    input_file = 'config/data_lctsc/image_all.csv'
    train_names_file = 'config/data_lctsc/image_train.csv'
    valid_names_file = 'config/data_lctsc/image_valid.csv'
    test_names_file  = 'config/data_lctsc/image_test.csv'
    with open(input_file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:]
    train_val_lines, test_lines = [], []
    for line in data_lines:
        if("Test" in  line):
            test_lines.append(line)
        else:
            train_val_lines.append(line)    

    N = len(train_val_lines)
    n1 = int(N * 0.75)
    print('training number', n1)
    print('validation number', N - n1)
    print('testing number', len(test_lines))
    shuffle(train_val_lines)
    train_lines  = sorted(train_val_lines[:n1])
    valid_lines  = sorted(train_val_lines[n1:])
    with open(train_names_file, 'w') as f:
        f.writelines(lines[:1] + train_lines)
    with open(valid_names_file, 'w') as f:
        f.writelines(lines[:1] + valid_lines)
    with open(test_names_file, 'w') as f:
        f.writelines(lines[:1] + test_lines)    

def get_evaluation_image_pairs(test_csv, gt_seg_csv):
    with open(test_csv, 'r') as f:
        input_lines = f.readlines()[1:]
        output_lines = []
        for item in input_lines:
            gt_name = item.split(',')[1]
            gt_name = gt_name.rstrip()
            seg_name = gt_name.split('/')[-1]
            output_lines.append([gt_name, seg_name])
    with open(gt_seg_csv, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["ground_truth", "segmentation"])
        for item in output_lines:
            csv_writer.writerow(item)


if __name__ == "__main__":
    data_dir    = '../../PyMIC_data/LCTSC2017'
    output_file = 'config/data_lctsc/image_all.csv'
    create_csv_file(data_dir, output_file)

    # split the data into training, validation and testing
    random_split_dataset()

    # obtain ground truth and segmentation pairs for evaluation
    test_csv    = "config/data_lctsc/image_test.csv"
    gt_seg_csv  = "config/data_lctsc/image_test_gt_seg.csv"
    get_evaluation_image_pairs(test_csv, gt_seg_csv)

