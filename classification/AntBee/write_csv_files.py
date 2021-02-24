"""Script for writing cvs files
"""

import os
import csv
import pandas as pd
import random
from random import shuffle

def create_csv_file(data_root, stage, output_csv):
    """
    create a csv file to store the paths of files for each image
    """
    class_names = ['ants', 'bees']
    file_class_items = []
    for idx in range(len(class_names)):
        one_cls = class_names[idx]
        file_dir = data_root + '/' + stage + '/' + one_cls
        file_names = os.listdir(file_dir)
        file_names = [item for item in file_names if ".jpg" in item]
        for file_name in file_names:
            item = [stage + '/' + one_cls + '/' + file_name, idx]
            file_class_items.append(item)
        print('class {0:}:'.format(one_cls), len(file_names))
    
    print('total image number', len(file_class_items))
    with open(output_csv, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', 
                            quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['image', 'label'])
        for item in file_class_items:
            csv_writer.writerow(item)


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
    # create cvs file for JSRT dataset
    AntBee_root   = '/home/guotai/disk2t/projects/torch_project/transfer_learning/hymenoptera_data'
    create_csv_file(AntBee_root, 'train', 'config/train_data.csv')
    create_csv_file(AntBee_root, 'val', 'config/valid_data.csv')

