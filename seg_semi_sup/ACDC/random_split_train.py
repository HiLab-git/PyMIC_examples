import os
import random

def random_select_labeled_images(lab_percent = 10):
    patient_num = 70
    lab_patient = patient_num * lab_percent/100 # number of labeled patients
    img_num = 140
    print("labeled images {0:}, unlabeled images {1:}".format(
        lab_patient*2, img_num - lab_patient*2))
    with open("config/data/image_train.csv", 'r') as f:
        lines = f.readlines()
    assert(img_num == len(lines) - 1)
    idx_list = list(range(patient_num))
    random.shuffle(idx_list)
    data_lab, data_unlab = [], []
    for i in range(patient_num):
        idx = idx_list[i]
        if(i < lab_patient):
            data_lab.extend(lines[2*idx+1:2*idx+3])
        else:
            data_unlab.extend(lines[2*idx+1:2*idx+3])
    data_lab = [lines[0]] + sorted(data_lab)
    data_unlab = [lines[0]] + sorted(data_unlab)
    with open("config/data/image_train_r{0:}_lab.csv".format(lab_percent), 'w') as f:
        f.writelines(data_lab)
    with open("config/data/image_train_r{0:}_unlab.csv".format(lab_percent), 'w') as f:
        f.writelines(data_unlab)
        
if __name__ == "__main__":
    random.seed(2022)
    # lab_percent: the percentage of labeled samples (0-100)
    random_select_labeled_images(10)
