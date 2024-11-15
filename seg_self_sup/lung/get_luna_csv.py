import os 
import random
import pandas as pd

def get_luna_csv(luna_dir):
    random.seed(2023)    
    csv_dir  = "./config/data_pretrain"
    if(not os.path.exists(csv_dir)):
        os.mkdir(csv_dir)
    train_img_names = []
    valid_img_names = []
    for i in range(10):
        sub_dir = "subset{0:}".format(i)
        img_names = os.listdir(luna_dir + "/" + sub_dir)
        img_names = [sub_dir + "/" + item for item in img_names if ".nii.gz" in item]
        img_names = sorted(img_names)
        train_img_names.extend(img_names)
        if(i < 9):
            train_img_names.extend(img_names)
        else:
            valid_img_names.extend(img_names)

    df = {"image": train_img_names}
    df = pd.DataFrame.from_dict(df)
    df.to_csv(csv_dir + "/luna_train.csv", index=False)


    # get csv for pretraining
    print("valid image number", len(valid_img_names))
    random.shuffle(valid_img_names)
    valid_img_names = valid_img_names[:200]
    valid_img_names = [item.split("/")[-1] for item in valid_img_names]
    valid_lab_names = [item.replace(".nii.gz", "_lab.nii.gz") for item in valid_img_names]
    df = {"image": valid_img_names, "label":valid_lab_names}
    df = pd.DataFrame.from_dict(df)
    df.to_csv(csv_dir + "/luna_valid.csv", index=False)

luna_dir = "/home/disk4t/data/lung/LUNA2016/preprocess"
get_luna_csv(luna_dir)