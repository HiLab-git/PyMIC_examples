import os 
import math
import numpy as np 
import SimpleITK as sitk 
from pymic.util.image_process import *
from pymic.io.image_read_write import *

def luna_preprocess(input_dir, output_dir):
    """
    Crop the images to a smaller subvolumes to speedup the data loading process during training.
    Also clip the intensity to the range of [-1000, 1000] and normalize to [-1, 1].
    """
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
    for idx in range(10):
        print("*****subset**** ", idx)
        sub_dir = input_dir + "/subset{0:}".format(idx)
        sub_out_dir = output_dir + "/subset{0:}".format(idx)
        if(not os.path.exists(sub_out_dir)):
            os.mkdir(sub_out_dir)
        img_names = os.listdir(sub_dir)
        img_names = [item for item in img_names if ".nii.gz" in item]
        print("image number", len(img_names))
        num_subvlume = 0
        for img_name in img_names:
            # print(img_name)
            input_name = sub_dir + "/" + img_name
            img_obj = sitk.ReadImage(input_name)
            spacing = img_obj.GetSpacing()
            space_out = [spacing[2], spacing[0], spacing[1]]
            img = sitk.GetArrayFromImage(img_obj)
            bb_min, bb_max = get_human_region_from_ct(img)
            img = crop_ND_volume_with_bounding_box(img, bb_min, bb_max)

            # normalize the image to  [-1, 1]
            img = np.clip(img, -1000, 1000)
            img = (img + 1000.0 ) / 2000.0 * 2 - 1.0
            
            # pad to minimial shape
            patch_shape = [96, 192, 192]
            stride      = [96, 96, 96]
            img_shape = img.shape
            pad_shape = [max(patch_shape[i], img_shape[i]) for i in range(3)]
            mgnp = [pad_shape[i] - img_shape[i] for i in range(3)]
            if(max(mgnp) > 0):
                ml  = [int(mgnp[i]/2)  for i in range(3)]
                mr  = [mgnp[i] - ml[i] for i in range(3)] 
                pad = [(ml[i], mr[i])  for i in range(3)]
                pad = tuple(pad)
                img = np.pad(img, pad, 'reflect') 
            
            # an initial crop
            crop_shape0 = [math.floor(img.shape[i]/stride[i])*stride[i] \
                for i in range(3)]
            img = random_crop_ND_volume(img, crop_shape0)

            # sequential crop
            D, H, W = img.shape
            k = 0
            for d in range(0, D-patch_shape[0] + 1, stride[0]):
                for h in range(0, H-patch_shape[1] + 1, stride[1]):
                    for w in range(0, W-patch_shape[2] + 1, stride[2]):
                        img_sub = img[d:d+patch_shape[0], h:h+patch_shape[1], w:w+patch_shape[2]]
                        k = k + 1
                        out_name = img_name.replace(".nii.gz", "_{0:}.nii.gz".format(k))
                        save_array_as_nifty_volume(img_sub, sub_out_dir + "/" + out_name, 
                            reference_name = None, spacing = space_out)
            print(img_name, k)
            num_subvlume = num_subvlume + k 
        print("total subvolume number", num_subvlume)

luna_dir   = "/home/disk4t/data/lung/LUNA2016/"   
input_dir  = luna_dir + "/raw_nii"
output_dir = luna_dir + "/preprocess"        
luna_preprocess(input_dir, output_dir)
