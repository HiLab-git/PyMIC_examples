import os
import numpy as np 
import SimpleITK as sitk 

def get_image_size_info():
    data_dir = "/home/disk2t/projects/PyMIC_project/PyMIC_data/ACDC/preprocess/"
    img_names = os.listdir(data_dir)
    img_names = [data_dir + '/' + item for item in img_names if \
        ("gt" not in item  and "scribble" not in item)]
    
    print("image number", len(img_names))
    shape_list = []
    spacing_list = []
    for img_name in img_names:
        img_obj = sitk.ReadImage(img_name)
        img = sitk.GetArrayFromImage(img_obj)
        spacing = img_obj.GetSpacing()
        [D, H, W] = img.shape
        print(D, H, W, spacing)
        shape_list.append([D, H, W])
        spacing_list.append(spacing)
    shape_array = np.asarray(shape_list)
    spacing_array = np.asarray(spacing_list)
    print("shape min", shape_array.min(axis = 0))
    print("shape max", shape_array.max(axis = 0))
    print("spacing min", spacing_array.min(axis = 0))
    print("spacing max", spacing_array.max(axis = 0))

if __name__ == "__main__":
    get_image_size_info()
