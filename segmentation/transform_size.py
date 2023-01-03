from skimage import io
from skimage import transform
import numpy as np
import os
import shutil

# This script helps to convert the image into three channels

total_path=""
for img_path in os.listdir(total_path+"/images"):
    img=io.imread(total_path+"/images/"+img_path)
    if len(img.shape)==2:
        img = np.stack([img] * 3, axis=2)
    io.imsave(arr=img,fname=total_path+"/images/"+img_path)
for img_path in os.listdir(total_path+"/masks"):
    img=io.imread(total_path+"/masks/"+img_path)
    if len(img.shape)==2:
        img = np.stack([img] * 3, axis=2)
    io.imsave(arr=img,fname=total_path+"/masks/"+img_path)

total_path=""
for img_path in os.listdir(total_path+"/images"):
    shutil.move(total_path+"/images/"+img_path,total_path+"/images/"+img_path.replace(".JPG",".jpg"))

for img_path in os.listdir(total_path+"/masks"):
    shutil.move(total_path+"/masks/"+img_path,total_path+"/masks/"+img_path.replace(".JPG",".jpg"))


