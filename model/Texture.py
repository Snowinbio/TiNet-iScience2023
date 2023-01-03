# 将json解析后的png重命名，并转移到指定文件夹中，形成mask和截取的数据
import os
from skimage import io,transform,color, img_as_ubyte
import numpy as np
import re

hos_path = "data"
mali_shape = "benign"
jpg_dir_path = "D:\\Code\\Follocular\\"+hos_path+"\\"+mali_shape
for jpg_path in os.listdir(jpg_dir_path):
    json_dir = jpg_path.replace(".JPG","").replace(".jpg","").replace(".","_")+"_json"
    origin_path=jpg_dir_path+"_image_json_dir\\"+json_dir+"\\label.png"
    img=io.imread(origin_path)
    img=np.array(img,copy=False)
    if len(img.shape)!=2:
        img=color.rgb2gray(color.rgba2rgb(img))
    # print(np.max(img))
    img=np.where(img>0,255,0)
    img=img_as_ubyte(img)
    save_path=jpg_dir_path+"_image\\"+jpg_path
    io.imsave(arr=img, fname=save_path)