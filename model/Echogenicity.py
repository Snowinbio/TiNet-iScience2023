# According to the nodule label, the nodule image is intercepted; 
# At the same time, the echo characteristics of the nodule are calculated;
import os
from skimage import io,transform,color,img_as_ubyte,feature,data,filters
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import matplotlib.mlab as mlab
from scipy.stats import norm

def get_nodule_echo(path):
    #Load nodule labels and inflate
    # jpg_img has a uniform image size
    img=io.imread(path)
    img=np.array(img,copy=False)
    img=transform.resize(image=img, output_shape=jpg_img.shape) * 255
    img_dila=sm.dilation(img,sm.square(dilation_value))
    img_dila=(img_dila-np.min(img_dila))/(np.max(img_dila)-np.min(img_dila))*255
    
    rows, cols, jpg_img_values, jpg_bound_values = [], [], [], []
    img_bound_values, back_bound_values= [], []
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img_dila[row][col]>125 and img[row][col]<125:
                jpg_bound_values.append(jpg_img[row][col]) 
                back_bound_values.append(img_bound[row][col]) 
            elif img[row][col]>125:
                rows.append(row)
                cols.append(col)
                jpg_img_values.append(jpg_img[row][col])
                img_bound_values.append(img_bound[row][col])
            else:
                back_bound_values.append(img_bound[row][col])
    if len(jpg_img_values)==0:
        empty_list.append(jpg_path)
        continue
        
    canny_ratio = np.mean(img_bound_values)
    back_canny_ratio = np.mean(back_bound_values)
    
    nodule_echo = np.mean(jpg_img_values)
    bound_echo = np.mean(jpg_bound_values)
    echo_sub = bound_echo - nodule_echo
    return echo_sub

# Echo difference feature calculated by Gaussian function
# benign_echo_sub_list: list of echo_sub for benign nodule
# malignant_echo_sub_list: list of echo_sub for malignant nodule
# echo_sub_list: list of echo_sub for all nodule

x_benign = np.array(benign_echo_sub_list)
mu_benign =np.mean(x_benign)
sigma_benign =np.std(x_benign)

x_malignant = np.array(malignant_echo_sub_list)
mu_malignant =np.mean(x_malignant)
sigma_malignant =np.std(x_malignant)

p_norm_echo=[]
for i in range(len(echo_sub_list)):
    p_norm_echo.append(norm.pdf(i, mu_malignant, sigma_malignant)-norm.pdf(i, mu_benign, sigma_benign))