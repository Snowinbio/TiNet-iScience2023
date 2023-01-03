# According to the nodule label, the nodule image is intercepted; 
# At the same time, the aspect ratio of the nodule are calculated;
import os
from skimage import io,transform,color,img_as_ubyte,feature,data,filters

def get_nodule_ratio(path):
    # jpg_img has a uniform image size
    img=io.imread(path)
    img=np.array(img,copy=False)
    img=transform.resize(image=img, output_shape=jpg_img.shape) * 255
    img_dila=sm.dilation(img,sm.square(dilation_value))
    img_dila=(img_dila-np.min(img_dila))/(np.max(img_dila)-np.min(img_dila))*255
    rows, cols, jpg_img_values= [], [], []
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row][col]>125:
                rows.append(row)
                cols.append(col)
                jpg_img_values.append(jpg_img[row][col])
                img_bound_values.append(img_bound[row][col])
            else:
                back_bound_values.append(img_bound[row][col])
    if len(jpg_img_values)==0:
        empty_list.append(jpg_path)
        continue
        
    # calculate the aspect ratio
    row_max, row_min, col_max, col_min = max(rows), min(rows), max(cols), min(cols)
    return (col_max-col_min+1)/(row_max-row_min+1)
