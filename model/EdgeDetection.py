from PIL import Image
import numpy as np
import os
import csv
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# read img and get the box data for edge feature.

def detection(img_dir, det_dir, dst_dir):
    img_list = os.listdir(img_dir)
    box_list = []

    for i in img_list:
        im = Image.open(os.path.join(img_dir, i))
        j = i[:-3] + 'png'
        de = Image.open(os.path.join(det_dir, j))

        det = np.array(de)
        nz = np.nonzero(det)

        row_min = int(np.min(nz[0]))
        row_max = int(np.max(nz[0]))
        col_min = int(np.min(nz[1]))
        col_max = int(np.max(nz[1]))

        box = (col_min, row_min, col_max, row_max)

        im_c = im.crop(box)

        im_c.save(os.path.join(dst_dir, j))

        im_box = (i, col_min, row_min, col_max, row_max)
        box_list.append(im_box)

    with open(os.path.join(dst_dir, "box.csv"), 'w') as f:
        writer = csv.writer(f)
        for i in box_list:
            writer.writerow(i)

    return 0
