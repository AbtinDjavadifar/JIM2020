# coding: utf-8

import json
import cv2
import numpy as np
import os
import sys

from os.path import join
from PIL import Image
from typing import List, Dict

import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file

classes_mapping =  {
    "Gripper": 206,
    "Wrinkle": 156,
    "Fabric": 56
}

np.set_printoptions(threshold=sys.maxsize)

mask_path = "/home/abtin/Abtin/VS3_0035.png"

mask = cv2.imread(mask_path)[:, :, 0]

for cls_name, color in classes_mapping.items():

    if isinstance(color, int):
        bool_mask = mask == color
        print("color: {}".format(color))
    elif isinstance(color, list):
        bool_mask = np.isin(mask, color)
    else:
        raise ValueError('Wrong color format. It must be integer, list of integers or special key string "__all__".')

    bitmap = sly.Bitmap(data=bool_mask)
    print(bitmap)

