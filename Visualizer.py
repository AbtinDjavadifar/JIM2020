
import numpy as np
import os
from PIL import Image
from pathlib import Path

path_dict = {'UNet':Path('E:/Common/UNet_Real_Test/ds/masks_machine'),
             'Deeplab':Path('E:/Common/DeepLabV3_Real_Test/ds/masks_machine'),
             'ICnet':Path('E:/Common/ICNet_Real_Test/ds/masks_machine'),
             'MaskRCNN':Path('E:/Common/MaskRCNN_Real_Test/ds/masks_machine')}

outputdir = Path('E:/Common/visualization')

models = ['Unet', 'Deeplab', 'ICnet', 'MaskRCNN']
ids = ['Cam_1_Pic_19-08-57.png', 'Cam_2_Pic_19-09-14.png', 'Cam_3_Pic_19-09-00.png', 'Cam_4_Pic_19-10-02.png', 'Cam_5_Pic_19-09-55.png']

for model, path in path_dict.items():
    for id in ids:

        img = Image.open(os.path.join(path, id))
        img = img.convert("RGBA")
        img = np.array(img)
        h, w, d = img.shape

        img = img.reshape((h*w), d)

        img[(img[:,0] == 0)] = [0, 0, 0, 0]
        img[(img[:,0] == 2)] = [31, 119, 180, 255]
        img[(img[:,0] == 3)] = [44, 160, 44, 255]
        img[(img[:,0] == 4)] = [255, 127, 14, 255]

        img = img.reshape(h,w,d)

        img = Image.fromarray(img, 'RGBA')

        id = "{}_{}".format(model,id)
        print(id)
        img.save(os.path.join(outputdir, id))

