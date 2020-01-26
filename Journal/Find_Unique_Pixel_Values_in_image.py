import os
import numpy as np
import imageio
from pathlib import Path

name = 'Cam_1_Pic_19-07-43.png'
labels_path = Path("E:/Master-UBC/CVPR2020/Data/Real/Total/Connected_Components/")
im = np.array(imageio.imread(os.path.join(labels_path, name)))
print(np.unique(im))