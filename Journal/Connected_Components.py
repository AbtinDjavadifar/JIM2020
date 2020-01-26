import os
import numpy as np
import imageio
from multiprocessing import Pool
from pathlib import Path
import cv2

labels_path = Path("E:/Master-UBC/CVPR2020/Data/Real/Total/Wrinkle_Labels/")
CC_path = Path("E:/Master-UBC/CVPR2020/Data/Real/Total/Connected_Components/")
annotations = [f for f in os.listdir(labels_path) if f.endswith(".png")]
brkn = open("broken.txt","w")
k = 20

def connecting(name):
    try:
        im = np.array(imageio.imread(os.path.join(labels_path,name)))
        im[np.where(im == 2)] = 1
        ret, labels = cv2.connectedComponents(im)

        imageio.imwrite(os.path.join(CC_path, name), labels.astype(np.uint8)*k)
        print("{} converted to a set of connected components".format(name))

    except IndexError:
        brkn.write(name + "\n")
        pass

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(connecting, annotations)