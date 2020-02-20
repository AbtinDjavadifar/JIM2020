import os
import numpy as np
import imageio
from multiprocessing import Pool
from pathlib import Path
import cv2

labels_path = Path(r"C:\Users\djava\Desktop\Test\ann")
CC_path = Path(r"C:\Users\djava\Desktop\GT CC")
annotations = [f for f in os.listdir(labels_path) if f.endswith(".png")]
brkn = open("broken.txt","w")
k = 20

def connecting(name):
    try:
        im = cv2.imread(os.path.join(labels_path, name), cv2.IMREAD_GRAYSCALE)
        im[np.where(im == 100)] = 1

        ret, labels = cv2.connectedComponents(im)
        imageio.imwrite(os.path.join(CC_path, name), labels.astype(np.uint8)*k)
        print("{} converted to a set of connected components".format(name))

    except IndexError:
        brkn.write(name + "\n")
        pass

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(connecting, annotations)