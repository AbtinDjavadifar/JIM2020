import os
import numpy as np
import imageio
from multiprocessing import Pool
from pathlib import Path

masks1_path = Path("Path/to/Masks1/")
masks2_path = Path("Path/to/Masks2/")
labels_path = Path("Path/to/Labels/")
annotations = [f for f in os.listdir(masks1_path) if f.endswith(".png")]
k = 30

def finding(name):
    try:
        im1 = np.array(imageio.imread(os.path.join(masks1_path,name)))
        im2 = np.array(imageio.imread(os.path.join(masks2_path, name[:-4]+'-annotation-2.png')))
        mask = np.stack([im1,im2], axis = -1)
        label = np.zeros(im1.shape)

        label[np.where(np.any(mask == 200, axis = -1))] = 2*k #Maybe Fabric
        label[np.where(np.all(mask == 200, axis=-1))] = 1 * k #Fabric

        label[np.where(np.any(mask == 100, axis = -1))] = 4 * k #Maybe Wrinkle
        label[np.where(np.all(mask == 100, axis=-1))] = 3 * k #Wrinkle

        label[np.where(np.any(mask == 50, axis = -1))] = 6 * k #Maybe Gripper
        label[np.where(np.all(mask == 50, axis=-1))] = 5 * k #Gripper

        imageio.imwrite(os.path.join(labels_path, name), label.astype(np.uint8))
        print("{} converted to masks".format(name))

    except IndexError:
        with open("broken.txt", "w") as outfile:
            outfile.write(name + "\n")
        pass

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(finding, annotations)


