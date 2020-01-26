import os
import numpy as np
import imageio
from multiprocessing import Pool
from pathlib import Path

# images_path = '/home/abtin/projects/def-najjaran/abtin/Data/DLR/Data/Virtual_Simulation/Images/'
annotations_path = Path("E:/Master-UBC/CVPR2020/Data/Real/2nd_Labeling/Annotations/")
masks_path = Path("E:/Master-UBC/CVPR2020/Data/Real/2nd_Labeling/Masks/")
annotations = [f for f in os.listdir(annotations_path) if f.endswith(".png")]
brkn = open("broken.txt","w")

def masking(name):
    try:
        im = np.array(imageio.imread(os.path.join(annotations_path,name)))
        mask = np.argmax(im[:,:,:3], axis=2)
        np.place(mask, mask==0, 200)
        np.place(mask, mask==1, 100)
        np.place(mask, mask==2, 50)
        mask[np.where(np.argmax(im[:,:,:3], axis=2) == np.argmin(im[:,:,:3], axis=2))] = 0

        imageio.imwrite(os.path.join(masks_path, name), mask.astype(np.uint8))
        print("{} converted to masks".format(name))

    except IndexError:
        brkn.write(name + "\n")
        pass

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(masking, annotations)
