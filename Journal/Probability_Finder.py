import os
import numpy as np
import imageio
from multiprocessing import Pool
from pathlib import Path

labels_path = Path("Path\to\labels")
CC_path = Path("Path\to\ConnectedComponents")

probability_path = Path("Path\to\Probabilities")
annotations = [f for f in os.listdir(labels_path) if f.endswith(".png")]

def calculating(name):
    try:
        label = np.array(imageio.imread(os.path.join(labels_path,name)))
        cc = np.array(imageio.imread(os.path.join(CC_path,name)))
        probability = np.zeros(label.shape)
        for val in np.unique(cc)[1:]:
            probability[np.where(cc == val)] = np.sum(np.logical_and((label == 50), (cc == val))) / np.sum(cc == val)
        print('The probability values for {}: {}'.format(name, np.unique(probability)))

        np.save(os.path.join(probability_path, name[:-4]), probability)
        print("{} converted to probability map".format(name))

    except IndexError:
        with open("broken.txt", "w") as outfile:
            outfile.write(name + "\n")
        pass

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(calculating, annotations)
