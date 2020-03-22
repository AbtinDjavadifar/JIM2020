import imageio
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import os

src_path = Path("Path\to\source")
dest_path = Path("Path\to\destination")
files = [file for file in os.listdir(src_path) if file.endswith(".png")]

def converting(file):
    im = np.array(imageio.imread(os.path.join(src_path, file)))
    im[np.where(im == 100)] = 0
    im[np.where(im == 50)] = 150
    imageio.imwrite(os.path.join(dest_path, file[:-4] + '.png'), im.astype(np.uint8))

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(converting, files)

