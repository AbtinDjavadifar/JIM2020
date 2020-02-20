import imageio
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import os


src_path = Path(r'E:\Master-UBC\CVPR2020\Data For Upload to Supervisely\Wrinkle classes\Test\ann')
dest_path = Path(r'C:\Users\djava\Desktop\Upload to Supervisely for Journal\W\Test\ann')
files = [file for file in os.listdir(src_path) if file.endswith(".png")]

def converting(file):
    im = np.array(imageio.imread(os.path.join(src_path, file)))
    im[np.where(im == 100)] = 0
    im[np.where(im == 50)] = 150
    imageio.imwrite(os.path.join(dest_path, file[:-4] + '.png'), im.astype(np.uint8))

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(converting, files)

