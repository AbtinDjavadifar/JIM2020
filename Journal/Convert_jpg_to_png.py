from PIL import Image
from pathlib import Path
from multiprocessing import Pool
import os

dest_path = Path(r'E:\Master-UBC\CVPR2020\Data\Real\2nd_Labeling\NotIncludedinDataset')
src_path = Path(r'E:\Master-UBC\DLR\Tests - Jan 2019\1. Dataset\GripperStereoPics\def_ply011')
files = [file for file in os.listdir(src_path) if file.endswith(".jpg")]

def converting(file):
    im = Image.open(os.path.join(src_path, file))
    im.save(os.path.join(dest_path, file[:-4] + '.png'))

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(converting, files)

