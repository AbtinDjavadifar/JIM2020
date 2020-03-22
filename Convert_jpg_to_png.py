from PIL import Image
from pathlib import Path
from multiprocessing import Pool
import os

src_path = Path("Path\to\source")
dest_path = Path("Path\to\destination")
files = [file for file in os.listdir(src_path) if file.endswith(".jpg")]

def converting(file):
    im = Image.open(os.path.join(src_path, file))
    im.save(os.path.join(dest_path, file[:-4] + '.png'))

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(converting, files)
