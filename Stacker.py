import numpy as np
import os
from PIL import Image
from multiprocessing import Pool
from pathlib import Path
# import tqdm

Unet_path = Path("Path\to\Unet")
Deeplab_path = Path("Path\to\Deeplab")
ICnet_path = Path("Path\to\ICnet")

outputdir = Path("Path\to\outputs")

ids = [file for file in os.listdir(Unet_path) if file.endswith(".png")]

# for id in tqdm.tqdm(ids):
def stacking(id):

    Unet = np.array(Image.open(os.path.join(Unet_path, id)))[:,:,0]
    Deeplab = np.array(Image.open(os.path.join(Deeplab_path, id)))[:,:,0]
    ICnet = np.array(Image.open(os.path.join(ICnet_path, id)))[:,:,0]

    arrays = [Deeplab, Unet, ICnet]
    stacked = np.stack(arrays, axis=2)
    stacked = stacked * 50

    img = Image.fromarray(stacked, 'RGB')
    img.save(os.path.join(outputdir, id))
    print(id)

if __name__ == '__main__':
    pool = Pool(os.cpu_count()-2)
    pool.map(stacking, ids)