
import numpy as np
import os
from PIL import Image
# import tqdm
from multiprocessing import Pool
from pathlib import Path

Unet_path = Path(r'C:\Users\djava\Desktop\UN AC test\ds\masks_machine')
Deeplab_path = Path(r'C:\Users\djava\Desktop\DL AC test\ds\masks_machine')
ICnet_path = Path(r'C:\Users\djava\Desktop\IC AC test\ds\masks_machine')

outputdir = Path(r'C:\Users\djava\Desktop\Stacked\Test\img')

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