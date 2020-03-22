import os
import numpy as np
import imageio
from pathlib import Path

name = '...'
labels_path = Path("...")
im = np.array(imageio.imread(os.path.join(labels_path, name)))
print(np.unique(im))