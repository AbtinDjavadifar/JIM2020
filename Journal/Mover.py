import os
import shutil
from pathlib import Path


src_path = Path("Path\to\source")
dest_path = Path("Path\to\destination")

images = Path("Path\to\images")

files = [file for file in os.listdir(images) if file.endswith(".png")]


for file in files:

    shutil.move(os.path.join(src_path, file), dest_path)
