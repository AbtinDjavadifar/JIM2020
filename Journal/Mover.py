import os
import shutil
from pathlib import Path


src_path = Path(r'C:\Users\djava\Desktop\Upload to Supervisely for Journal\AC\Masks')
dest_path = Path(r'C:\Users\djava\Desktop\Upload to Supervisely for Journal\AC\Train\ann')

images_test = Path(r'C:\Users\djava\Desktop\Upload to Supervisely for Journal\AC\Train\img')

files = [file for file in os.listdir(images_test) if file.endswith(".png")]


for file in files:

    shutil.move(os.path.join(src_path, file), dest_path)
