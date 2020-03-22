import os
import shutil
import random
from pathlib import Path


images_path = Path("Path\to\images")
annotations_path = Path("Path\to\annotations")

images_test = Path("Path\to\test\images")
annotations_test = Path("Path\to\test\annotations")

files = [file for file in os.listdir(images_path) if file.endswith(".png")]

test = open("test.txt","w")

test_amount = round(0.15*len(files))

for x in range(test_amount):

    file = random.choice(files)
    files.remove(file)
    shutil.move(os.path.join(images_path, file), images_test)
    shutil.move(os.path.join(annotations_path, file), annotations_test)
    test.write(file + "\n")