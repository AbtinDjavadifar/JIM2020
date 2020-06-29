

def convert_annotations_to_masks(annotations_path, masks_path):

    import os
    import numpy as np
    import imageio
    from multiprocessing import Pool
    from pathlib import Path

    annotations = [f for f in os.listdir(annotations_path) if f.endswith(".png")]

    def masking(name):
        try:
            im = np.array(imageio.imread(os.path.join(annotations_path,name)))
            mask = np.argmax(im[:,:,:3], axis=2)
            np.place(mask, mask==0, 200)
            np.place(mask, mask==1, 100)
            np.place(mask, mask==2, 50)
            mask[np.where(np.argmax(im[:,:,:3], axis=2) == np.argmin(im[:,:,:3], axis=2))] = 0

            imageio.imwrite(os.path.join(masks_path, name), mask.astype(np.uint8))
            print("{} converted to masks".format(name))

        except IndexError:
            with open("broken.txt", "w") as outfile:
                outfile.write(name + "\n")
            pass

    if __name__ == '__main__':
        pool = Pool(os.cpu_count()-2)
        pool.map(masking, annotations)

def binary_mask_value_finder():

    import json
    import cv2
    import numpy as np
    import os
    import sys

    from os.path import join
    from PIL import Image
    from typing import List, Dict

    import supervisely_lib as sly
    from supervisely_lib.io.json import load_json_file

    classes_mapping = {
        "Gripper": 206,
        "Wrinkle": 156,
        "Fabric": 56
    }

    np.set_printoptions(threshold=sys.maxsize)

    mask_path = "./VS3_0035.png"

    mask = cv2.imread(mask_path)[:, :, 0]

    for cls_name, color in classes_mapping.items():

        if isinstance(color, int):
            bool_mask = mask == color
            print("color: {}".format(color))
        elif isinstance(color, list):
            bool_mask = np.isin(mask, color)
        else:
            raise ValueError(
                'Wrong color format. It must be integer, list of integers or special key string "__all__".')

        bitmap = sly.Bitmap(data=bool_mask)
        print(bitmap)

def connect_components(labels_path, connected_components_path, k=20):

    import os
    import numpy as np
    import imageio
    from multiprocessing import Pool
    from pathlib import Path
    import cv2

    annotations = [f for f in os.listdir(labels_path) if f.endswith(".png")]

    def connecting(name):
        try:
            # im = np.array(imageio.imread(os.path.join(labels_path,name)))
            im = cv2.imread(os.path.join(labels_path, name), cv2.IMREAD_GRAYSCALE)
            im[np.where(im == 100)] = 1

            ret, labels = cv2.connectedComponents(im)
            imageio.imwrite(os.path.join(connected_components_path, name), labels.astype(np.uint8) * k)
            print("{} converted to a set of connected components".format(name))

        except IndexError:
            with open("broken.txt", "w") as outfile:
                outfile.write(name + "\n")
            pass

    if __name__ == '__main__':
        pool = Pool(os.cpu_count() - 2)
        pool.map(connecting, annotations)

def convert_jpg_to_png(src_path, dest_path):

    from PIL import Image
    from pathlib import Path
    from multiprocessing import Pool
    import os

    files = [file for file in os.listdir(src_path) if file.endswith(".jpg")]

    def converting(file):
        im = Image.open(os.path.join(src_path, file))
        im.save(os.path.join(dest_path, file[:-4] + '.png'))

    if __name__ == '__main__':
        pool = Pool(os.cpu_count() - 2)
        pool.map(converting, files)

def calculate_F1_score(DCNN_connected_components_path, ground_truth_connected_components_path):

    import os
    import numpy as np
    import imageio
    from pathlib import Path
    import pandas as pd

    size_threshold = 0.2
    # overlapping_threshold = 0.7

    scores = pd.DataFrame(columns=['Overlapping threshold', 'Average precision', 'Average recall', 'Average F1'])

    annotations = [f for f in os.listdir(ground_truth_connected_components_path) if f.endswith(".png")]

    for overlapping_threshold in np.arange(0, 1.05, 0.05):

        Precision_Avg = 0
        Recall_Avg = 0
        F1_Avg = 0

        for name in annotations:
            GT = np.array(imageio.imread(os.path.join(ground_truth_connected_components_path, name)))
            DCNN = np.array(imageio.imread(os.path.join(DCNN_connected_components_path, name)))
            GT_values = []
            DCNN_values = []
            GT_wrinkles = list(np.unique(GT))
            GT_wrinkles.remove(0)
            DCNN_wrinkles = list(np.unique(DCNN))
            DCNN_wrinkles.remove(0)
            if GT_wrinkles == []:
                Precision = 1
                Recall = 1
                F1 = 1
            else:
                size_avg = 0
                for val in GT_wrinkles:
                    size = sum(GT[np.where(GT == val)])
                    size_avg += size
                size_avg = size_avg / len(GT_wrinkles)
                # for val in GT_wrinkles:
                #     if sum(GT[np.where(GT == val)]) < (size_threshhold * size_avg):
                #         GT_wrinkles.remove(val)
                for val in GT_wrinkles:

                    target_area = DCNN[np.where(GT == val)]
                    wrinkle_coverage = np.count_nonzero(target_area) / len(target_area)
                    if wrinkle_coverage > overlapping_threshold:
                        GT_values.append(val)
                        DCNN_values.extend(np.unique(target_area))

                DCNN_values = list(np.trim_zeros(np.unique(DCNN_values)))

                for val in list(set(DCNN_wrinkles) - set(DCNN_values)):
                    if sum(DCNN[np.where(DCNN == val)]) < (size_threshold * size_avg):
                        DCNN_wrinkles.remove(val)

                TP = len(GT_values)
                FN = len(GT_wrinkles) - len(GT_values)
                FP = len(DCNN_wrinkles) - len(DCNN_values)
                Precision = TP / (TP + FP)
                Recall = TP / (TP + FN)
                if Precision == 0 and Recall == 0:
                    F1 = 0
                else:
                    F1 = 2 * Precision * Recall / (Precision + Recall)
            Precision_Avg += Precision
            Recall_Avg += Recall
            F1_Avg += F1

            # print("{}: Precision: {} , Recall: {} , F1: {}".format(name,Precision,Recall,F1))

        Precision_Avg = Precision_Avg / len(annotations)
        Recall_Avg = Recall_Avg / len(annotations)
        F1_Avg = F1_Avg / len(annotations)

        print("Overlapping threshold: {} , Average Precision: {} , Average Recall: {} , Average F1: {}".format(
            overlapping_threshold, Precision_Avg, Recall_Avg, F1_Avg))
        scores = scores.append({'Overlapping threshold': overlapping_threshold, 'Average precision': Precision_Avg,
                                'Average recall': Recall_Avg, 'Average F1': F1_Avg}, ignore_index=True)

    scores.to_csv('scores.csv')

def find_unique_pixel_values_in_image(name, labels_path):

    import os
    import numpy as np
    import imageio
    from pathlib import Path

    print(np.unique(np.array(imageio.imread(os.path.join(labels_path, name)))))

def find_union_and_intersection(masks1_path, masks2_path, labels_path, k = 30):

    import os
    import numpy as np
    import imageio
    from multiprocessing import Pool
    from pathlib import Path

    annotations = [f for f in os.listdir(masks1_path) if f.endswith(".png")]

    def finding(name):
        try:
            im1 = np.array(imageio.imread(os.path.join(masks1_path, name)))
            im2 = np.array(imageio.imread(os.path.join(masks2_path, name[:-4] + '-annotation-2.png')))
            mask = np.stack([im1, im2], axis=-1)
            label = np.zeros(im1.shape)

            label[np.where(np.any(mask == 200, axis=-1))] = 2 * k  # Maybe Fabric
            label[np.where(np.all(mask == 200, axis=-1))] = 1 * k  # Fabric

            label[np.where(np.any(mask == 100, axis=-1))] = 4 * k  # Maybe Wrinkle
            label[np.where(np.all(mask == 100, axis=-1))] = 3 * k  # Wrinkle

            label[np.where(np.any(mask == 50, axis=-1))] = 6 * k  # Maybe Gripper
            label[np.where(np.all(mask == 50, axis=-1))] = 5 * k  # Gripper

            imageio.imwrite(os.path.join(labels_path, name), label.astype(np.uint8))
            print("{} converted to masks".format(name))

        except IndexError:
            with open("broken.txt", "w") as outfile:
                outfile.write(name + "\n")
            pass

    if __name__ == '__main__':
        pool = Pool(os.cpu_count() - 2)
        pool.map(finding, annotations)

def move_files(src_path, dest_path, images):

    import os
    import shutil
    from pathlib import Path

    files = [file for file in os.listdir(images) if file.endswith(".png")]

    for file in files:
        shutil.move(os.path.join(src_path, file), dest_path)

def calculate_probability_score(connected_components_path, labels_path, probability_path):

    import os
    import numpy as np
    import imageio
    from multiprocessing import Pool
    from pathlib import Path

    annotations = [f for f in os.listdir(labels_path) if f.endswith(".png")]

    def calculating(name):
        try:
            label = np.array(imageio.imread(os.path.join(labels_path, name)))
            cc = np.array(imageio.imread(os.path.join(connected_components_path, name)))
            probability = np.zeros(label.shape)
            for val in np.unique(cc)[1:]:
                probability[np.where(cc == val)] = np.sum(np.logical_and((label == 50), (cc == val))) / np.sum(
                    cc == val)
            print('The probability values for {}: {}'.format(name, np.unique(probability)))

            np.save(os.path.join(probability_path, name[:-4]), probability)
            print("{} converted to probability map".format(name))

        except IndexError:
            with open("broken.txt", "w") as outfile:
                outfile.write(name + "\n")
            pass

    if __name__ == '__main__':
        pool = Pool(os.cpu_count() - 2)
        pool.map(calculating, annotations)

def single_class(src_path, dest_path):

    import imageio
    from pathlib import Path
    from multiprocessing import Pool
    import numpy as np
    import os

    files = [file for file in os.listdir(src_path) if file.endswith(".png")]

    def converting(file):
        im = np.array(imageio.imread(os.path.join(src_path, file)))
        im[np.where(im == 100)] = 0
        im[np.where(im == 50)] = 150
        imageio.imwrite(os.path.join(dest_path, file[:-4] + '.png'), im.astype(np.uint8))

    if __name__ == '__main__':
        pool = Pool(os.cpu_count() - 2)
        pool.map(converting, files)

def stack_images(Unet_path, Deeplab_path, ICnet_path, stacked_path):

    import numpy as np
    import os
    from PIL import Image
    from multiprocessing import Pool
    from pathlib import Path
    # import tqdm

    ids = [file for file in os.listdir(Unet_path) if file.endswith(".png")]

    # for id in tqdm.tqdm(ids):
    def stacking(id):
        Unet = np.array(Image.open(os.path.join(Unet_path, id)))[:, :, 0]
        Deeplab = np.array(Image.open(os.path.join(Deeplab_path, id)))[:, :, 0]
        ICnet = np.array(Image.open(os.path.join(ICnet_path, id)))[:, :, 0]

        arrays = [Deeplab, Unet, ICnet]
        stacked = np.stack(arrays, axis=2)
        stacked = stacked * 50

        img = Image.fromarray(stacked, 'RGB')
        img.save(os.path.join(stacked_path, id))
        print(id)

    if __name__ == '__main__':
        pool = Pool(os.cpu_count() - 2)
        pool.map(stacking, ids)

def create_test_dir(images_path, annotations_path, images_test, annotations_test):

    import os
    import shutil
    import random
    from pathlib import Path

    files = [file for file in os.listdir(images_path) if file.endswith(".png")]

    test = open("test.txt", "w")

    test_amount = round(0.15 * len(files))

    for x in range(test_amount):
        file = random.choice(files)
        files.remove(file)
        shutil.move(os.path.join(images_path, file), images_test)
        shutil.move(os.path.join(annotations_path, file), annotations_test)
        test.write(file + "\n")

def visualize(Unet_path, Deeplab_path, ICnet_path, MaskRCNN_path, out_path):

    import numpy as np
    import os
    from PIL import Image
    from pathlib import Path

    path_dict = {'UNet': Unet_path,
                 'Deeplab': Deeplab_path,
                 'ICnet': ICnet_path,
                 'MaskRCNN': MaskRCNN_path}

    models = ['Unet', 'Deeplab', 'ICnet', 'MaskRCNN']
    ids = ['Cam_1_Pic_19-08-57.png', 'Cam_2_Pic_19-09-14.png', 'Cam_3_Pic_19-09-00.png', 'Cam_4_Pic_19-10-02.png',
           'Cam_5_Pic_19-09-55.png']

    for model, path in path_dict.items():
        for id in ids:
            img = Image.open(os.path.join(path, id))
            img = img.convert("RGBA")
            img = np.array(img)
            h, w, d = img.shape

            img = img.reshape((h * w), d)

            img[(img[:, 0] == 0)] = [0, 0, 0, 0]
            img[(img[:, 0] == 2)] = [31, 119, 180, 255]
            img[(img[:, 0] == 3)] = [44, 160, 44, 255]
            img[(img[:, 0] == 4)] = [255, 127, 14, 255]

            img = img.reshape(h, w, d)

            img = Image.fromarray(img, 'RGBA')

            id = "{}_{}".format(model, id)
            print(id)
            img.save(os.path.join(out_path, id))

