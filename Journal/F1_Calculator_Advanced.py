import os
import numpy as np
import imageio
from pathlib import Path
import pandas as pd

size_threshold = 0.2
# overlapping_threshold = 0.7

scores = pd.DataFrame(columns=['Overlapping threshold','Average precision','Average recall, Average F1'])

DCNN_CC_path = Path(r"C:\Users\djava\Desktop\F1\DL CC")
GT_CC_path = Path(r"C:\Users\djava\Desktop\F1\GT CC")
annotations = [f for f in os.listdir(GT_CC_path) if f.endswith(".png")]

for overlapping_threshold in np.arange(0, 1.05, 0.05):

    Precision_Avg = 0
    Recall_Avg = 0
    F1_Avg = 0

    for name in annotations:
        GT = np.array(imageio.imread(os.path.join(GT_CC_path,name)))
        DCNN = np.array(imageio.imread(os.path.join(DCNN_CC_path, name)))
        GT_values = []
        DCNN_values = []
        GT_wrinkles = list(np.unique(GT))
        GT_wrinkles.remove(0)
        DCNN_wrinkles = list(np.unique(DCNN))
        DCNN_wrinkles.remove(0)
        if GT_wrinkles == []:
            Precision = 1
            Recall = 1
            F1  = 1
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

    print("Overlapping threshold: {} , Average Precision: {} , Average Recall: {} , Average F1: {}".format(overlapping_threshold,Precision_Avg,Recall_Avg,F1_Avg))
    scores = scores.append({'Overlapping threshold': overlapping_threshold, 'Average precision': Precision_Avg,'Average recall': Recall_Avg, 'Average F1': F1_Avg}, ignore_index=True)

scores.to_csv('scores.csv')
