import os
import numpy as np
import imageio
from pathlib import Path

DCNN_CC_path = Path(r"C:\Users\djava\Desktop\F1\DL CC")
GT_CC_path = Path(r"C:\Users\djava\Desktop\F1\GT CC")
annotations = [f for f in os.listdir(GT_CC_path) if f.endswith(".png")]

Precision_Avg = 0
Recall_Avg = 0
F1_Avg = 0

for name in annotations:
    GT = np.array(imageio.imread(os.path.join(GT_CC_path,name)))
    DCNN = np.array(imageio.imread(os.path.join(DCNN_CC_path, name)))
    GT_values = []
    DCNN_values = []
    wrinkles = list(np.unique(GT))
    wrinkles.remove(0)
    if wrinkles == []:
        Precision = 1
        Recall = 1
        F1  = 1
    else:
        for val in wrinkles:
            target_area = DCNN[np.where(GT == val)]
            wrinkle_coverage = np.count_nonzero(target_area) / len(target_area)
            if wrinkle_coverage > 0.7:
                GT_values.append(val)
                DCNN_values.extend(np.unique(target_area))
        TP = len(GT_values)
        FN = len(wrinkles) - len(GT_values)
        FP = len(np.unique(DCNN)) - len(np.unique(DCNN_values))
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        if Precision == 0 and Recall == 0:
            F1 = 0
        else:
            F1 = 2 * Precision * Recall / (Precision + Recall)
    Precision_Avg += Precision
    Recall_Avg += Recall
    F1_Avg += F1

    print("{}: Precision: {} , Recall: {} , F1: {}".format(name,Precision,Recall,F1))

Precision_Avg = Precision_Avg / len(annotations)
Recall_Avg = Recall_Avg / len(annotations)
F1_Avg = F1_Avg / len(annotations)

print("Average Precision: {} , Average Recall: {} , Average F1: {}".format(Precision_Avg,Recall_Avg,F1_Avg))