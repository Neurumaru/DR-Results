import os
import numpy as np
from matplotlib import pyplot as plt


def load_true(filename):
    y_true = set()

    if os.path.isfile(filename) is False:
        return None
    
    with open(filename, 'r') as f:
        for line in f:
            drug, disease = line.strip().split('\t')
            y_true.add((drug, disease))
    
    return y_true


def load_score(filename):
    y_score = []
    
    if os.path.isfile(filename) is False:
        return None

    with open(filename, 'r') as f:
        for line in f:
            drug, disease, score = line.strip().split('\t')
            y_score.append((drug, disease, float(score)))

    return y_score


def AUC_PR(y_true, y_score, sorted=True):
    AUPR = 0
    TP, FP = 0, 0
    pre_recall = 0
    precisions, recalls = [], []

    if sorted is False:
        y_score.sort(key=lambda x: x[2], reverse=True)

    TP_FN = len(y_true)
    FP_TN = len(y_score) - TP_FN
    for drug, disease, _ in y_score:
        if (drug, disease) in y_true:
            TP += 1
        else:
            FP += 1
        FN = TP_FN - TP
        TPR = TP / TP_FN
        FNR = FN / TP_FN
        FPR = FP / FP_TN
        precision = TPR / (TPR + FPR)
        recall = TPR / (TPR + FNR)
        if TP > 0:
            precisions.append(precision)
            recalls.append(recall)
            AUPR += precision * (recall - pre_recall)
        pre_recall = recall

    return AUPR, precisions, recalls


def cmap_to_float(cmap):
    for label in cmap:
        cmap[label] = np.array(cmap[label], dtype=np.float32) / 255.0
    return cmap


cmap = {
    'ANMF': (230, 0, 18, 255),
    'BGMSDDA': (243, 152, 0, 255), 
    'BNNR': (255, 241, 0, 255), 
    'DR-IBRW': (34, 172, 56, 255), 
    'DRIMC': (0, 160, 233, 255), 
    'DRRS': (0, 71, 157, 255), 
    'LAGCN': (96, 25, 134, 255), 
    'MBiRW': (228, 0, 127, 255), 
    'MSBMF': (160, 160, 160, 255),
    'OMC': (83, 83, 83, 255),
    'TPNRWRH': (0, 0, 0, 255)
}
cmap = cmap_to_float(cmap)


print(f'{"data":^8s} {"CV":^7s} {"algorithm":^12s} {"AUPR":^6s}')
print(f'====================================')
for data in ['atc-code', 'chemical']:
    y_true = load_true(f'results/{data}/association.txt')
    for fold in ['Drug',  'Disease']:
        plt.figure()
        for algorithm in ['ANMF', 'BGMSDDA', 'BNNR', 'DR-IBRW', 'DRIMC', 'DRRS', 'LAGCN', 'MBiRW', 'MSBMF', 'OMC', 'TPNRWRH']:

            y_score = load_score(f'results/{data}/{algorithm}_{fold}.txt')

            if y_score == None:
                print(f'{data:^8s} {fold:^7s} {algorithm:^12s}')
                continue

            AUPR, precision, recall = AUC_PR(y_true, y_score)
            print(f'{data:^8s} {fold:^7s} {algorithm:^12s} {AUPR:.4f}')

            plt.title(f'CV-{fold} ({data})')
            plt.plot(recall, precision, c=cmap[algorithm], label=f'{algorithm})')
            # plt.plot(recall, precision, label=f'{algorithm} (AUPR={AUPR:.4f})')
        plt.ylabel('Precision*')
        plt.xlabel('Recall')
        plt.legend()
        plt.savefig(f'images/AUPR* CV-{fold} ({data}).png')
# plt.show()