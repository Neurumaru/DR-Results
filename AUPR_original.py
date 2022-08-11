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
    for drug, disease, _ in y_score:
        if (drug, disease) in y_true:
            TP += 1
        else:
            FP += 1
        precision = TP / (TP + FP)
        recall = TP / TP_FN
        if TP > 0:
            precisions.append(precision)
            recalls.append(recall)
            AUPR += precision * (recall - pre_recall)
        pre_recall = recall

    return AUPR, precisions, recalls


print(f'{"data":^8s} {"CV":^7s} {"algorithm":^12s} {"AUPR":^6s}')
print(f'====================================')
for data in ['atc-code', 'chemical']:
    y_true = load_true(f'results/{data}/association.txt')
    for fold in ['Drug',  'Disease']:
        plt.figure()
        for algorithm in ['BGMSDDA', 'BNNR', 'DR-IBRW', 'DRIMC', 'DRRS', 'MBiRW', 'OMC', 'TRNRWRH']:
            y_score = load_score(f'results/{data}/{algorithm}_{fold}.txt')

            if y_score == None:
                continue

            AUPR, precision, recall = AUC_PR(y_true, y_score)
            print(f'{data:^8s} {fold:^7s} {algorithm:^12s} {AUPR:.4f}', end='')

            plt.title(f'CV-{fold} ({data})')
            plt.plot(recall, precision, label=f'{algorithm} (AUPR={AUPR:.4f})')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.legend()
plt.show()