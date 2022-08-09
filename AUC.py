import os
import pickle
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


def AUC_ROC(y_true, y_score, sorted=True):
    TP, FP = 0, 0
    TPs, FPs = [], []
    TP_sum = 0

    if sorted is False:
        y_score.sort(key=lambda x: x[2], reverse=True)

    for drug, disease, _ in y_score:
        if (drug, disease) in y_true:
            TP += 1
        else:
            FP += 1
            TP_sum += TP
        TPs.append(TP)
        FPs.append(FP)
    AUC = TP_sum / (TP * FP)
    TPR = np.array(TPs) / TP
    FPR = np.array(FPs) / FP

    return AUC, TPR, FPR


print(f'{"data":^8s} {"CV":^7s} {"algorithm":^12s} {"AUC":^6s}')
print(f'===================================')
for data in ['atc-code', 'chemical']:
    y_true = load_true(f'results/{data}/association.txt')
    with open(data + '/outputs/dictionary/drug_disease_association.pickle', 'rb') as f:
        y_true = pickle.load(f)
    for fold in ['Drug',  'Disease']:
        plt.figure()
        for algorithm in ['BGMSDDA', 'DR-IBRW']:
            print(f'{data:^8s} {fold:^7s} {algorithm:^12s}', end='')

            y_score = load_score(f'results/{data}/{algorithm}_{fold}.txt')

            if y_score == None:
                continue

            AUC, TPR, FPR = AUC_ROC(y_true, y_score)
            print(f' {AUC:.4f}')

            plt.title(f'CV-{fold} ({data})')
            plt.plot(FPR, TPR, label=f'{algorithm} (AUC={AUC:.4f})')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.legend()
plt.show()