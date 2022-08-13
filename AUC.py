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


def cmap_to_float(cmap):
    for label in cmap:
        cmap[label] = np.array(cmap[label], dtype=np.float32) / 255.0
    return cmap


cmap_dict = {
    'custom': {
        'ANMF': (230, 0, 18, 255),
        'BGMSDDA': (243, 152, 0, 255), 
        'BNNR': (255, 241, 0, 255), 
        'DR-IBRW': (34, 172, 56, 255), 
        'DRIMC': (0, 160, 233, 255), 
        'DRRS': (0, 71, 157, 255), 
        'LAGCN': (96, 25, 134, 255), 
        'MBiRW': (228, 0, 127, 255), 
        'MSBMF': (209, 192, 165, 255),
        'OMC': (106, 57, 6, 255),
        'TPNRWRH': (128, 128, 128, 255)
    },
    'tab10': {
        'ANMF': (31, 119, 180, 255),
        'BGMSDDA': (255, 127, 14, 255), 
        'BNNR': (44, 160, 44, 255), 
        'DR-IBRW': (214, 39, 40, 255), 
        'DRIMC': (148, 103, 189, 255), 
        'DRRS': (140, 86, 75, 255), 
        'LAGCN': (227, 119, 194, 255), 
        'MBiRW': (127, 127, 127, 255), 
        'MSBMF': (188, 189, 34, 255),
        'OMC': (23, 190, 207, 255),
        'TPNRWRH': (32, 32, 32, 255)
    }
}
cmap = cmap_to_float(cmap_dict['tab10'])

print(f'{"data":^8s} {"CV":^7s} {"algorithm":^12s} {"AUC":^6s}')
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

            AUC, TPR, FPR = AUC_ROC(y_true, y_score)
            print(f'{data:^8s} {fold:^7s} {algorithm:^12s} {AUC:.4f}')

            plt.title(f'CV-{fold} ({data})')
            plt.plot(FPR, TPR, c=cmap[algorithm], label=f'{algorithm}')
            # plt.plot(FPR, TPR, label=f'{algorithm} (AUC={AUC:.4f})')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.legend()
        plt.savefig(f'images/AUC CV-{fold} ({data}).png')
# plt.show()