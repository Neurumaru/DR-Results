import os
import json
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


def AUC_PR_a(y_true, y_score, alpha=1, sorted=True):
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
        TPR = TP / TP_FN
        FPR = FP / FP_TN
        precision = TPR / (TPR + FPR * alpha)
        recall = TPR
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

if os.path.isfile('results/AUPR-a.json'):
    with open('results/AUPR-a.json', 'r') as f:
        results = json.load(f)
else:
    results = dict()

datas = ['atc-code', 'chemical']
folds = ['Drug',  'Disease']
algorithms = ['ANMF', 'BGMSDDA', 'BNNR', 'DR-IBRW', 'DRIMC', 'DRRS', 'LAGCN', 'MBiRW', 'MSBMF', 'OMC', 'TPNRWRH']
P_N = {
    ('atc-code', 'Drug'): (98745, 6570965),
    ('atc-code', 'Disease'): (98745, 12504835),
    ('chemical', 'Drug'): (144429, 11102409),
    ('chemical', 'Disease'): (144429, 44047212),
}

print(f'{"data":^8s} {"CV":^7s} {"algorithm":^12s} {"alpha"::^5s} {"AUPR":^6s}')
print(f'==========================================')
for data in datas:
    results.setdefault(data, dict())
    y_true = load_true(f'results/{data}/association.txt')
    P = len(y_true)
    for fold in folds:
        results[data].setdefault(fold, dict())
        P, N = P_N[(data, fold)]
        plt.figure()
        values = [[] for _ in range(1, int(N/P))]
        for algorithm in algorithms:
            results[data][fold].setdefault(algorithm, dict())

            if os.path.isfile(f'results/{data}/{algorithm}_{fold}.txt') is False:
                print(f'{data:^8s} {fold:^7s} {algorithm:^12s}')
                continue

            y_score = None
            for a in range(1, int(N/P)):
                if f'{a}' not in results[data][fold][algorithm]:
                    if y_score == None:
                        y_score = load_score(f'results/{data}/{algorithm}_{fold}.txt')

                    AUPR, _, _ = AUC_PR_a(y_true, y_score, a)
                    results[data][fold][algorithm][f'{a}'] = AUPR
                else:
                    AUPR = results[data][fold][algorithm][f'{a}']
                values[a-1].append(AUPR)
                print(f'{data:^8s} {fold:^7s} {algorithm:^12s} {a:^5d} {AUPR:^6.4f}')
                with open('results/AUPR-a.json', 'w') as f:
                    json.dump(results, f)

        max_values = np.max(np.array(values), axis=-1)
        min_values = np.min(np.array(values), axis=-1)
        for algorithm in algorithms:
            if os.path.isfile(f'results/{data}/{algorithm}_{fold}.txt') is False:
                print(f'{data:^8s} {fold:^7s} {algorithm:^12s}')
                continue

            x = list(results[data][fold][algorithm].keys())
            y = (np.array(list(results[data][fold][algorithm].values())))
            plt.title(f'CV-{fold} ({data})')
            plt.plot(x, y, c=cmap[algorithm], label=f'{algorithm})')
        plt.ylabel('AUPR')
        plt.xlabel('a')
        plt.legend()
        plt.savefig(f'images/AUPR-a CV-{fold} ({data}).png')
# plt.show()