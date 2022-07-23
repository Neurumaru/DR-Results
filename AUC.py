import numpy as np
import pickle
import time
import random
import os
import scipy.io as sio
from matplotlib import pyplot as plt

from progress import progress, progressEnd


def folds(filename, association, data_S2I, i):
    with open(filename, 'r') as f:
        lines = f.readlines()

    folds_list = [list() for _ in range(len(lines))]
    folds_dict = dict()
    data_list = []
    for line_idx, line in enumerate(lines):
        splitted = line.strip().split()
        data = list(map(data_S2I.get, splitted))
        data_list.append(data)
        fold_dict = dict(zip(data, np.full(len(splitted), line_idx)))
        folds_dict.update(fold_dict)

    for connection in association:
        folds_list[folds_dict[connection[i]]].append(connection)

    return folds_list, data_list


def connection_S2I(connection, data1_S2I, data2_S2I):
    new_connection = dict()
    for c in connection:
        new_connection[(data1_S2I.get(c[0]), data2_S2I.get(c[1]))] = connection[c]
    return new_connection


drug_similarity = np.load('outputs/drug_similarity.npy')
disease_similarity = np.load('outputs/disease_similarity.npy')
drug_disease_association_matrix = np.load('outputs/drug_disease_association.npy')
with open('outputs/drug_disease_association.pickle', 'rb') as f:
    drug_disease_association_connection = pickle.load(f)
with open('outputs/drug_S2I.pickle', 'rb') as f:
    drug_S2I = pickle.load(f)
with open('outputs/disease_S2I.pickle', 'rb') as f:
    disease_S2I = pickle.load(f)
drug_disease_association_connection = connection_S2I(drug_disease_association_connection, drug_S2I, disease_S2I)
disease_10_fold, disease_10_fold_data = folds("inputs/disease_10-fold.txt", drug_disease_association_connection, disease_S2I, 1)
drug_10_fold, drug_10_fold_data = folds("inputs/drug_10-fold.txt", drug_disease_association_connection, drug_S2I, 0)

print(f'drug_similarity: {drug_similarity.shape}')
print(f'drug_S2I: {len(drug_S2I)}')
print(f'disease_similarity: {disease_similarity.shape}')
print(f'disease_S2I: {len(disease_S2I)}')
print(f'drug_disease_association_matrix : {drug_disease_association_matrix.shape}')
print(f'drug_disease_association_connection: {len(drug_disease_association_connection)}')
print()
disease_drug_association = drug_disease_association_matrix.T

foldername = 'results'

# filename = 'results.mat'
# matrixname = 'results'
# transpose = False
filename = 'final_results.mat'
matrixname = 'final_results'
transpose = True
# filename = 'DDR.mat'
# matrixname = 'drdi'
# transpose = True

for fold in range(10):
    if os.path.isdir(f'{foldername}/Disease{fold}') is False:
        continue
    if os.path.isfile(f'{foldername}/Disease{fold}/{filename}') is False:
        continue
    predict = sio.loadmat(f'{foldername}/Disease{fold}/{filename}')[matrixname]
    if transpose:
        predict = predict.T

    score_list = []
    fold_data = sorted(disease_10_fold_data[fold])
    for j in range(len(fold_data)):
        for i in range(len(drug_S2I)):
            score_list.append((i, fold_data[j], predict[fold_data[j], i]))

    score_list.sort(key=lambda x: (x[2]), reverse=True)
    TP, FP = 0, 0
    TP_sum = 0
    indics = []
    scores = []
    labels = []
    TPR = []
    for idx, (drug, disease, score) in enumerate(score_list):
        indics.append(idx)
        scores.append(score)
        labels.append(drug_disease_association_matrix[drug, disease])
        if drug_disease_association_matrix[drug, disease] == 1:
            TP += 1
        else:
            FP += 1
            TP_sum += TP
            TPR.append(TP)
    AUC = TP_sum / (TP * FP)

    if np.max(scores) == 0:
        print(f'Disease{fold}: No Score')
        continue
    scores = np.array(scores) / np.max(scores)

    plt.figure()
    plt.plot(indics, labels)
    plt.plot(indics, scores)
    plt.title(f'Disease{fold}')
    plt.figure()

    plt.plot(range(FP), TPR)
    plt.xscale('linear')
    plt.title(f'Disease{fold}')

    print(f'Disease{fold}: {AUC}')


for fold in range(10):
    if os.path.isdir(f'{foldername}/Drug{fold}') is False:
        continue
    if os.path.isfile(f'{foldername}/Drug{fold}/{filename}') is False:
        continue
    predict = sio.loadmat(f'{foldername}/Drug{fold}/{filename}')[matrixname]
    if transpose:
        predict = predict.T

    score_list = []
    fold_data = sorted(drug_10_fold_data[fold])
    for i in range(len(fold_data)):
        for j in range(len(disease_S2I)):
            score_list.append((fold_data[i], j, predict[j, fold_data[i]]))

    score_list.sort(key=lambda x: x[2], reverse=True)
    TP, FP = 0, 0
    TP_sum = 0
    indics = []
    scores = []
    labels = []
    TPR = []
    for idx, (drug, disease, score) in enumerate(score_list):
        indics.append(idx)
        scores.append(score)
        labels.append(drug_disease_association_matrix[drug, disease])
        if drug_disease_association_matrix[drug, disease] == 1:
            TP += 1
        else:
            FP += 1
            TP_sum += TP
            TPR.append(TP)
    AUC = TP_sum / (TP * FP)

    if np.max(scores) == 0:
        print(f'Drug{fold}: No Score')
        continue
    scores = np.array(scores) / np.max(scores)
    
    plt.figure()
    plt.plot(indics, labels)
    plt.plot(indics, scores)
    plt.title(f'Drug{fold}')
    plt.figure()

    plt.plot(range(FP), TPR)
    plt.xscale('linear')
    plt.title(f'Drug{fold}')
    
    print(f'Drug{fold}: {AUC}')

# plt.show()