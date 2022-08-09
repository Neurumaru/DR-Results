import os
import pickle
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt


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

for data in ['atc-code', 'chemical']:
    drug_similarity = np.load(data + '/outputs/numpy/drug_similarity.npy')
    disease_similarity = np.load(data + '/outputs/numpy/disease_similarity.npy')
    drug_disease_association_matrix = np.load(data + '/outputs/numpy/drug_disease_association.npy')
    with open(data + '/outputs/dictionary/drug_disease_association.pickle', 'rb') as f:
        drug_disease_association_connection = pickle.load(f)
    with open(data + '/outputs/dictionary/drug_S2I.pickle', 'rb') as f:
        drug_S2I = pickle.load(f)
    with open(data + '/outputs/dictionary/disease_S2I.pickle', 'rb') as f:
        disease_S2I = pickle.load(f)
    with open(data + '/outputs/dictionary/drug_I2S.pickle', 'rb') as f:
        drug_I2S = pickle.load(f)
    with open(data + '/outputs/dictionary/disease_I2S.pickle', 'rb') as f:
        disease_I2S = pickle.load(f)
    drug_disease_association_connection = connection_S2I(drug_disease_association_connection, drug_S2I, disease_S2I)
    disease_10_fold, disease_10_fold_data = folds(data + '/inputs/disease_10-fold.txt', drug_disease_association_connection, disease_S2I, 1)
    drug_10_fold, drug_10_fold_data = folds(data + '/inputs/drug_10-fold.txt', drug_disease_association_connection, drug_S2I, 0)
    disease_drug_association = drug_disease_association_matrix.T

    foldername = data + '/results'

    results = [
        ('BGMSDDA', 'results.mat', 'results', False),
        ('DR-IBRW', 'final_results.mat', 'final_results', True),
    ]
    
    print(f'Disease 10-fold ({data})')
    for algorithm, filename, matrixname, transpose in results:
        print(algorithm)
        AUCs = []
        scores_list = []
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
            scores_list.extend(score_list)
            
        scores_list.sort(key=lambda x: x[2], reverse=True)
        with open(f'results/{data}/{algorithm}_disease.txt', 'w') as f:
            for drug, disease, score in scores_list:
                f.write(f'{drug_I2S[drug]}\t{disease_I2S[disease]}\t{score}\n')

    print(f'Drug 10-fold ({data})')
    for algorithm, filename, matrixname, transpose in results:
        print(algorithm)
        AUCs = []
        scores_list= []
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
            scores_list.extend(score_list)
        scores_list.sort(key=lambda x: x[2], reverse=True)
        with open(f'results/{data}/{algorithm}_drug.txt', 'w') as f:
            for drug, disease, score in scores_list:
                f.write(f'{drug_I2S[drug]}\t{disease_I2S[disease]}\t{score}\n')