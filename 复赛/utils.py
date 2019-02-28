import pandas as pd
import numpy as np
from config import *


def f1_score(y_true,y_pred):
    y_pred = y_pred.max(axis=1).round()
    y_true = y_true.max(axis=1)
    return 2*np.sum(y_true*y_pred)/(np.sum(y_true)+np.sum(y_pred))

def f1_score_v2(y_true,y_pred,bias):
    y_pred = y_pred.max(axis=1).round()
    y_true = y_true.max(axis=1)
    return 2 * np.sum(y_true * y_pred) / (bias + np.sum(y_pred))

def cal_total_relations(file_ids):

    all_num = 0
    for file_id in file_ids:
        label = pd.read_csv(f'{tr_path}{file_id}.ann', header=None, sep='\t')
        all_num += len(label[label[0].str.startswith('R')])
    return all_num


def split_data(dataset,file_id):
    idx = []
    for i,f_id in enumerate(dataset['file_id']):
        if f_id in file_id:
            idx.append(i)
    return idx,{k:dataset[k][idx] for k in dataset.keys()}

def concat_data(data1,data2):
    return {k:np.concatenate([data1[k],data2[k]],axis=0) for k in data1.keys()}


def filter_samples(data,distance_thres,pred,score_thres):
    condi1 = data['min_len'] < distance_thres
    condi2 = ((pred > score_thres) + (pred < (1-score_thres))).astype(bool)
    condi = condi2 * condi1
    return {k:v[condi] for k,v in data.items()}








