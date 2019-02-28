import os
import numpy as np
import pandas as pd
import pickle
from config import *
from utils import split_data
from tqdm import tqdm

def get_context_feature(dataset,oof,output_name):


    df = pd.DataFrame({
        'file_id': dataset['file_id'],
        'e1': dataset['id1'],
        'e2': dataset['id2'],
        'len': dataset['min_len'],
        'pred': oof.max(axis=1),
        'e1b': dataset['e1b'],
        'e2b': dataset['e2b'],
    })

    df['e2_dis_e1'] = df['len'] * ((df['e1b'] >= 0).astype(int) - 0.5) * 2
    df['e1_dis_e2'] = - df['e2_dis_e1']

    def work1(df, window=2):
        res = {}
        dis = {}
        temp = df.sort_values(by=['file_id', 'e1', 'e2_dis_e1'])

        for i, sub_df in tqdm(temp.groupby(by=['file_id', 'e1'])):
            score_list = [0] * window + sub_df['pred'].tolist() + [0] * window
            dist_list = [-200] * window + sub_df['e2_dis_e1'].tolist() + [200] * window
            e1 = sub_df['e1'].values[0]
            file_id = sub_df['file_id'].values[0]
            for i, e2 in enumerate(sub_df['e2']):
                res[file_id + e1 + e2] = score_list[i:i + window] + score_list[i + window + 1:window * 2 + i + 1]
                # res[file_id + e1 + e2] = score_list[i:window * 2 + i + 1]
                dis[file_id + e1 + e2] = dist_list[i:i + window] + dist_list[i + window + 1:window * 2 + i + 1]
        return res, dis

    def work2(df, window=2):
        res = {}
        dis = {}
        temp = df.sort_values(by=['file_id', 'e2', 'e1_dis_e2'])

        for i, sub_df in tqdm(temp.groupby(by=['file_id', 'e2'])):
            score_list = [0] * window + sub_df['pred'].tolist() + [0] * window
            dist_list = [-200] * window + sub_df['e1_dis_e2'].tolist() + [200] * window
            e2 = sub_df['e2'].values[0]
            file_id = sub_df['file_id'].values[0]
            for i,e1 in enumerate(sub_df['e1']):
                res[file_id + e2 + e1] = score_list[i:i + window] + score_list[i + window + 1:window * 2 + i + 1]
                # res[file_id + e2 + e1] = score_list[i:window * 2 + i + 1]
                dis[file_id + e2 + e1] = dist_list[i:i + window] + dist_list[i + window + 1:window * 2 + i + 1]
        return res, dis

    a, b = work1(df)
    c, d = work2(df)

    a.update(c)
    b.update(d)

    dataset['e2e1'] = np.array([a[k] for k in df['file_id'] + df['e2'] + df['e1']])
    dataset['e1e2'] = np.array([a[k] for k in df['file_id'] + df['e1'] + df['e2']])
    dataset['e2dist'] = np.array([b[k] for k in df['file_id'] + df['e2'] + df['e1']])
    dataset['e1dist'] = np.array([b[k] for k in df['file_id'] + df['e1'] + df['e2']])

    with open(f'../DataSets/test_data/{output_name}.pkl', 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':

    with open(data_path + 'test_data/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    get_context_feature(dataset, np.load('./output/oof_aver.npy'), 'dataset_v2')

    with open(data_path + 'test_data/test_B.pkl', 'rb') as f:
        dataset = pickle.load(f)
    get_context_feature(dataset, np.load('./output/test_B.npy'), 'test_B_v2')




