import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)
import argparse
import numpy as np
import pickle
from sklearn.cross_validation import StratifiedKFold,KFold
from glob import glob
from models import *
from utils import *
import os



def train(cfg,data_dir):


    with open(data_dir+cfg['data_pkl'], 'rb') as f:
        dataset = pickle.load(f)
    with open(data_dir+'tool.pkl', 'rb') as f:
        tool = pickle.load(f)
    cfg['num_word'] = len(tool['word'][0])
    cfg['num_pg'] = len(tool['flag'][0])
    cfg['maxlen'] = len(dataset['text'][0])


    print(cfg)
    tr_file = list(sorted(set(filename.split('.')[0] for filename in os.listdir(tr_path))))
    tr_file = np.array([idx for idx in tr_file if idx != ''])    #  tr_path 下有个奇怪的隐藏文件夹删不掉
    folds = KFold(len(tr_file),cfg['nfold'],shuffle=True,random_state=66666)

    if cfg['use_adj_feat']:
        oof_y = np.load('./output/oof_aver.npy')

    for n_fold, (tr_idx, val_idx) in enumerate(folds):
        if n_fold not in cfg['fold']:
            continue

        print(n_fold,'-----------------')
        idx_t,tr_data = split_data(dataset,set(tr_file[tr_idx]))
        idx_v,val_data = split_data(dataset,set(tr_file[val_idx]))


        model = cfg['model'](cfg)
        if n_fold == 0:
            print(model.summary())

        f1_best = np.float('-inf')
        best_i = 0

        num_r = cal_total_relations(tr_file[val_idx])
        print('num_r',num_r)

        if cfg['use_adj_feat']:
            print(f"oof score {f1_score_v2(val_data['y'],oof_y[idx_v],num_r)}")

        for e in range(1000):
            if e - best_i > 3:
                break
            print(f'epochs_{e}.......')
            if cfg['use_adj_feat']:
                model.fit(tr_data, (tr_data['y']+oof_y[idx_t])/2,
                        batch_size=cfg['bs'],
                        epochs=1,
                        verbose=2)
            else:
                model.fit(tr_data, tr_data['y'],
                          batch_size=cfg['bs'],
                          epochs=1,
                          verbose=2)

            pred = model.predict(val_data, batch_size=256, verbose=0)
            f1 = f1_score_v2(val_data['y'],pred,num_r)

            if f1_best < f1:
                f1_best = f1
                best_i = e
                print(f'f1_score{f1}, improved save model.......')
                model.save_weights(f"../weights/{cfg['name']}_fold{n_fold}.h5")
            else:
                print(f'f1_score{f1}, best f1_score{f1_best}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--fold', type=str, required=True)


    from config import *

    args = parser.parse_args()
    cfg = {}
    cfg['nfold'] = 10
    cfg['fold'] = [int(fold) for fold in args.fold]

    cfg['model'] = rnn_model
    cfg['word_dim'] = 300
    cfg['alpha'] = 0.55   # 正样本权重
    cfg['lr'] = 0.0005
    cfg['bs'] = 256
    cfg['unit1'] = 320
    cfg['unit2'] = 320
    cfg['emb'] = 0.2
    cfg['use_adj_feat'] = False
    cfg['data_pkl'] = None

    assert args.stage in [1,2],'stage error, stage in [1,2]'

    if args.stage == 1:
        cfg['data_pkl'] = 'dataset.pkl'
        cfg['use_adj_feat'] = False

        cfg['encode_name'] = 'gru'
        cfg['name'] = 'gru'
        train(cfg, data_path + 'test_data/')
        cfg['encode_name'] = 'lstm'
        cfg['name'] = 'lstm'
        train(cfg, data_path + 'test_data/')
        cfg['encode_name'] = 'grulstm'
        cfg['name'] = 'grulstm'
        train(cfg, data_path + 'test_data/')
        cfg['encode_name'] = 'lstmgru'
        cfg['name'] = 'lstmgru'
        train(cfg, data_path + 'test_data/')

    elif args.stage == 2:
        cfg['data_pkl'] = 'dataset_v2.pkl'
        cfg['use_adj_feat'] = True

        cfg['encode_name'] = 'gru'
        cfg['name'] = 'super_gru'
        train(cfg, data_path + 'test_data/')

        cfg['encode_name'] = 'lstm'
        cfg['name'] = 'super_lstm'
        train(cfg,data_path+'test_data/')

        cfg['encode_name'] = 'grulstm'
        cfg['name'] = 'super_grulstm'
        train(cfg, data_path + 'test_data/')

        cfg['encode_name'] = 'lstmgru'
        cfg['name'] = 'super_lstmgru'
        train(cfg, data_path + 'test_data/')







