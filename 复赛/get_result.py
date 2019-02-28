import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)
import argparse
import os
import shutil
import pickle
from models import *
from config import *
from tqdm import tqdm
from utils import *
from sklearn.cross_validation import StratifiedKFold,KFold

def make_submit(cfg,pred_path,data_pkl='dataset_v2.pkl',test_path=te_path):


    with open(cfg['data_dir'] + data_pkl, 'rb') as f:
        dataset = pickle.load(f)
    te_idx = list(sorted(set(filename.split('.')[0] for filename in os.listdir(test_path))))
    _,dataset = split_data(dataset,te_idx)


    result = np.load('./output/'+pred_path)
    result = result.max(axis=1)
    print((result > 0.5).sum(), (result <= 0.5).sum())


    # R158	Symptom_Disease Arg1:T137 Arg2:T136
    relations = {}
    r_num = 0
    for score, idx, e1id, e2id, cate,l in tqdm(
            zip(result, dataset['file_id'], dataset['id1'], dataset['id2'], dataset['category'],dataset['min_len'])):
        if score > 0.5:
            r_num += 1
            if idx not in relations:
                relations[idx] = []
            if cate == 'SideEff_Drug':
                relations[idx].append(f'R{len(relations[idx])+1}	SideEff-Drug Arg1:{e1id} Arg2:{e2id}')
            else:
                relations[idx].append(f'R{len(relations[idx])+1}	{cate} Arg1:{e1id} Arg2:{e2id}')
    print(r_num)
    if not os.path.exists('../submit/'):
        os.makedirs('../submit/')


    for idx in te_idx:
        with open(f'{test_path}{idx}.ann', encoding='utf-8') as f:
            text = f.read().strip('\n')
        text += ('\n' + '\n'.join(relations[idx]))
        with open(f'../submit/{idx}.ann', encoding='utf-8', mode='w') as f:
            f.write(text)

    shutil.make_archive('submit', 'zip', '../submit/')

def average(tr_files,te_files,cfg,data_pkl='dataset_v2.pkl'):
    if len(tr_files) > 0:
        with open(cfg['data_dir'] + data_pkl, 'rb') as f:
            dataset = pickle.load(f)

        tr_text = list(sorted(set(filename.split('.')[0] for filename in os.listdir(tr_path))))
        tr_text = np.array([idx for idx in tr_text if idx != ''])  # tr_path 下有个奇怪的隐藏文件夹删不掉
        num_r = cal_total_relations(tr_text)

        _,tr = split_data(dataset,tr_text)

        tr_pred = [np.load('./output/'+file) for file in tr_files]
        [print(pred.shape) for pred in tr_pred]
        [print(f1_score_v2(tr['y'],pred[:len(tr['text'])],num_r)) for pred in tr_pred]

        print(f1_score_v2(tr['y'],np.average(tr_pred,axis=0)[:len(tr['text'])],num_r))

        if cfg['save_oof']:
            np.save('./output/'+cfg['oof_average_name'],np.average(tr_pred,axis=0))
    if cfg['save_pred']:
        np.save('./output/'+cfg['average_name'],np.average([np.load('./output/'+file) for file in te_files],axis=0))

def get_pred_npy(cfg,data_pkl='dataset_v2.pkl',test_path=te_path):

    with open(cfg['data_dir'] + data_pkl, 'rb') as f:
        dataset = pickle.load(f)
    te_idx = list(sorted(set(filename.split('.')[0] for filename in os.listdir(test_path))))
    _,dataset = split_data(dataset,te_idx)


    models = [(fold, cfg['model'](cfg)) for fold in range(cfg['nfold'])]
    for fold, model in models:
        model.load_weights(f"../weights/{cfg['name']}_fold{fold}.h5")
    result = [model.predict(dataset, verbose=1,batch_size=cfg['bs']) for fold, model in tqdm(models)]
    result = np.average(result, axis=0)

    if cfg['save_pred']:
        np.save(f"./output/{cfg['name']}_pred", result)


def get_oofy(cfg,data_pkl='dataset_v2.pkl'):
    with open(cfg['data_dir'] + data_pkl, 'rb') as f:
        dataset = pickle.load(f)

    tr_file = list(sorted(set(filename.split('.')[0] for filename in os.listdir(tr_path))))
    tr_file = np.array([idx for idx in tr_file if idx != ''])  # tr_path 下有个奇怪的隐藏文件夹删不掉
    folds = KFold(len(tr_file), cfg['nfold'], shuffle=True, random_state=66666)

    oof_y = np.zeros_like(dataset['y'])
    model = cfg['model'](cfg)

    for fold, (tr_idx, val_idx) in enumerate(folds):
        model.load_weights(f"../weights/{cfg['name']}_fold{fold}.h5")
        idx_v, val_data = split_data(dataset, set(tr_file[val_idx]))
        oof_y[idx_v] = model.predict(val_data, batch_size=cfg['bs'], verbose=1)

    if cfg['save_oof']:
        np.save(f"./output/{cfg['name']}_y",oof_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--stage', type=int, required=True)

    args = parser.parse_args()

    cfg = {}
    cfg['nfold'] = 10
    cfg['model'] = rnn_model
    cfg['maxlen'] = 200
    cfg['word_dim'] = 300
    cfg['alpha'] = 0.55  # 正样本权重
    cfg['lr'] = 0.001
    cfg['bs'] = 2048
    cfg['unit1'] = 320
    cfg['unit2'] = 320
    cfg['data_dir'] = '../DataSets/test_data/'
    with open(cfg['data_dir'] + 'tool.pkl', 'rb') as f:
        tool = pickle.load(f)
    cfg['num_word'] = len(tool['word'][0])
    cfg['num_pg'] = len(tool['flag'][0])
    cfg['emb'] = 0.2
    cfg['encode_name'] = 'gru'
    cfg['use_adj_feat'] = True
    cfg['save_oof'] = False
    cfg['save_pred'] = False
    cfg['oof_average_name'] = None
    cfg['average_name'] = None

    assert args.stage in [1, 2], 'stage error, stage in [1,2]'

    if args.stage == 1:
        if True:  # get adj feat
            feat_model_name = [
                'gru',
                'grulstm',
                'lstm',
                'lstmgru'
            ]
            cfg['use_adj_feat'] = False
            cfg['save_oof'] = True
            cfg['save_pred'] = True
            for name in feat_model_name:
                cfg['name'] = name
                cfg['encode_name'] = name
                get_oofy(cfg,data_pkl='dataset.pkl')
                get_pred_npy(cfg,'test_B.pkl',te_B_path)

            cfg['average_name'] = 'test_B'
            cfg['oof_average_name'] = 'oof_aver'
            average(
                [feat+'_y.npy' for feat in feat_model_name],
                [feat+'_pred.npy' for feat in feat_model_name],
                cfg,
                data_pkl='dataset.pkl'
            )
    elif args.stage == 2:


        if True:  # get_result
            feat_model_name = [
                'super_gru',
                'super_grulstm',
                'super_lstm',
                'super_lstmgru'
            ]
            cfg['use_adj_feat'] = True
            cfg['save_oof'] = True
            cfg['save_pred'] = True
            for name in feat_model_name:
                cfg['name'] = name
                cfg['encode_name'] = name[6:]
                get_oofy(cfg)
                get_pred_npy(cfg,'test_B_v2.pkl',te_B_path)

            cfg['save_oof'] = False
            cfg['average_name'] = 'super_test_B'
            average(
                [feat + '_y.npy' for feat in feat_model_name],
                [feat + '_pred.npy' for feat in feat_model_name],
                cfg
            )

        make_submit(cfg,'super_test_B.npy','test_B_v2.pkl',te_B_path)





















