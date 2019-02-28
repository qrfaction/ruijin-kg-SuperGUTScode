import argparse
import numpy as np
from data_loader import get_data_with_windows
from config import gpu_config
from utils import *
import pickle
from sklearn.cross_validation import StratifiedKFold,KFold
from glob import glob
from models import *

def get_data(dataset,files):
    data = {}
    cols = dataset['0'].keys()
    for col in cols:
        data[col] = np.concatenate([dataset[file][col] for file in files],axis=0)
    return data

def train_windows(cfg):


    with open('../data/working/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    with open('../data/working/dict.pkl', 'rb') as f:
        feature_dict = pickle.load(f)

    tr_files = np.array(list(sorted([file.split('/')[-1][:-4] for file in glob('../data/working/train/*.csv')])))


    cfg['vocab'] = len(feature_dict['word'][0])
    cfg['num_tags'] = len(feature_dict['label'][0])
    cfg['num_pg'] = len(feature_dict['pos_tag'][0])
    cfg['num_bound'] = len(feature_dict['bound'][0])
    cfg['num_pinyin'] = len(feature_dict['pinyin'][0])
    cfg['num_radical'] = len(feature_dict['radical'][0])

    print(cfg)
    folds = KFold(len(tr_files),cfg['num_fold'],shuffle=True,random_state=666)
    for n_fold, (tr_idx, val_idx) in enumerate(folds):
        if n_fold not in cfg['fold']:
            continue
        print(n_fold,'-----------------')
        tr_data = get_data(dataset,tr_files[tr_idx])
        val_data = get_data(dataset,tr_files[val_idx])

        model = cfg['model'](cfg)
        if n_fold == 0:
            print(model.summary())
        f1_best = np.float('-inf')
        best_i = 0
        for e in range(1000):
            if e - best_i > 3:
                break

            print(f'epochs_{e}.......')
            model.fit(tr_data, tr_data['label'],
                      batch_size=cfg['bs'],
                      epochs=1,
                      verbose=2,
                    )

            pred = model.predict(val_data, batch_size=128, verbose=0)
            pred = np.argmax(pred, axis=2)


            f1 = F1_score_v2(val_data['label'][:,:,0], pred, val_data['index'])

            # print(F1_score_v2(val_data['label'][:,:,0], pred, val_data['index']))
            if f1_best < f1:
                f1_best = f1
                best_i = e
                print(f'f1_score{f1}, improved save model.......')
                model.save_weights(f"../weights/{cfg['weight']}_fold{n_fold}.h5")
            else:
                print(f'f1_score{f1}, best f1_score{f1_best}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--weight', type=str, required=True)
    # parser.add_argument('--word_dim', type=int, default=300)

    args = parser.parse_args()
    cfg = {}
    cfg['fold'] = [int(fold) for fold in args.fold]
    cfg['word_dim'] = args.word_dim
    cfg['bs'] = args.bs
    cfg['gpu'] = args.gpu
    cfg['lr'] = 0.001
    cfg['weight'] = args.weight
    cfg['num_fold'] = 5
    cfg['maxlen'] = 513
    cfg['model'] = crf_model
    gpu_config(args.gpu)

    cfg['lr_layer2'] = 1
    cfg['lr_layer1'] = 1

    cfg['unit2'] = 128
    cfg['unit1'] = 320

    cfg['sdp'] = 0.15
    cfg['o_w'] = 0.55
    train_windows(cfg)
















