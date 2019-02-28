import argparse
import os
import numpy as np
import shutil
import pickle
from models import crf_model
from glob import glob
from config import gpu_config
from train import get_data
from tqdm import tqdm

with open('../data/working/dict.pkl', 'rb') as f:
    feature_dict = pickle.load(f)

def submit(text_id, result_tags):
    with open(f'../data/test/{text_id}.txt', encoding='utf-8') as f:
        text = f.read()

    with open(f'../submit/{text_id}.ann', 'w', encoding='utf-8') as tag_data:
        tags = []
        for tag in result_tags:
            if tag != 'O':
                tag_ = tag.split('-')[1]
            else:
                tag_ = tag
            tags.append(tag_)

        prev = tags[0]
        start = 0
        num = 0
        for i in range(1, len(tags)):
            cur = tags[i]
            if cur != prev:
                end = i
                if prev != 'O':
                    num += 1
                    content = text[start:end]
                    content = content.replace('\n', ' ')
                    tag_data.write(
                        'T' + str(num) + '\t' + prev + ' ' + str(start) + ' ' + str(end) + '\t' + content + '\n')
                start = i
                prev = cur


def make_submit(models):
    for filename in tqdm(os.listdir('../data/test/')):
        idx = filename.split('.')[0]
        predict_v2(models, idx)
    shutil.make_archive('submit', 'zip', '../submit/')


def predict_v2(models, test_id):

    with open(f'../data/test/{test_id}.txt', encoding='utf-8') as f:
        text = f.read()
    with open('../data/working/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)


    cfg['vocab'] = len(feature_dict['word'][0])
    cfg['num_tags'] = len(feature_dict['label'][0])
    cfg['num_pg'] = len(feature_dict['pos_tag'][0])
    cfg['num_bound'] = len(feature_dict['bound'][0])
    cfg['num_pinyin'] = len(feature_dict['pinyin'][0])
    cfg['num_radical'] = len(feature_dict['radical'][0])


    result = 0
    for model in models:
        result += model.predict(dataset[test_id], batch_size=256)
    result /= 5


    result = result.argmax(axis=2)
    tags = []
    num_w = 0
    for r, (i, j), seq in zip(result, dataset[test_id]['index'],dataset[test_id]['word']):
        idx = seq > 0
        r = r[idx][i:j]
        tags += [feature_dict['label'][1][i] for i in r]
        num_w += (j-i)
    print(num_w,len(text))
    submit(test_id, tags)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    cfg = {}
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--unit', type=int, default=320)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--weight', type=str, required=True)

    args = parser.parse_args()

    cfg['word_dim'] = args.word_dim
    cfg['bs'] = args.bs
    cfg['gpu'] = args.gpu
    cfg['unit'] = args.unit
    cfg['lr'] = args.lr
    cfg['lr_crf'] = 1
    cfg['lr_gru'] = 1
    cfg['weight'] = args.weight
    cfg['num_fold'] = 5
    cfg['maxlen'] = 513
    cfg['vocab'] = len(feature_dict['word'][0])
    cfg['num_tags'] = len(feature_dict['label'][0])
    cfg['num_pg'] = len(feature_dict['pos_tag'][0])
    cfg['num_bound'] = len(feature_dict['bound'][0])
    cfg['num_pinyin'] = len(feature_dict['pinyin'][0])
    cfg['num_radical'] = len(feature_dict['radical'][0])
    gpu_config(args.gpu)

    model0 = crf_model(cfg)
    model0.load_weights(f"../weights/{cfg['weight']}_fold{0}.h5")
    model1 = crf_model(cfg)
    model1.load_weights(f"../weights/{cfg['weight']}_fold{1}.h5")
    model2 = crf_model(cfg)
    model2.load_weights(f"../weights/{cfg['weight']}_fold{2}.h5")
    model3 = crf_model(cfg)
    model3.load_weights(f"../weights/{cfg['weight']}_fold{3}.h5")
    model4 = crf_model(cfg)
    model4.load_weights(f"../weights/{cfg['weight']}_fold{4}.h5")

    models = [model0, model1, model2, model3, model4]
    make_submit(models)
