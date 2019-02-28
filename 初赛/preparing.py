import os
import shutil
import pandas as pd
from collections import Counter
import pickle
from utils import split_text
from tqdm import tqdm
import jieba.posseg


def process_text(idx, target_dir='train', split_method=None):

    assert target_dir in ['train','test'],"数据只分训练集测试集"

    data = {}

    # -------------------  get word ---------------------------------------------
    if not split_method:
        with open(f'../data/{target_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()  # 按行分割
    else:
        with open(f'../data/{target_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
            texts = split_method(texts)  # 自定义分割方法
    data['word'] = texts

    # -------------------  get label ----------------------------------------------
    tag_list = ['PAD' for s in texts for x in s]
    if target_dir=='train':
        tag = pd.read_csv(f'../data/{target_dir}/{idx}.ann', header=None, sep='\t')
        for i in range(tag.shape[0]):
            tag_item = tag.iloc[i][1].split(' ')
            cls, start, end = tag_item[0], int(tag_item[1]), int(tag_item[-1])

            tag_list[start] = 'B-' + cls
            for j in range(start + 1, end):
                tag_list[j] = 'I-' + cls
    assert len([x for s in texts for x in s]) == len(tag_list)
    tags = []
    start = 0
    end = 0
    for item in texts:
        l = len(item)
        end += l
        tags.append(tag_list[start:end])
        start += l

    data['label'] = tags
    # ----------------------------- 词性词边界--------------------------------------------
    word_bounds = ['I' for s in texts for x in s]
    word_flags = []
    # 增加词性与词范围

    for text in texts:
        for word, flag in jieba.posseg.cut(text):
            if len(word) == 1:
                start = len(word_flags)
                word_bounds[start] = 'S'
                word_flags.append(flag)
            else:
                start = len(word_flags)
                word_bounds[start] = 'B'
                word_flags += [flag] * len(word)
                end = len(word_flags) - 1
                word_bounds[end] = 'E'
    data['bound'] = []
    data['pos_tag'] = []
    start = 0
    end = 0
    for item in texts:
        l = len(item)
        end += l
        data['pos_tag'].append(word_flags[start:end])
        data['bound'].append(word_bounds[start:end])
        start += l
    assert len(word_bounds) == len([x for s in texts for x in s])
    assert len(word_flags) == len(word_bounds)
    # --------------------------------  部首  --------------------------------
    from cnradical import Radical, RunOption
    radical = Radical(RunOption.Radical)
    pinyin = Radical(RunOption.Pinyin)
    radical_out = [[radical.trans_ch(ele) if radical.trans_ch(ele) is not None else 'PAD' for ele in text] for text in texts]
    data['pinyin'] = [[pinyin.trans_ch(ele) if pinyin.trans_ch(ele) is not None else 'PAD' for ele in text] for text in texts]
    data['radical'] = radical_out

    # --------------------------------------------------------------------------
    num_samples = len(texts)
    num_col = len(data.keys())
    train_file = f'../data/working/{target_dir}/{idx}.csv'

    dataset = []
    for i in range(num_samples):
        records = list(zip(*[list(v[i]) for v in data.values()]))
        dataset += records+[['sep']*num_col]
    dataset = dataset[:-1]
    dataset = pd.DataFrame(dataset,columns=data.keys())


    def clean_word(w):
        if w=='\n':
            return 'LB'
        elif w in [' ','\t','\u2003']:
            return 'SPACE'
        elif w.isdigit():
            return 'num'
        else:
            return w
    dataset['word'] = dataset['word'].apply(clean_word)
    dataset.to_csv(train_file,sep='\t',index=False,encoding='utf8')


def process_raw(split_method=None):
    if os.path.exists('../data/working/'):
        shutil.rmtree('../data/working/')
    if not os.path.exists('../data/working/'):
        os.makedirs('../data/working/train/')
        os.makedirs('../data/working/test/')
    import multiprocessing as mp

    num_worker = mp.cpu_count()
    pool = mp.Pool(num_worker)
    results = []

    ids = set([x.split('.')[0] for x in os.listdir('../data/train/')])
    for idx in ids:
        result = pool.apply_async(process_text, args=(idx,'train',split_method))
        results.append(result)
    ids = set([x.split('.')[0] for x in os.listdir('../data/test/')])
    for idx in ids:
        result = pool.apply_async(process_text, args=(idx, 'test', split_method))
        results.append(result)

    pool.close()
    pool.join()
    [r.get() for r in tqdm(results)]


def get_feature_dict():

    feature_dict = {}

    # -----------------------------  word --------------------------------------
    from glob import glob

    all_w = []
    for file in glob('../data/working/train/*.csv') + glob('../data/working/test/*.csv'):
        all_w += pd.read_csv(file, sep='\t')['word'].tolist()
    word_counts = Counter(w for w in all_w)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    w2i = dict((w, i + 2) for i, w in enumerate(vocab))
    w2i['PAD'] = 0
    w2i['UNK'] = 1
    i2w = [w for i,w in enumerate(w2i.keys())]

    feature_dict['word'] = [w2i,i2w]

    # ---------------------------------  label ---------------------------------------
    i2tag = ['PAD', 'B-Disease', 'I-Disease', 'B-Reason', 'I-Reason', "B-Symptom", "I-Symptom", "B-Test", "I-Test",
                  "B-Test_Value", "I-Test_Value", "B-Drug", "I-Drug", "B-Frequency", "I-Frequency", "B-Amount",
                  "I-Amount", "B-Treatment", "I-Treatment", "B-Operation", "I-Operation", "B-Method", "I-Method",
                  "B-SideEff", "I-SideEff", "B-Anatomy", "I-Anatomy", "B-Level", "I-Level", "B-Duration", "I-Duration"]
    tag2i = {t:i for i,t in enumerate(i2tag)}

    feature_dict['label'] = [tag2i,i2tag]
    # --------------------------------------------------------------------------------
    i2bound = ['PAD','S','I','E','B']
    bound2i = {t:i for i,t in enumerate(i2bound)}
    feature_dict['bound'] = [bound2i,i2bound]

    # --------------------------------------------------------------------------------
    i2col = []
    for col in ['pos_tag','pinyin','radical']:
        for file in glob('../data/working/train/*.csv') + glob('../data/working/test/*.csv'):
            i2col += pd.read_csv(file,sep='\t')[col].tolist()
        i2col = set(i2col)
        if 'PAD' in i2col:
            i2col.remove('PAD')
        i2col = ['PAD'] + list(i2col)
        col2i = {t: i for i, t in enumerate(i2col)}
        feature_dict[col] = [col2i,i2col]

    # ---------------------------------------------------------------------------------
    with open('../data/working/dict.pkl', 'wb') as outp:
        pickle.dump(feature_dict, outp)


if __name__ == '__main__':
    process_raw(split_method=split_text)
    get_feature_dict()
