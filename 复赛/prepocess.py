import pandas as pd
import numpy as np
from config import *
import os
import shutil
import jieba.posseg
from tqdm import tqdm
import pickle
from collections import Counter
import multiprocessing as mp

def get_samples_from_text(target_dir, idx, maxlen):
    def clean(text):
        text = text.replace('\n','&')
        return text

    relation = set([
        'Test_Disease',
        'Symptom_Disease',
        'Treatment_Disease',
        'Drug_Disease',
        'Anatomy_Disease',
        'Frequency_Drug',
        'Duration_Drug',
        'Amount_Drug',
        'Method_Drug',
        'SideEff_Drug'
    ])

    with open(target_dir+f'{idx}.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    label = pd.read_csv(target_dir + f'{idx}.ann', header=None, sep='\t',keep_default_na=False)

    label_T = label[label[0].str.startswith('T')]
    label_T.columns = ['id','entity','text']
    label_T['category'] = [e.split()[0] for e in label_T['entity'].tolist()]
    label_T['start'] = [int(e.split()[1]) for e in label_T['entity'].tolist()]
    label_T['end'] = [int(e.split()[-1]) for e in label_T['entity'].tolist()]

    datasets = []
    for r in tqdm(relation):
        c1,c2 = r.split('_')
        e1_data = label_T[label_T['category']==c1]
        e2_data = label_T[label_T['category']==c2]

        for i,e1 in e1_data.iterrows():
            for j,e2 in e2_data.iterrows():


                begin = min(e1['start'], e2['start'])
                end = max(e1['end'], e2['end'])
                if end - begin > 150:
                    continue

                if end - begin > maxlen:
                    continue
                min_len = max(e1['start'], e2['start']) - min(e1['end'], e2['end'])


                num_pad = maxlen - (end - begin)
                if end - begin <= 100:
                    window = 50
                elif end - begin <= 140:
                    window = 30
                else:
                    window = 0

                left_w = min(num_pad // 2,window)
                right_w = min(num_pad - left_w,window)
                begin = max(0,begin - left_w)
                end = min(len(text),end+right_w)

                sentence = clean(text[begin:end])

                # 前五个 东西位置不能动 --------------------
                if e1['start'] < e2['start']:
                    e1b = e1['start'] - begin
                    e2b = -end + e2['start']
                else:
                    e2b = e2['start'] - begin
                    e1b = - end + e1['start']
                e2e = e2b + len(e2['text'])
                e1e = e1b + len(e1['text'])

                datasets.append([
                    e1['id'],
                    e2['id'],
                    e1['category'] + '_' + e2['category'],
                    -1,
                    end - begin,
                    min_len,
                    e1b,
                    e1e,
                    e2b,
                    e2e,
                    idx,
                    sentence,
                ])

    label_R = label[label[0].str.startswith('R')][1].tolist()
    if len(label_R)>0:
        label_R = set((r.split()[1][5:],r.split()[2][5:]) for r in label_R)
        for i in range(len(datasets)):
            if tuple(datasets[i][:2]) in label_R:
                datasets[i][3] = 1
            else:
                datasets[i][3] = 0
    return datasets

def cut_word_worker(sentences):

    #  cut sentences
    alltexts = []
    allflags = []
    for sent in tqdm(sentences):
        text = []
        flags = []
        for w, flag in jieba.posseg.cut(sent):
            text.append(w)
            flags.append(flag)
        alltexts.append(text)
        allflags.append(flags)
    return alltexts,allflags

def dataset2dict(filename,datapath,maxlen):


    i2r = [
        'Test_Disease',
        'Symptom_Disease',
        'Treatment_Disease',
        'Drug_Disease',
        'Anatomy_Disease',
        'Frequency_Drug',
        'Duration_Drug',
        'Amount_Drug',
        'Method_Drug',
        'SideEff_Drug'
    ]
    r2i = {r:i for i,r in enumerate(i2r)}

    datasets = pd.read_csv(datapath+filename+'.csv',keep_default_na=False)

    data_dict = {col:np.array(datasets[col].tolist()) for col in datasets.columns}

    if True: #  get label & mask
        masks = np.zeros((len(datasets),len(i2r)))
        labels = np.zeros((len(datasets),len(i2r)))
        for i,(r,label) in enumerate(zip(data_dict['category'],data_dict['label'])):
            if label == 1:
                labels[i, r2i[r]] = 1
            masks[i, r2i[r]] = 1

        data_dict['mask'] = masks
        data_dict['y'] = labels

    if True: # cut word
        def cut_corpus(texts):
            num_worker = mp.cpu_count()
            pool = mp.Pool(num_worker)
            num_samples = len(texts)
            ave = 1 + num_samples // num_worker
            results = []

            for i in range(num_worker):
                result = pool.apply_async(cut_word_worker, args=(texts[i * ave:(i + 1) * ave],))
                results.append(result)
            pool.close()
            pool.join()

            alltexts = []
            allflags = []
            for r in results:
                subtexts, subflags = r.get()
                alltexts += subtexts
                allflags += subflags
            assert len(allflags) == len(data_dict['text'])
            assert len(alltexts) == len(data_dict['text'])
            return alltexts,allflags

        alltexts,allflags = cut_corpus(data_dict['text'])


    if True:  # word2char
        temp_texts = []
        temp_flags = []
        for words,flags,l in zip(alltexts,allflags,data_dict['len']):
            chars = []
            char_flags = []
            for w,flag in zip(words,flags):
                chars += list(w)
                char_flags += len(w)*[flag]
            assert l==len(chars),print(chars,words,l)
            assert l==len(char_flags),print(chars,words,l)
            temp_flags.append(char_flags)
            temp_texts.append(chars)

        alltexts = temp_texts
        allflags = temp_flags



    if True:   # get w2i i2w
        word_counts = Counter(w for text in alltexts for w in text)
        vocab = [w for w, f in word_counts.items() if f >= 3]
        w2i = dict((w, i + 3) for i, w in enumerate(vocab))
        w2i['PAD'] = 0
        w2i['UNK'] = 1
        w2i['SEP'] = 2
        i2w = [w for w,i in sorted(w2i.items(),key=lambda x:x[1])]

        i2flag = list(set(flag for flags in allflags for flag in flags))
        i2flag.remove('x')
        i2flag = ['x'] + i2flag
        flag2i = {flag:i for i,flag in enumerate(i2flag)}


    if True: #  pad_sequnce & get position embedding
        def cal_pos(begin,end,l):
            if begin < 0:
                begin += l
                end += l
            median = (begin+end)//2
            return np.arange(l) - median

        textseq = []
        flagseq = []
        position = np.zeros((len(alltexts),maxlen,2)) + maxlen
        segment = np.zeros((len(alltexts),maxlen))

        for text,flags,e1b,e1e,e2b,e2e in tqdm(zip(alltexts,allflags,data_dict['e1b'],data_dict['e1e'],data_dict['e2b'],data_dict['e2e'])):
            pad_len = maxlen - len(text)

            seg = np.zeros(len(text))
            seg[e1b:e1e] += 0.1
            seg[e2b:e2e] -= 0.1

            e1pos = cal_pos(e1b,e1e,len(text))
            e2pos = cal_pos(e2b,e2e,len(text))

            if pad_len >= 0:
                pad = pad_len * [0]

                position[len(textseq),pad_len:,0] = e1pos
                position[len(textseq),pad_len:,1] = e2pos

                segment[len(textseq),pad_len:] = seg

                textseq.append(pad + [w2i.get(w, 1) for w in text])
                flagseq.append(pad + [flag2i.get(flag, 0) for flag in flags])
            else:
                pad_len = -pad_len
                index1 = (len(text) - pad_len -1 )//2
                index2 = index1 + 1 + pad_len

                e1_sep = e1pos[len(e1pos)//2]
                e2_sep = e1pos[len(e2pos)//2]

                position[len(textseq),:,0] = np.concatenate([e1pos[:index1],[e1_sep],e1pos[index2:]])
                position[len(textseq),:,1] = np.concatenate([e2pos[:index1],[e2_sep],e2pos[index2:]])

                segment[len(textseq),:] = np.concatenate([seg[:index1],[0],seg[index2:]])


                textseq.append([w2i.get(w, 1) for w in text[:index1]+['SEP']+text[index2:]])
                flagseq.append([flag2i.get(flag,0) for flag in flags[:index1]+['x']+flags[index2:]])
            assert len(textseq[-1]) == maxlen
            assert len(flagseq[-1]) == maxlen



        position /= (2*maxlen)
        data_dict['flag'] = flagseq
        data_dict['text'] = textseq
        data_dict['position'] = position
        data_dict['segment'] = segment

    tools = {}
    tools['word'] = [w2i,i2w]
    tools['flag'] = [flag2i,i2flag]

    for k in data_dict.keys():
        data_dict[k] = np.array(data_dict[k])

    with open(datapath+'tool.pkl', 'wb') as outp:
        pickle.dump(tools, outp)
    with open(datapath+filename+'.pkl', 'wb') as outp:
        pickle.dump(data_dict, outp)

def prepocess(output_dir,maxlen):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    import multiprocessing as mp

    def get_dataset(filename,file_path):

        num_worker = mp.cpu_count()
        pool = mp.Pool(num_worker)
        results = []

        ids = list(sorted(set([x.split('.')[0] for x in os.listdir(file_path)])))
        print(list(os.listdir(file_path)))
        for idx in ids:
            if idx == '':
                continue
            result = pool.apply_async(get_samples_from_text, args=(file_path,idx, maxlen))
            results.append(result)
        pool.close()
        pool.join()

        dataset = []
        for r in results:
            dataset += r.get()

        dataset = pd.DataFrame(dataset,columns=['id1','id2','category','label','len','min_len','e1b','e1e','e2b','e2e','file_id','text'])
        dataset.to_csv(output_dir+filename,index=False)

        if True: #  EDA
            print(len(dataset),'\n')
            print(Counter(dataset['category']+dataset['label'].astype(str)),'\n')
            print(Counter(dataset['label']),'\n')

    get_dataset('test.csv', te_path)
    get_dataset('train.csv',tr_path)

    te = pd.read_csv(output_dir + 'test.csv')
    tr = pd.read_csv(output_dir + 'train.csv')
    dataset = tr.append(te).reset_index(drop=True)
    dataset.to_csv(output_dir+'dataset.csv',index=False)

    dataset2dict('dataset',output_dir,maxlen)

    if not os.path.exists('../weights/'):
        os.makedirs('../weights/')
    if not os.path.exists('./output/'):
        os.makedirs('./output/')


if __name__ == '__main__':
    prepocess(data_path+'test_data/',200)

























