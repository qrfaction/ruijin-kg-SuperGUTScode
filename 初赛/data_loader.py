import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
from tqdm import tqdm
# data map
# word2idx map



def _process_sequence(data, w2i, maxlen, feature,pad_value=-1,):
    if feature == 'word':
        x = [[w2i.get(w,1) for w in s] for s in data]
    else:
        x = [[w2i[w] for w in s] for s in data]
    x = pad_sequences(x, maxlen, padding='pre', value=pad_value)
    return np.array(x)

def get_data_with_windows(files, padding=1):
    with open('../data/working/dict.pkl', 'rb') as f:
        feature_dict = pickle.load(f)

    ### train chunk_tags
    result = {}
    maxlen = 0
    for file in tqdm(files):

        file_id = file.split('/')[-1][:-4]
        dataset = {k: [] for k in feature_dict.keys()}

        samples = pd.read_csv(file,sep='\t')

        sep_idx = [-1] + samples[samples['label']=='sep'].index.tolist() + [int(1e6)]
        num_data = len(sep_idx)-1
        for i in range(num_data):
            start = sep_idx[i]+1
            end = sep_idx[i+1]
            for k in samples.columns:
                dataset[k].append(list(samples[k])[start:end])


        for k in samples.columns:
            sep_tag = feature_dict[k][1][0] if k!='word' else 'LB'
            dataset[k] = [[sep_tag]*padding] + dataset[k] + [[sep_tag]*padding]


        result[file_id] = {k: [] for k in dataset.keys()}
        result[file_id]['index'] = []

        start = padding
        end = num_data + padding
        for i in range(start, end):

            seq_len = [len(seq) for seq in dataset['word'][i-padding: i+padding+1]]
            b = sum(seq_len[:padding])
            e = sum(seq_len[:padding + 1])
            result[file_id]['index'].append((b, e))

            for k,v in dataset.items():
                temp = []
                for item in v[i-padding: i+padding+1]:
                    temp += item
                if len(temp) > maxlen:
                    maxlen = len(temp)
                result[file_id][k].append(temp)




    for file in tqdm(files):
        file_id = file.split('/')[-1][:-4]
        for k,(w2i,i2w) in feature_dict.items():
            value = -1 if k=='label' else 0
            # value = 0
            result[file_id][k] = _process_sequence(result[file_id][k],w2i,
                                                   maxlen,k,pad_value=value)
            if k=='label':
                result[file_id][k] = result[file_id][k][:,:,np.newaxis]

    with open('../data/working/dataset.pkl', 'wb') as f:
        pickle.dump(result, f)



def multi_process():

    from glob import glob
    files = glob('../data/working/train/*.csv') + glob('../data/working/test/*.csv')
    get_data_with_windows(files)



if __name__ == '__main__':

    multi_process()
