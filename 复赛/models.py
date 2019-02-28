from keras.models import Sequential, Model
from keras.layers import *
from multipliers import *
from keras import backend as K


def f1_metric(y_true,y_pred):
    y_true = K.max(y_true, axis=1)
    y_pred = K.max(y_pred, axis=1)
    y_pred = K.round(y_pred)
    return 2*K.sum(y_pred*y_true)/(K.sum(y_true)+K.sum(y_pred)+1e-12)


def relation_attention(inputs,text):

    def share_layer(feats,f):
        return [f(x) for x in feats]

    e1_mask = Lambda(lambda x: K.expand_dims(K.cast(K.greater(x, 0), 'float32'), axis=1))(inputs['seg'])
    e2_mask = Lambda(lambda x: K.expand_dims(K.cast(K.less(x, 0), 'float32'), axis=1))(inputs['seg'])
    e1 = Lambda(lambda x: K.batch_dot(x[0], x[1]) / K.sum(x[0], keepdims=True))([e1_mask, text])
    e2 = Lambda(lambda x: K.batch_dot(x[0], x[1]) / K.sum(x[0], keepdims=True))([e2_mask, text])


    encode_feat = [e2,e1,text]
    # encode_feat = share_layer(encode_feat,Conv1D(256,1,activation='relu'))
    encode_feat = share_layer(encode_feat,Conv1D(256,1,activation='tanh'))

    atte = Lambda(lambda x:(x[2]-(x[1]+x[0])))(encode_feat)
    atte = Dense(1)(atte)
    atte = Flatten()(atte)

    atte = Activation('softmax')(atte)
    atte = Reshape((1,-1))(atte)
    text = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atte,text])
    text = Flatten()(text)


    e1 = Flatten()(e1)
    e2 = Flatten()(e2)
    return text,e1,e2


def Input_moudle(cfg):


    x_in = Input((cfg['maxlen'],), name='text')
    pt_in = Input((cfg['maxlen'],), name='flag')
    pe_in = Input((cfg['maxlen'],2),name='position')
    seg_in = Input((cfg['maxlen'],), name='segment')
    len_in = Input((1,), name='min_len')
    prior_in = Input((10,), name='mask')
    pos_tag = Embedding(cfg['num_pg'], 16, name='embpos')(pt_in)
    emb = Embedding(cfg['num_word'], cfg['word_dim'], trainable=True, name='emb')



    seg = Lambda(lambda x:K.expand_dims(x))(seg_in)
    x = emb(x_in)
    x = concatenate([x, pe_in, pos_tag], axis=-1)
    x = add([x,seg])
    x = BatchNormalization(name='bn1')(x)
    x = SpatialDropout1D(0.2)(x)

    inputs = {
        'text': x_in,
        'pos_tag': pt_in,
        'position': pe_in,
        'mask': prior_in,
        'seg': seg_in,
        'len': len_in,
    }

    if cfg['use_adj_feat']:
        e1e2_in = Input((4,), name='e1e2')
        e2e1_in = Input((4,), name='e2e1')
        e1dist_in = Input((4,), name='e1dist')
        e2dist_in = Input((4,), name='e2dist')
        inputs.update({
            'e2e1': e2e1_in,
            'e1e2': e1e2_in,
            'e1dist': e1dist_in,
            'e2dist': e2dist_in,
        })

    return x,inputs


def encoder(x,encode_name,inputs,cfg):
    if encode_name == 'gru':
        x = Bidirectional(CuDNNGRU(cfg['unit1'], return_sequences=True, name='gru1'), merge_mode='sum')(x)
        x = Bidirectional(CuDNNGRU(cfg['unit2'], return_sequences=True, name='gru2'), merge_mode='sum')(x)
    elif encode_name == 'lstmgru':
        x = Bidirectional(CuDNNLSTM(cfg['unit1'], return_sequences=True, name='gru1'), merge_mode='sum')(x)
        x = Bidirectional(CuDNNGRU(cfg['unit2'], return_sequences=True, name='gru2'), merge_mode='sum')(x)
    elif encode_name == 'lstm':
        x = Bidirectional(CuDNNLSTM(cfg['unit1'], return_sequences=True, name='gru1'), merge_mode='sum')(x)
        x = Bidirectional(CuDNNLSTM(cfg['unit2'], return_sequences=True, name='gru2'), merge_mode='sum')(x)
    elif encode_name == 'grulstm':
        x = Bidirectional(CuDNNGRU(cfg['unit2'], return_sequences=True, name='gru2'), merge_mode='sum')(x)
        x = Bidirectional(CuDNNLSTM(cfg['unit2'], return_sequences=True, name='gru2'), merge_mode='sum')(x)
    else:
        raise RuntimeError("no this encode")
    return x


def rnn_model(cfg):

    def bce_loss(y_true, y_pred):
        y_true = K.max(y_true, axis=1)
        y_pred = K.max(y_pred, axis=1)
        return -K.mean(cfg['alpha']*y_true*K.log(y_pred)+(1-cfg['alpha'])*(1-y_true)*K.log(1-y_pred))


    x,inputs = Input_moudle(cfg)
    x = encoder(x,cfg['encode_name'],inputs,cfg)

    weighted_pool,e1,e2 = relation_attention(inputs,x)
    weighted_pool = Dropout(0.2)(weighted_pool)

    len_feat = BatchNormalization()(inputs['len'])
    feat = [weighted_pool,len_feat]
    if cfg['use_adj_feat']:
        prob_feat = concatenate([inputs['e2e1'], inputs['e1e2']])
        feat.append(prob_feat)
    merge = concatenate(feat,axis=1)

    output = Dense(256, activation='relu',name='d1')(merge)
    output = Dense(256, activation='relu',name='d2')(output)
    output = Dense(10,activation='sigmoid',name='d3')(output)
    output = multiply([inputs['mask'],output])


    lr_dict = {
        'emb':cfg['emb'],
        'embpos':cfg['emb'],
    }
    model = Model(list(inputs.values()), output)
    model.compile(loss=bce_loss, optimizer=M_Nadam(cfg['lr'],multipliers=lr_dict),metrics=[f1_metric])

    return model
