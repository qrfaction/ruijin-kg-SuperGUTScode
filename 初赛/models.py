from keras.models import Sequential, Model
from keras.layers import *
from crf import CRF
from keras.optimizers import Nadam
from multipliers import *
import tensorflow as tf
from keras.initializers import *


def return_input(cfg):
    x_in = Input((cfg['maxlen'],), name='word')
    pos_tag_in = Input((cfg['maxlen'],), name='pos_tag')
    py_in = Input((cfg['maxlen'],), name='pinyin')
    radical_in = Input((cfg['maxlen'],), name='radical')
    bound_in = Input((cfg['maxlen'],), name='bound')


    x = Embedding(cfg['vocab'], cfg['word_dim'], trainable=True, name='emb')(x_in)
    x = SpatialDropout1D(0.2)(x)

    pos_tag = Embedding(cfg['num_pg'], 16, name='embpos')(pos_tag_in)
    bound = Embedding(cfg['num_bound'], 4, name='embbound')(bound_in)
    pinyin = Embedding(cfg['num_pinyin'], 16)(py_in)
    radical = Embedding(cfg['num_radical'], 16)(radical_in)
    x = concatenate([pos_tag, x, bound], axis=-1)

    return x,{
        'word':x_in,
        'pos_tag':pos_tag_in,
        'bound':bound_in,
    }


def rnn_model(cfg):


    x,inputs = return_input(cfg)
    mask = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(inputs['word'])
    x = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([x, mask])


    x = Bidirectional(CuDNNGRU(cfg['unit1'], return_sequences=True, name='layer1'), merge_mode='sum')(x)
    x = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([x, mask])
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNGRU(cfg['unit2'], return_sequences=True, name='layer2',), merge_mode='sum')(x)
    x = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([x, mask])


    x = Conv1D(cfg['num_tags'],kernel_size=1,padding='same')(x)
    def softmax(x):
        x = K.exp(x-K.max(x,axis=2,keepdims=True))
        return x/K.sum(x,axis=2,keepdims=True)
    output = Lambda(softmax)(x)


    def focalloss(y_true, y_pred):
        num_c = K.int_shape(y_pred)[-1]

        pad_mask = K.cast(K.greater(y_true, -1), 'float32')
        o_mask = K.cast(K.equal(y_true, 0), 'float32')

        y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), num_c)


        base_loss =  y_true * K.log(y_pred) * pad_mask * ((1-y_pred)**cfg['gamma'])

        o_loss = cfg['o_w'] * o_mask * base_loss
        loss_2 = (1-cfg['o_w']) * (1-o_mask)  * base_loss

        return -K.sum(o_loss+loss_2)/K.sum(pad_mask)

    model = Model(inputs=list(inputs.values()), outputs=[output])

    multipliers = {
        'emb': 0.1,
        'embbound': 0.1,
        'embpos': 0.1,
        'layer1': cfg['lr_layer1']
    }
    model.compile(optimizer=M_Nadam(cfg['lr'], multipliers=multipliers), loss=focalloss)
    return model


def crf_model(cfg):

    x, inputs = return_input(cfg)
    mask = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(inputs['word'])
    x = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([x, mask])

    x = Bidirectional(CuDNNGRU(cfg['unit1'], return_sequences=True, name='gru1'), merge_mode='sum')(x)
    x = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([x, mask])
    x = SpatialDropout1D(0.3)(x)

    x = Bidirectional(CuDNNGRU(cfg['unit2'], return_sequences=True, name='gru2'), merge_mode='sum')(x)
    x = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([x, mask])
    x = SpatialDropout1D(0.15)(x)


    crf = CRF(cfg['num_tags'], sparse_target=True,name='crf')
    output = crf(x, mask=mask)

    model = Model(inputs=list(inputs.values()), outputs=[output])

    multipliers = {
        'emb':0.1,
        'embbound':0.1,
        'embpos':0.1,
        # 'crf':cfg['lr_crf'],
        'gru1':cfg['lr_layer1']
    }
    model.compile(optimizer=M_Nadam(cfg['lr'],multipliers=multipliers), loss=crf.loss_function)
    return model









if __name__ == '__main__':
    pass












