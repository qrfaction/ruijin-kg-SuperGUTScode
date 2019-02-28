# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     config
   Description :
   Author :       haxu
   date：          2018/10/20
-------------------------------------------------
   Change Activity:
                   2018/10/20:
-------------------------------------------------
"""
__author__ = 'haxu'


def gpu_config(gpu_num):
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    KTF.set_session(sess)
    print('GPU config done!')

