#!/usr/bin/python
# -*- coding:utf8 -*-

from utils.reader import Reader
from utils.nn import CNN, CnnLstm, SCNN

import numpy as np

# from keras.utils.np_utils import to_categorical
# print to_categorical([1,2],10)


conf = {
    "vocab_size": 100000,
    "maxlen": 420,
    "batch_size": 20,
    "embedding_dims": 100,
    "nb_filter": 250,
    "filter_length": 4,
    "pool_length": 3,
    "hidden_size": 200,
    "nb_epoch": 10,
    "dropout": 0.2, 
    "train_file": "data/train_pdtb_imp.json",
    "vocab_file": "data/vocab",
    "test_file": "",
    "valid_file": "data/dev_pdtb_imp.json",
    "vocab_size": 100000,
}

reader = Reader(conf)
reader.get_full_train_data()
reader.get_full_valid_data(get_id=True)

features = [[],[]]
targets = []
v_features = [[], []]
v_targets = []
v_id = []
v_im_features = [[], []]
v_im_targets = []
v_im_id = []

for i in xrange(len(reader.train)):
    features[0].append(reader.train[i][0][0])
    features[1].append(reader.train[i][0][1])
    targets.append(reader.train[i][1])

for i in xrange(len(reader.valid)):
    v_features[0].append(reader.valid[i][1][0][0])
    v_features[1].append(reader.valid[i][1][0][1])
    v_targets.append(reader.valid[i][1][1])
    v_id.append(reader.valid[i][0])

for i in xrange(len(reader.valid_im)):
    v_im_features[0].append(reader.valid_im[i][1][0][0])
    v_im_features[1].append(reader.valid_im[i][1][0][1])
    v_im_targets.append(reader.valid_im[i][1][1])
    v_im_id.append(reader.valid_im[i][0])

features = [np.array(features[0]), np.array(features[1])]
targets = np.array(targets)
v_features = [np.array(v_features[0]), np.array(v_features[1])]
v_targets = np.array(v_targets)
vv_features = [np.array(v_im_features[0]), np.array(v_im_features[1])]
vv_targets = np.array(v_im_targets)


cnnlstm = SCNN(conf)
# cnnlstm.build_net()
# cnnlstm.save()
cnnlstm.load()
# cnnlstm.train(features, targets, v_features, v_targets, vv_features, vv_targets, v_id, v_im_id)
cnnlstm.test(v_features, v_targets, vv_features, vv_targets, v_id, v_im_id)
# cnnlstm.save()

