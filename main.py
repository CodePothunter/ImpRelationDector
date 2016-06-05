#!/usr/bin/python
# -*- coding:utf8 -*-

#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

from utils.reader import Reader
from utils.nn import CNN, CnnLstm, SCNN, Lstm

import numpy as np

# from keras.utils.np_utils import to_categorical
# print to_categorical([1,2],10)



conf = {
    "vocab_size": 100000,
    "maxlen": 420,
    "batch_size": 20,
    "embedding_dims": 100,
    "nb_filter": 250,
    "filter_length": 20,
    "pool_length": 2,
    "hidden_size": 200,
    "nb_epoch": 50,
    "dropout": 0.5, 
    "train_file": "data/train_pdtb_imp.json",
    "vocab_file": "data/vocab",
    "test_file": "",
    "valid_file": "data/dev_pdtb_imp.json",
    "vocab_size": 100000,
}
print str(conf)
reader = Reader(conf)
reader.get_full_train_data()
reader.get_full_valid_data(get_id=True)

features = [[[], []],[[], []]]
targets = []
v_features = [[[], []],[[], []]]
v_targets = []
v_id = []
v_im_features = [[[], []],[[], []]]
v_im_targets = []
v_im_id = []

# for i in xrange(len(reader.train)):
#     features[0].append(reader.train[i][0][0])
#     features[1].append(reader.train[i][0][1])
#     targets.append(reader.train[i][1])

# for i in xrange(len(reader.valid)):
#     v_features[0].append(reader.valid[i][1][0][0])
#     v_features[1].append(reader.valid[i][1][0][1])
#     v_targets.append(reader.valid[i][1][1])
#     v_id.append(reader.valid[i][0])

# for i in xrange(len(reader.valid_im)):
#     v_im_features[0].append(reader.valid_im[i][1][0][0])
#     v_im_features[1].append(reader.valid_im[i][1][0][1])
#     v_im_targets.append(reader.valid_im[i][1][1])
#     v_im_id.append(reader.valid_im[i][0])

for i in xrange(len(reader.train)):
    features[0][0].append(reader.train[i][0][0])
    features[0][1].append(reader.train[i][2][0])
    features[1][0].append(reader.train[i][0][1])
    features[1][1].append(reader.train[i][2][1])
    targets.append(reader.train[i][1])

for i in xrange(len(reader.valid)):
    v_features[0][0].append(reader.valid[i][1][0][0])
    v_features[0][1].append(reader.valid[i][1][2][0])
    v_features[1][0].append(reader.valid[i][1][0][1])
    v_features[1][1].append(reader.valid[i][1][2][1])
    v_targets.append(reader.valid[i][1][1])
    v_id.append(reader.valid[i][0])

for i in xrange(len(reader.valid_im)):
    v_im_features[0][0].append(reader.valid[i][1][0][0])
    v_im_features[0][1].append(reader.valid[i][1][2][0])
    v_im_features[1][0].append(reader.valid[i][1][0][1])
    v_im_features[1][1].append(reader.valid[i][1][2][1])
    v_im_targets.append(reader.valid_im[i][1][1])
    v_im_id.append(reader.valid_im[i][0])

x = [np.array(features[0][0]), np.array(features[0][1]), np.array(features[1][0]), np.array(features[1][1])]
y = np.array(targets)
v_x = [np.array(v_features[0][0]), np.array(v_features[0][1]), np.array(v_features[1][0]), np.array(v_features[1][1])]
v_y = np.array(v_targets)
vv_features = [np.array(v_im_features[0][0]), np.array(v_im_features[0][1]),
            np.array(v_im_features[1][0]), np.array(v_im_features[1][1])]
vv_targets = np.array(v_im_targets)


cnnlstm = CNN(conf)
cnnlstm.build_net()
# cnnlstm.save()
#cnnlstm.load()
# print x
# print v_x
cnnlstm.train(x, y, v_x, v_y)
# cnnlstm.model.fit([x[0],x[2]], y, [v_x[0], v_x[2]], v_y)
#cnnlstm.test(v_features, v_targets, vv_features, vv_targets, v_id, v_im_id)
# cnnlstm.save()

