#!/usr/bin/python
# -*- coding:utf8 -*-


import sys

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Lambda, LSTM
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Merge, Reshape
from keras.datasets import imdb
from keras import backend as K


def max_1d(X):
    return K.max(X, axis=1)

class CNN(object):
    """docstring for CNN"""
    def __init__(self, conf):
        self.vs = conf["vocab_size"]
        self.ml = conf["maxlen"]
        self.bs = conf["batch_size"]
        self.ed = conf["embedding_dims"]
        self.nf = conf["nb_filter"]
        self.fl = conf["filter_length"]
        self.hs = conf["hidden_size"]
        self.ep = conf["nb_epoch"]
        self.sm = conf.get("save_model", "models/default.cnn")
        self.lm = conf.get("load_model", "models/default.cnn")
        self.do = conf.get("dropout",0.2)
        self.model = Sequential()

    def build_net(self):
        model_1 = Sequential()
        model_2 = Sequential()
        model_1.add(Embedding(self.vs,
                    self.ed,
                    input_length=self.ml,
                    dropout=self.do))
        model_1.add(Convolution1D(nb_filter=self.nf,
                        filter_length=self.fl,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
        model_1.add(Lambda(max_1d, output_shape=(self.nf, )))
        model_2.add(Embedding(self.vs,
                    self.ed,
                    input_length=self.ml,
                    dropout=self.do))
        model_2.add(Convolution1D(nb_filter=self.nf,
                        filter_length=self.fl,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
        model_2.add(Lambda(max_1d, output_shape=(self.nf, )))
        self.model.add(Merge([model_1, model_2]))
        self.model.add(Dense(self.hs))
        self.model.add(Dropout(self.do))
        self.model.add(Dense(11))
        self.model.add(Activation("softmax"))
        self.model.compile(loss='categorical_crossentropy',
                        optimizer='adam')
        print "Network compile completed..."

    def save(self):
        f = open(self.sm, "w")
        f.write(self.model.to_json())
        f.close()

    def load(self):
        f = open(self.lm, "r")
        self.model = model_from_json(f.read())

    def train(self, x, y, vx, vy):
        print "Begin to train ..."
        ep = 0
        max_accuracy = 0
        while ep < self.ep:
            loss = 0
            cnt = 0
            accuracy = 0.0
            v_accuracy = 0.0
            num_of_batch = int(len(y)/self.bs)
            idx_move = num_of_batch / 60
            for i in xrange(0, len(y), self.bs):
                x_ = [x[0][i:i+self.bs], x[1][i:i+self.bs]]
                y_ = y[i:i+self.bs]
                loss_ = self.model.train_on_batch(x_, y_)
                pred_ = self.model.predict_on_batch(x_)
                acc_ = 0.0
                for j in xrange(len(pred_)):
                    max_p = np.argmax(pred_[j])
                    max_y = np.argmax(y_[j])
                    acc_ += (max_p == max_y)

                acc_ /= len(pred_)
                accuracy += acc_
                # print acc_
                loss += loss_
                cnt += 1
                sys.stdout.flush()
                if cnt % idx_move == 0:
                    sys.stderr.write("=>\b")
                    sys.stderr.flush()
            print ">"

            v_pred = self.model.predict_on_batch(vx)
            for j in xrange(len(v_pred)):
                max_p = np.argmax(v_pred[j])
                max_y = np.argmax(vy[j])
                v_accuracy += (max_p == max_y)
            v_accuracy /= len(v_pred)
            if v_accuracy > max_accuracy:
                self.save()
                max_accuracy = v_accuracy
            ep += 1
            print "Epoch {0}, training loss {1}, train-accuracy {2}, valid-accuracy {3}".format(
                ep, loss / cnt, accuracy / cnt, v_accuracy)
            sys.stdout.flush()

class CnnLstm(object):
    """docstring for cnn-lstm"""
    def __init__(self, conf):
        self.vs = conf["vocab_size"]
        self.ml = conf["maxlen"]
        self.bs = conf["batch_size"]
        self.ed = conf["embedding_dims"]
        self.nf = conf["nb_filter"]
        self.fl = conf["filter_length"]
        self.hs = conf["hidden_size"]
        self.ep = conf["nb_epoch"]
        self.pl = conf["pool_length"]
        self.sm = conf.get("save_model", "models/default.milstm")
        self.lm = conf.get("load_model", "models/default.milstm")
        self.do = conf.get("dropout",0.2)
        self.model = Sequential()

    def build_net(self):
        model_1 = Sequential()
        model_2 = Sequential()
        model_1.add(Embedding(self.vs,
                    self.ed,
                    input_length=self.ml,
                    dropout=self.do))
        model_1.add(Convolution1D(nb_filter=self.nf,
                        filter_length=self.fl,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
        model_1.add(Lambda(max_1d, output_shape=(self.nf, )))
        model_2.add(Embedding(self.vs,
                    self.ed,
                    input_length=self.ml,
                    dropout=self.do))
        model_2.add(Convolution1D(nb_filter=self.nf,
                        filter_length=self.fl,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
        model_2.add(Lambda(max_1d, output_shape=(self.nf, )))
        self.model.add(Merge([model_1, model_2],mode='concat'))
        print self.model.output_shape
        self.model.add(Reshape((2, self.nf), input_shape=(self.nf*2,)))
        self.model.add(LSTM(self.hs))
        self.model.add(Dense(self.hs))
        self.model.add(Dropout(self.do))
        self.model.add(Dense(11))
        self.model.add(Activation("softmax"))
        self.model.compile(loss='categorical_crossentropy',
                        optimizer='adam')
        print "Network compile completed..."

    def save(self):
        f = open(self.sm, "w")
        f.write(self.model.to_json())
        f.close()

    def load(self):
        f = open(self.lm, "r")
        self.model = model_from_json(f.read())

    def train(self, x, y, vx, vy):
        print "Begin to train ..."
        ep = 0
        max_accuracy = 0
        while ep < self.ep:
            loss = 0
            cnt = 0
            accuracy = 0.0
            v_accuracy = 0.0
            num_of_batch = int(len(y)/self.bs)
            idx_move = num_of_batch / 60
            for i in xrange(0, len(y), self.bs):
                x_ = [x[0][i:i+self.bs], x[1][i:i+self.bs]]
                y_ = y[i:i+self.bs]
                loss_ = self.model.train_on_batch(x_, y_)
                pred_ = self.model.predict_on_batch(x_)
                acc_ = 0.0
                for j in xrange(len(pred_)):
                    max_p = np.argmax(pred_[j])
                    max_y = np.argmax(y_[j])
                    acc_ += (max_p == max_y)

                acc_ /= len(pred_)
                accuracy += acc_
                # print acc_
                loss += loss_
                cnt += 1
                sys.stdout.flush()
                if cnt % idx_move == 0:
                    sys.stderr.write("=>\b")
                    sys.stderr.flush()
            print ">"

            v_pred = self.model.predict_on_batch(vx)
            for j in xrange(len(v_pred)):
                max_p = np.argmax(v_pred[j])
                max_y = np.argmax(vy[j])
                v_accuracy += (max_p == max_y)
            v_accuracy /= len(v_pred)
            if v_accuracy > max_accuracy:
                self.save()
                max_accuracy = v_accuracy
            ep += 1
            print "Epoch {0}, training loss {1}, train-accuracy {2}, valid-accuracy {3}".format(
                ep, loss / cnt, accuracy / cnt, v_accuracy)
            sys.stdout.flush()



if __name__ == '__main__':
    conf = {
        "vocab_size": 100000,
        "maxlen": 450,
        "batch_size": 30,
        "embedding_dims": 100,
        "nb_filter": 250,
        "filter_length":3,
        "hidden_size": 300,
        "nb_epoch": 10,
        "dropout": 0.5, 
        "train_file": "data/train_pdtb_imp.json",
        "vocab_file": "data/vocab",
        "test_file": "",
        "valid_file": "data/dev_pdtb_imp.json",
        "vocab_size": 100000,
    }

    # cnn = CNN(conf)
    # cnn.build_net()
    # print "begin"
    # for i in xrange(100000000):
    #     if i % 1000000 == 0:
    #         sys.stdout.write("\b->")
    #         sys.stdout.flush()
    lstm = CnnLstm(conf)
    lstm.build_net()


# f.close()
# f = open("dev_pdtb.json", "r")
# line = f.readline()
# while line != "":
#     l1 = jsn['Arg1']['Lemma']
#     l2 = jsn['Arg2']['Lemma']
#     if len(l1) > maxlen:
#         print l1
#         maxlen = len(l1)
#     if len(l2) > maxlen:
#         print l2
#         maxlen = len(l2)
#     line = f.readline()
