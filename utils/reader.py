#!/usr/bin/python
# -*- coding:utf8 -*-

import json
label2idx = {"Instantiation":0,"Synchrony":1,"Pragmatic cause":2,"List":3,"Asynchronous":4,"Restatement":5,"Concession":6,"Conjunction":7,"Cause":8,"Alternative":9,"Contrast":10, "Other":11}
pos2idx = {"PRP$":0,"VBG":1,"VBD":2,"VB":3,",":4,"''":5,"VBP":6,"VBN":7,"JJ":8,"WP":9,"VBZ":10,"DT":11,"#":12,"RP":13,"$":14,"NN":15,"FW":16,"POS":17,".":18,"TO":19,"PRP":20,"RB":21,"-LRB-":22,":":23,"NNS":24,"NNP":25,"``":26,"WRB":27,"CC":28,"LS":29,"PDT":30,"RBS":31,"RBR":32,"CD":33,"EX":34,"IN":35,"WP$":36,"MD":37,"NNPS":38,"-RRB-":39,"JJS":40,"JJR":41,"UH":42,"WDT":43}
def set_vocab(file, vocab_size):
    vocab_dict = {}
    f = open(file, "r")
    line = f.readline()
    cnt = 0
    while line != "":
        cnt += 1
        if cnt % 10 == 0:
            print str(cnt)+'\b'*(len(str(cnt))+2),

        words = line.split()
        for word in words:
            word = word.lower()
            vocab_dict[word] = vocab_dict.get(word, 0) + 1
        line = f.readline()
    f.close()
    vocab_dict = sorted(vocab_dict.iteritems(), key=lambda d:d[1], reverse=True)[:vocab_size-1]
    f = open("data/vocab", "w")
    f.write("NULL\n")
    for word in vocab_dict:
        print word[0]
        f.write(word[0]+'\n')
    f.close()


def get_labels(label_seg):
    labels = []
    for label in label_seg:
        labels.add(label.split(".")[-1])
    return labels

def get_discourse(jsn, r, maxlen, train=False, get_pos=False):
    """
    return a list containint a pair where the first
    element is a pair of two subsentences and the 
    second element is a vector stands for their 
    label (one hot)
    """
    words1 = get_words(None, tmp = jsn["Arg1"]["Lemma"])
    words2 = get_words(None, tmp = jsn["Arg2"]["Lemma"])
    pos1 = get_words(None, tmp=jsn["Arg1"]["POS"])
    pos2 = get_words(None, tmp=jsn["Arg2"]["POS"])
    idx1 = []
    idx2 = []
    idx1_pos = []
    idx2_pos = []
    for word in words1:
        idx1.append(r.word2idx(word))
    for pos in pos1:
        idx1_pos.append(pos2idx.get(pos, 12))
    # if len(idx1) > maxlen:
    #     print len(idx1)
    while len(idx1) < maxlen:
        idx1.append(0)
    while len(idx1_pos) < maxlen:
        idx1_pos.append(0)
    for word in words2:
        idx2.append(r.word2idx(word))
    for pos in pos2:
        idx2_pos.append(pos2idx.get(pos, 12))
    # if len(idx2) > maxlen:
    #     print len(idx2)
    while len(idx2) < maxlen:
        idx2.append(0)
    while len(idx2_pos) < maxlen:
        idx2_pos.append(0)
    i = [idx1, idx2]
    p = [idx1_pos, idx2_pos]
    labels = []
    if jsn["Type"] == "Implicit":
        for label in jsn["Sense"]:
            labels.append(label2idx[label.split(".")[-1]])
    else:
        labels.append(label2idx["Other"])
    if train != True:
        o = [0]*12
        for label_idx in labels:
            o[label_idx] = 1
        if get_pos:
            io = [i, o, p]
        else:
            io = [i, o]
        return io
    else:
        io_list = []
        for label_idx in labels:
            o = [0] * 12
            o[label_idx] = 1
            if get_pos:
                io = [i, o, p]
            else:
                io = [i, o]
            io_list.append(io)
        return io_list



def get_words(jsn, tmp = None, to_lower = False):
    if tmp == None:
        tmp = jsn["Arg1"]["Lemma"] + jsn["Arg2"]["Lemma"]
    words = []
    for word in tmp:
        if u"\u00a0" in word:
            word = "NULL"
        if not to_lower:
            words.append(word.strip())
        else:
            words.append(word.strip().lower())
    return words


class Vocab(object):
    """Build the vocab of inputs"""
    def __init__(self, conf):
        self.file = conf.get("vocab_file", "")
        if self.file == "":
            self.file = conf["train_file"]
        self.idx = ["NULL"]
        self.word = {0: "NULL"}
        self.vocab_size = 1
        self.gconf = conf
        self.ml = conf["maxlen"]

    def word2idx(self, word):
        return self.word.get(word, 0)

    def idx2word(self, idx):
        if idx >= len(self.idx):
            return self.word[0]
        return self.word[idx]

    def add_word(self, word):
        if self.word2idx(word) == 0:
            self.word[word] = len(self.idx)
            self.idx.append(word)
            self.vocab_size += 1

    def read_vocab(self):
        f = open(self.file, "r")
        if self.file == self.gconf["train_file"]:
            line = f.readline()
            while line != "":
                jsn = json.loads(line)
                words = get_words(jsn, to_lower=True)
                for word in words:
                    self.add_word(word)
                line = f.readline()
        else:
            line = f.readline()
            while line != "":
                self.add_word(line[:-1])
                line = f.readline()
        f.close()
        print "Vocab read..."

    def save_vocab(self):
        f = open("data/vocab.save", "w")
        for word in self.idx[1:]:
            try:
                f.write(word+"\n")
            except:
                print word
        f.close()
        print "Vocab saved"



class Reader(object):
    """This is a reader"""
    def __init__(self, conf):
        self.train_file = conf["train_file"]
        self.test_file = conf["test_file"]
        self.valid_file = conf["valid_file"]
        self.ml = conf['maxlen']
        self.train = []
        self.valid = []
        self.valid_im = []
        self.vocab = Vocab(conf)
        self.vocab.read_vocab()

    def get_full_train_data(self):
        f = open(self.train_file, "r")
        line = f.readline()
        while line != "":
            jsn = json.loads(line)
            gd = get_discourse(jsn, self.vocab, self.ml, train=True, get_pos=True)
            for item in gd:
                self.train.append(item)
            line = f.readline()
        print "training data ready..."
        return self.train

    def get_full_valid_data(self, get_id=False):
        f = open(self.valid_file, "r")
        line = f.readline()
        while line != "":
            jsn = json.loads(line)
            idl = jsn["ID"]
            gd = get_discourse(jsn, self.vocab, self.ml, get_pos=True)
            if get_id:
                self.valid.append((idl, gd))
            else:
                self.valid.append(gd)
            if jsn["Type"] == "Implicit":
                if get_id:
                    self.valid_im.append((idl, gd))   
                else:
                    self.valid_im.append(gd)
            line = f.readline()
        print "validation data ready..."
        return self.valid
if __name__ == '__main__':
    conf = {
        "train_file": "train_pdtb_imp.json",
        "vocab_file": "data/vocab",
        "test_file": "",
        "valid_file": "",
        "vocab_size": 100000,
    }
    # build vocab from google-billion-word-corpus
    # set_vocab("/home/slhome/xyw00/Documents/google-billion-word/training-monolingual.tokenized.shuffled/big", 100000)
    # # test vocab
    # r = Vocab(conf)
    # r.read_vocab()
    # print r.vocab_size

    # # test discourse reader
    # jsn = json.loads("""{"DocID": "wsj_2201", "Arg1": {"RawText": "It considered running them during tomorrow night's World Series broadcast but decided not to when the market recovered yesterday", "NER": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], "Word": ["It", "considered", "running", "them", "during", "tomorrow", "night", "'s", "World", "Series", "broadcast", "but", "decided", "not", "to", "when", "the", "market", "recovered", "yesterday"], "POS": ["PRP", "VBD", "VBG", "PRP", "IN", "NN", "NN", "POS", "NNP", "NNP", "NN", "CC", "VBD", "RB", "TO", "WRB", "DT", "NN", "VBD", "NN"], "Lemma": ["it", "consider", "run", "they", "during", "tomorrow", "night", "'s", "World", "Series", "broadcast", "but", "decide", "not", "to", "when", "the", "market", "recover", "yesterday"]}, "Arg2": {"RawText": "Other brokerage firms, including Merrill Lynch & Co., were plotting out potential new ad strategies", "NER": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], "Word": ["Other", "brokerage", "firms", ",", "including", "Merrill", "Lynch", "&", "Co.", ",", "were", "plotting", "out", "potential", "new", "ad", "strategies"], "POS": ["JJ", "NN", "NNS", ",", "VBG", "NNP", "NNP", "CC", "NNP", ",", "VBD", "VBG", "RP", "JJ", "JJ", "NN", "NNS"], "Lemma": ["other", "brokerage", "firm", ",", "include", "Merrill", "Lynch", "&", "Co.", ",", "be", "plot", "out", "potential", "new", "ad", "strategy"]}, "Connective": {"RawText": ["meanwhile"]}, "Sense": ["Expansion.Conjunction", "Temporal.Synchrony"], "Type": "Implicit", "ID": "35975"}
    #     """)
    # print get_discourse(jsn, r)

    # test reader
    # reader = Reader(conf)
    # reader.get_full_train_data()
