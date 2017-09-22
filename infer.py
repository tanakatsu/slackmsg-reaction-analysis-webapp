#!/usr/bin/env python
"""Example to generate text from a recurrent neural network language model.

This code is ported from following implementation.
https://github.com/longjie/chainer-char-rnn/blob/master/sample.py

"""
import json
import numpy as np
import six
import six.moves.cPickle as pickle

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers

import net


class Predictor():
    INPUT_VOCAB_FILE = 'model/input_vocab.bin'
    LABEL_VOCAB_FILE = 'model/label_vocab.bin'
    TRAINED_MODEL_FILE = 'model/rnnlm.model'

    def __init__(self, n_units, seq_len):
        self.seq_len = seq_len

        self.input_vocab, self.label_vocab = self.load_data()
        self.input_ivocab = {i: c for c, i in self.input_vocab.items()}
        self.label_ivocab = {i: c for c, i in self.label_vocab.items()}

        lm = net.RNNLM(len(self.input_vocab), len(self.label_vocab), n_units, train=False)
        self.model = L.Classifier(lm)
        serializers.load_npz(self.TRAINED_MODEL_FILE, self.model)

    def load_data(self):
        with open(self.INPUT_VOCAB_FILE, 'rb') as f:
            input_vocab = pickle.load(f)
        with open(self.LABEL_VOCAB_FILE, 'rb') as f:
            label_vocab = pickle.load(f)
        return input_vocab, label_vocab

    def conv(self, words):
        u_words = [x.decode('utf-8') for x in words]  # str -> utf-8
        # convert to int
        # int_data = [self.input_vocab[x] if x in self.input_vocab.keys() else 0 for x in u_words]  # 0 for unknown word
        int_data = [self.input_vocab[x] for x in u_words if x in self.input_vocab.keys()]  # skip unkkown word
        # padding
        return ([0] * self.seq_len + int_data)[-self.seq_len:]

    def iconv(self, int_data):
        return [self.input_ivocab[x] for x in int_data if not x == 0]

    def eval(self, words):
        n_top = 3
        data = self.conv(words)
        data = np.array(data, dtype=np.int32)
        print(data)

        self.model.predictor.reset_state()
        for j in six.moves.range(self.seq_len):
            word = chainer.Variable(np.array([data[j]]), volatile='on')
            pred = F.softmax(self.model.predictor(word))
            if j == self.seq_len - 1:
                pred_data = pred.data
                indice = pred_data[0].argsort()[-n_top:][::-1]
                probs = pred_data[0][indice]
                result = [(self.label_ivocab[idx], float(prob)) for (idx, prob) in zip(indice, probs)]
        return result
