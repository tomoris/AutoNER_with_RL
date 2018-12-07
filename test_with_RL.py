#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle

import hjson as json
import torch

import loader
from calc_F_score import calc_F_score  # ~/work/misc
from model.AutoNER_with_RL import AutoNER_with_RL


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is BiLSTM-CRF \
        trainable imcomplete annotation')
    parser.add_argument('--model', type=str, default=None,
                        help='load a model information, please input file stem \
        without ".info"/".weight"')
    parser.add_argument('--test_file', type=str, default=None,
                        help='test corpus')
    parser.add_argument('--minibatch', type=int, default=32)
    parser.add_argument('--token_splitter', type=str, default=' ',
                        help="that's/O a/O apple-pen/PPAP -> \
        [that's/O, a/O, apple-pen/PP|AP]")
    parser.add_argument('--tag_splitter', type=str, default='/',
                        help=" -> [O, O, PP|AP]")
    args = parser.parse_args()

    test_data = open(args.test_file, 'r').read().split('\n')
    test_data = [_.split(args.token_splitter) for _ in test_data if _ != '']

    with open(args.model + '.info.hjson', mode='rb') as f:
        model_info = pickle.load(f)
    autoner = AutoNER_with_RL(model_info['prams'],
                              model_info['w2i'],
                              model_info['i2w'],
                              model_info['ch2i'],
                              model_info['i2ch'],
                              model_info['t2i'],
                              model_info['i2t'])
    autoner.load_state_dict(torch.load(args.model + '.weight',
                                       map_location=lambda storage, loc: storage))
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
        if int(gpu_id) >= 0:
            cuda_device = torch.device('cuda')
            autoner.to(cuda_device)
            autoner.to_gpu(cuda_device)

    dev_sents, dev_sent_labels = \
        loader.read_corpus_column(args.test_file, {}, {}, {}, {}, {}, {}, {}, {},
                                  '\n\n', '\n', ' ', None, 0, 0, None,
                                  vocab_expand=False, return_BIESO=True)
    dev_corpus = [dev_sents, dev_sent_labels]
    for i, (sent, sent_label) in enumerate(zip(dev_sents, dev_sent_labels)):
        if len(sent) == 0:
            dev_sents.remove(sent)
            dev_sent_labels.remove(sent_label)
    assert(len(dev_corpus[0]) == len(dev_corpus[1]))

    minibatch = args.minibatch
    autoner.eval()
    predict_sents = []
    dev_sents = dev_corpus[0]
    for i in range(0, len(dev_sents), minibatch):
        input_sents = []
        for r in range(i, min(i+minibatch, len(dev_corpus[0]))):
            input_sents.append(dev_sents[r])
        predicts = autoner.predict(input_sents)
        predict_sents.extend(predicts)
    prec, recall, f_score = calc_F_score(dev_corpus[1],
                                         predict_sents,
                                         ' ',
                                         '</>',
                                         'O',
                                         '-',
                                         0)
    print(prec, recall, f_score)
    # for words in test_data:
    #     predict_tags = model.predict([words])
    #     newline = ''
    #     for t, word in enumerate(words):
    #         newline += word + args.tag_splitter + predict_tags[t] + \
    #             args.token_splitter
    #     newline = newline[:-len(args.token_splitter)]
    #     print(newline)
