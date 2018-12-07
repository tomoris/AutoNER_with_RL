#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import argparse
import time
import pickle

import hjson as json
import torch
import torch.optim as optim

from model.AutoNER_with_RL import AutoNER_with_RL
import loader
from calc_F_score import calc_F_score  # ~/work/misc

PAD = '<pad>'
PAD_id = 0
UNK = '<unk>'
UNK_id = 1
BOS = '<bos>'
BOS_id = 2
EOS = '<eos>'
EOS_id = 3


def calc_dev_score(autoner, dev_corpus, token_splitter, tag_splitter, neg_tag,
                   BIES_splitter, index_of_splitted_BIES):
    autoner.eval()
    minibatch_RL = 2048
    autoner.eval()
    predict_sents = []
    dev_sents = dev_corpus[0]
    for j in range(0, len(dev_sents), minibatch_RL):
        input_sents = []
        for r in range(j, min(j+minibatch_RL, len(dev_corpus[0]))):
            input_sents.append(dev_sents[r])
        predicts = autoner.predict(input_sents)
        predict_sents.extend(predicts)
    prec, recall, f_score = calc_F_score(dev_corpus[1],
                                         predict_sents,
                                         token_splitter,
                                         tag_splitter,
                                         neg_tag,
                                         BIES_splitter,
                                         index_of_splitted_BIES)
    autoner.train()
    return prec, recall, f_score


def make_NE_info_list(sents, NE_dict_file):
    NE_dict = {}
    for line in open(NE_dict_file):
        line = line.replace('\n', '')
        line_sp = line.split('\t')
        NE_dict[line_sp[1].lower()] = line_sp[0]
    N = 8
    NE_info_list = [[], []]
    for sent in sents:
        NE_info_list[0].append([])
        start = 0
        flag = False
        while(start < len(sent)):
            end = min(start+N+1, len(sent))
            ngram_end_list = list(range(start+1, end))
            ngram_end_list.reverse()
            for ngram_end in ngram_end_list:
                ngram = ' '.join(sent[start:ngram_end]).lower()
                if ngram in NE_dict:
                    NE_info_list[0][-1].append(sent[start:ngram_end])
                    start += len(sent[start:ngram_end])
                    flag = True
                    break
            if flag:
                flag = False
            else:
                start += 1
        NE_info_list[1].append(len(NE_info_list[0][-1]))
    return NE_info_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default=None,
                        help="load config json file")
    args = parser.parse_args()
    h_pram = json.load(open(args.config, 'r'))

    if h_pram['load'] is not None:
        h_pram['pretrain_file'] = None
    input_corpus = [h_pram['train_file']]
    bool_full_annotaion_corpus = [False]
    print('Loading files')
    train_data_list, train_tags_list, w2i, i2w, ch2i, i2ch, t2i, i2t, embeddings = \
        loader.adjust_dataset(input_corpus, bool_full_annotaion_corpus,
                              pretrain_file=h_pram['pretrain_file'],
                              line_splitter=h_pram['line_splitter'],
                              token_splitter=h_pram['token_splitter'],
                              tag_splitter=h_pram['tag_splitter'],
                              multi_tag_splitter=h_pram['multi_tag_splitter'],
                              w_TH=h_pram['prams']['w_TH'],
                              ch_TH=h_pram['prams']['ch_TH'],
                              w_norm=h_pram['prams']['w_norm'])

    if h_pram['valid']:
        train_end_index = int(len(train_data_list[0]) * 0.95)
        train_data = train_data_list[0][:train_end_index]
        train_span_tags = train_tags_list[0][0][:train_end_index]
        train_type_tags = train_tags_list[0][1][:train_end_index]
        valid_data = train_data_list[0][train_end_index:]
        valid_span_tags = train_tags_list[0][0][train_end_index:]
        valid_type_tags = train_tags_list[0][1][train_end_index:]
    else:
        train_data = train_data_list[0]
        train_span_tags = train_tags_list[0][0]
        train_type_tags = train_tags_list[0][1]
        valid_data = []

    if 'None' not in t2i:
        t2i['None'] = len(t2i)
        i2t[len(i2t)] = 'None'

    if h_pram['dev_file'] is not None:
        dev_sents, dev_sent_labels = \
            loader.read_corpus_column(h_pram['dev_file'],
                                      {}, {}, {}, {}, {}, {}, {}, {},
                                      '\n\n', '\n', ' ', None, 0, 0, None,
                                      vocab_expand=False, return_BIESO=True)
        dev_corpus = [dev_sents, dev_sent_labels]
        for i, (sent, sent_label) in enumerate(zip(dev_sents, dev_sent_labels)):
            if len(sent) == 0:
                dev_sents.remove(sent)
                dev_sent_labels.remove(sent_label)
        assert(len(dev_corpus[0]) == len(dev_corpus[1]))

    print('Building model')
    if h_pram['load'] is None:
        autoner = AutoNER_with_RL(
            h_pram['prams'], w2i, i2w, ch2i, i2ch, t2i, i2t)
        if h_pram['pretrain_file'] is not None:
            autoner.load_pretraining_embeddings(embeddings)
    else:
        with open(h_pram['load'] + '.info.hjson', mode='rb') as f:
            model_info = pickle.load(f)
        # pythonのjsonではdictのkeyをintからstrに変換して保存するので、戻す
        model_info['i2w'] = {int(i): w for i, w in model_info['i2w'].items()}
        model_info['i2t'] = {int(i): t for i, t in model_info['i2t'].items()}
        autoner = AutoNER_with_RL(model_info['prams'], model_info['w2i'],
                                  model_info['i2w'], model_info['ch2i'],
                                  model_info['i2ch'], model_info['t2i'],
                                  model_info['i2t'])
        autoner.load_state_dict(torch.load(h_pram['load'] + '.weight',
                                           map_location=lambda storage, loc: storage))
        autoner.all_select = h_pram['prams']['all_select']

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
        if int(gpu_id) >= 0:
            cuda_device = torch.device('cuda')
            autoner.to(cuda_device)
            autoner.to_gpu(cuda_device)
    if h_pram['prams']['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(autoner.parameters(),
                              lr=h_pram['prams']['initial_lr'],
                              momentum=h_pram['prams']['momentum'])
        optimizer_RL = optim.SGD(autoner.parameters(),
                                 lr=h_pram['prams']['RL_lr'])
    elif h_pram['prams']['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(autoner.parameters(),
                               weight_decay=h_pram['prams']['weight_decay'])
    elif h_pram['prams']['optimizer_name'] == 'SparseAdam':
        optimizer = optim.SparseAdam(autoner.Parameter)

    train_data_adjust = []
    for data in train_data:
        input_batch_w, input_batch_ch, mask = autoner.adjust_input([data])
        train_data_adjust.append(
            (input_batch_w[0], input_batch_ch[0], mask[0]))
    valid_data_adjust = []
    for data in valid_data:
        input_batch_w, input_batch_ch, mask = autoner.adjust_input([data])
        valid_data_adjust.append(
            (input_batch_w[0], input_batch_ch[0], mask[0]))

    NE_info_list = make_NE_info_list(train_data, h_pram['dict_file'])
    for i, NEs in enumerate(NE_info_list[0]):
        for j, NE in enumerate(NEs):
            for k, word in enumerate(NE):
                NE_info_list[0][i][j][k] = autoner.tagger.w2id.get(word.lower(),
                                                                   UNK_id)

    if h_pram['save'] is not None:
        h_pram['w2i'] = w2i
        h_pram['i2w'] = i2w
        h_pram['ch2i'] = ch2i
        h_pram['i2ch'] = i2ch
        h_pram['t2i'] = t2i
        h_pram['i2t'] = i2t
        with open(h_pram['save'] + '.info.hjson', mode='wb') as f:
            pickle.dump(h_pram, f)
        torch.save(autoner.state_dict(),
                   h_pram['save'] + '.weight' + '.initial')

    prec, recall, f_score = calc_dev_score(autoner, dev_corpus,
                                           ' ', '</>',
                                           h_pram['neg_tag'],
                                           h_pram['BIES_splitter'],
                                           h_pram['index_of_splitted_BIES'])
    original_f_score = f_score

    print('Training model')
    lr = float(h_pram['prams']['initial_lr'])
    max_dev_score = 0.0
    f_score = 0.0
    minibatch = h_pram['prams']['batch']
    prev_rewards = [0.0 for i in range(5)]
    max_reward = 0.0
    RL_epoch = h_pram['prams']['RL_epoch']
    for epoch in range(h_pram['prams']['epoch']):
        # 今だけ
        if epoch == 0:
            autoner.tagger.drop = torch.nn.Dropout(p=0.0)
            autoner.all_select = True
            minibatch = 1
        else:
            autoner.tagger.drop = torch.nn.Dropout(
                p=h_pram['prams']['dropout'])
            autoner.all_select = h_pram['prams']['all_select']
            minibatch = h_pram['prams']['batch']
        autoner.train()
        print('epoch', epoch, flush=True)
        total_loss = 0.0
        total_minus_reward = 0.0
        rand_list = [r for r in range(len(train_data_adjust))]
        random.shuffle(rand_list)
        start_time = time.time()
        autoner.epsilon = max(0.05, autoner.epsilon * 0.95)
        total_select_num = 0
        itr = 0
        for i in range(0, len(train_data_adjust), minibatch):
            itr += 1
            if itr % 100 == 0:
                print(itr,
                      '{0:.4f}%'.format(
                          itr / (len(train_data_adjust) // minibatch)))
            input_batch = [[], [], []]
            target_span_tags = []
            target_type_tags = []
            NE_info = [[], []]
            for r in range(i, min(i+minibatch, len(train_data_adjust))):
                data = train_data_adjust[rand_list[r]]
                input_batch[0].append(data[0])
                input_batch[1].append(data[1])
                input_batch[2].append(data[2])
                target_span_tags.append(train_span_tags[rand_list[r]])
                target_type_tags.append(train_type_tags[rand_list[r]])
                NE_info[0].append(NE_info_list[0][rand_list[r]])
                NE_info[1].append(NE_info_list[1][rand_list[r]])

            loss, A, select_num = autoner.calc_tagger_loss(input_batch,
                                                           target_span_tags,
                                                           target_type_tags,
                                                           NE_info)
            total_select_num += select_num
            if loss is not None:
                autoner.zero_grad()
                loss.backward(retain_graph=True)
                lr = float(h_pram['prams']['initial_lr'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.step()
                total_loss += loss.item()

            if epoch >= RL_epoch:
                prec, recall, f_score = calc_dev_score(autoner, dev_corpus,
                                                       ' ', '</>',
                                                       h_pram['neg_tag'],
                                                       h_pram['BIES_splitter'],
                                                       h_pram['index_of_splitted_BIES'])
                reward_t = f_score
                if epoch == 0:
                    reward = reward_t - original_f_score
                else:
                    reward = reward_t - (sum(prev_rewards) / len(prev_rewards))
                autoner.train()
                if epoch == 0:
                    lr = float(h_pram['prams']['initial_lr']) * -1.0
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    optimizer.step()

                autoner.train()
                autoner.zero_grad()
                reward *= -1.0 * torch.log(A.mean())
                optimizer_RL.step()
                prev_rewards.append(reward_t)
                prev_rewards = prev_rewards[1:]
                total_minus_reward += reward_t

        print(total_loss, total_minus_reward, total_select_num)

        # Evaluate now model using development corpus
        if h_pram['dev_file'] is not None:
            prec, recall, f_score = calc_dev_score(autoner, dev_corpus, ' ',
                                                   '</>',
                                                   h_pram['neg_tag'],
                                                   h_pram['BIES_splitter'],
                                                   h_pram['index_of_splitted_BIES'])
            print(prec, recall, f_score)

        end_time = time.time()
        calc_time = end_time - start_time
        print('calc time = ', calc_time)
        # Save
        if h_pram['save'] is not None and ((h_pram['dev_file'] is not None and
                                            f_score > max_dev_score)
                                           or epoch == 0):
            max_dev_score = f_score
            torch.save(autoner.state_dict(),
                       h_pram['save'] + '.weight' + '.epoch_' + str(epoch))
    print('Done')
