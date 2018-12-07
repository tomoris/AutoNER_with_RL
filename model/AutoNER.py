#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn as nn
from torch.utils.checkpoint import checkpoint

PAD = '<pad>'
PAD_id = 0
UNK = '<unk>'
UNK_id = 1
BOS = '<bos>'
BOS_id = 2
EOS = '<eos>'
EOS_id = 3

space = ' '


class AutoNER(nn.Module):
    def __init__(self, prams, w2id, id2w, ch2id, id2ch, tag2id, id2tag):
        super(AutoNER, self).__init__()
        self.w_embed_dim = prams['w_embed_dim']
        self.ch_embed_dim = prams['ch_embed_dim']
        self.ch_rnn_dim = prams['ch_rnn_dim']
        self.span2id = {'unknown': 0, 'break': 1,
                        'tie': 2}  # unknown is ignored
        self.id2span = {0: 'unknown', 1: 'break', 2: 'tie'}
        self.break_id = self.span2id['break']
        self.span_out_size = len(self.span2id)
        self.type_out_size = len(tag2id)  # 'None' is ignored
        self.w_norm = prams['w_norm']

        self.w2id = w2id
        self.id2w = id2w
        self.ch2id = ch2id
        self.id2ch = id2ch
        self.tag2id = tag2id
        self.id2tag = id2tag
        self.w_space_id = self.w2id[space]
        self.ch_space_id = self.ch2id[space]

        self.drop = nn.Dropout(p=prams['dropout'])
        self.sigmoid = nn.Sigmoid()

        self.w_embed = nn.Embedding(len(self.w2id), self.w_embed_dim,
                                    padding_idx=PAD_id, sparse=True).float()
        self.ch_embed = nn.Embedding(len(self.ch2id), self.ch_embed_dim,
                                     padding_idx=PAD_id, sparse=True).float()
        self.ch_rnn = nn.LSTM(self.ch_embed_dim + self.w_embed_dim,
                              self.ch_rnn_dim // 2,
                              batch_first=True, bidirectional=True).float()

        self.fc_for_span_detection_1 = nn.Linear(
            self.ch_rnn_dim, 50).float()
        self.fc_for_span_detection_2 = nn.Linear(
            50, self.span_out_size).float()
        self.fc_for_type_classification_1 = nn.Linear(
            2*self.ch_rnn_dim, 50).float()
        self.fc_for_type_classification_2 = nn.Linear(
            50, self.type_out_size).float()

        # unknown id (== 0) and None id (== 0) is ignored
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

        self.one = torch.ones((1,), dtype=torch.uint8)
        self.zero = torch.zeros((1,), dtype=torch.uint8)
        self.device = None

    def to_gpu(self, cuda_device):
        self.device = cuda_device
        self.one = torch.ones((1,), dtype=torch.uint8, device=self.device)
        self.zero = torch.zeros((1,), dtype=torch.uint8, device=self.device)

    def load_pretraining_embeddings(self, embeddings):
        self.w_embed.weight.data.copy_(
            torch.from_numpy(np.array(embeddings))).float()

    def adjust_input(self, input_batch):
        minibatch = len(input_batch)
        input_batch_w = [[] for b in range(minibatch)]
        input_batch_ch = [[] for b in range(minibatch)]
        mask = [[] for b in range(minibatch)]
        for b in range(minibatch):
            cur_word = 0
            chars = list(space.join(input_batch[b]))
            input_batch_w[b].append(BOS_id)
            input_batch_ch[b].append(BOS_id)
            mask[b].append(0)
            input_batch_w[b].append(self.w_space_id)
            input_batch_ch[b].append(self.ch_space_id)
            mask[b].append(1)
            for ch in chars:
                if ch == space:
                    input_batch_w[b].append(self.w_space_id)
                    input_batch_ch[b].append(self.ch_space_id)
                    mask[b].append(1)
                    cur_word += 1
                else:
                    if self.w_norm:
                        word_norm = input_batch[b][cur_word].lower()
                    else:
                        word_norm = input_batch[b][cur_word]
                    input_batch_w[b].append(self.w2id.get(word_norm, UNK_id))
                    input_batch_ch[b].append(self.ch2id.get(ch, UNK_id))
                    mask[b].append(0)

            input_batch_w[b].append(self.w_space_id)
            input_batch_ch[b].append(self.ch_space_id)
            mask[b].append(1)
            input_batch_w[b].append(EOS_id)
            input_batch_ch[b].append(EOS_id)
            mask[b].append(0)

        return input_batch_w, input_batch_ch, mask

    def forward(self, input_batch_w, input_batch_ch, mask):
        w_emb = self.w_embed(input_batch_w)
        ch_emb = self.ch_embed(input_batch_ch)

        emb = torch.cat([w_emb, ch_emb], dim=2)
        emb = self.drop(emb)

        h, (_, _) = self.ch_rnn(emb)
        # spaceに関するoutputだけを取り出す
        h = h.masked_select(mask.unsqueeze(2).expand(-1, -1, self.ch_rnn_dim))
        h = h.view(-1, self.ch_rnn_dim)
        return h

    def detect_span(self, h):
        # space_out = self.fc_for_span_detection(h)
        space_out = self.fc_for_span_detection_1(h)
        space_out = self.fc_for_span_detection_2(self.drop(space_out))
        return space_out

    def classify_chunk(self, h, mask):
        chunk_emb = h.masked_select(mask.unsqueeze(1).expand(-1,
                                                             self.ch_rnn_dim))
        chunk_emb = chunk_emb.view(-1, self.ch_rnn_dim)
        chunk_emb = torch.cat(
            [chunk_emb[:-1, :], chunk_emb[1:, :]], dim=1)
        chunk_emb = chunk_emb.view(-1, 2*self.ch_rnn_dim)
        # type_out = self.fc_for_type_classification(chunk_emb)
        type_out = self.fc_for_type_classification_1(chunk_emb)
        type_out = self.fc_for_type_classification_2(self.drop(type_out))
        return type_out

    def calc_loss(self, input_batch, target_span_tags, target_type_tags):
        input_batch_w_list = input_batch[0]
        input_batch_ch_list = input_batch[1]
        mask_list = input_batch[2]

        # padding
        minibatch = len(input_batch_w_list)
        max_sent_len = max([len(sent) for sent in input_batch_ch_list])
        padded_input_batch_ch_list = [None for i in range(minibatch)]
        padded_input_batch_w_list = [None for i in range(minibatch)]
        padded_mask_list = [None for i in range(minibatch)]
        for b in range(len(input_batch_ch_list)):
            pad_size = max_sent_len - len(input_batch_ch_list[b])
            padded_input_batch_ch_list[b] = input_batch_ch_list[b] + \
                [PAD_id] * pad_size
            padded_input_batch_w_list[b] = input_batch_w_list[b] + \
                [PAD_id] * pad_size
            padded_mask_list[b] = mask_list[b] + [0] * pad_size

        input_batch_w = torch.tensor(
            padded_input_batch_w_list, dtype=torch.long, device=self.device)
        input_batch_ch = torch.tensor(
            padded_input_batch_ch_list, dtype=torch.long, device=self.device)
        mask = torch.tensor(
            padded_mask_list, dtype=torch.uint8, device=self.device)

        loss = self.calc_tagger_loss(input_batch_w, input_batch_ch, mask,
                                     target_span_tags, target_type_tags)
        return loss

    def calc_tagger_loss(self, input_batch_w, input_batch_ch, mask,
                         target_span_tags, target_type_tags):
        h = self.forward(input_batch_w, input_batch_ch, mask)

        h = self.drop(h)
        span_out = self.detect_span(h)
        target_span = []
        for flatten_item in target_span_tags:
            flatten_item = list(map(lambda a: self.span2id[a], flatten_item))
            flatten_item = [self.break_id] + flatten_item
            target_span.extend(flatten_item)
        target_span = torch.tensor(target_span, device=self.device)
        span_loss = self.cross_entropy(span_out, target_span)

        mask = torch.where(target_span == self.span2id['break'],
                           self.one.expand(span_out.size()[0]),
                           self.zero.expand(span_out.size()[0]))

        type_out = self.classify_chunk(h, mask)
        target_type = []
        for flatten_item in target_type_tags:
            target_type.extend(map(lambda a: self.tag2id[a], flatten_item))
            target_type.append(PAD_id)
        target_type = torch.tensor(target_type[:-1], device=self.device)
        type_loss = self.cross_entropy(type_out, target_type)

        loss = 0.5 * (span_loss + type_loss)
        return loss

    def predict(self, input_batch):
        input_batch_w, input_batch_ch, mask = self.adjust_input(input_batch)

        # padding
        max_sent_len = max([len(sent) for sent in input_batch_ch])
        for b in range(len(input_batch_ch)):
            pad_size = max_sent_len - len(input_batch_ch[b])
            input_batch_ch[b] += [PAD_id] * pad_size
            input_batch_w[b] += [PAD_id] * pad_size
            mask[b] += [0] * pad_size

        input_batch_w = torch.tensor(
            input_batch_w, dtype=torch.long, device=self.device)
        input_batch_ch = torch.tensor(
            input_batch_ch, dtype=torch.long, device=self.device)
        mask = torch.tensor(mask, dtype=torch.uint8, device=self.device)

        h = self.forward(input_batch_w, input_batch_ch, mask)
        span_out = self.detect_span(h)
        # unknown のタグが選ばれないようにする
        span_out[:, self.span2id['unknown']] = -10000.0
        span_out = torch.argmax(span_out, dim=1)
        sent_word_lens = list(map(len, input_batch))
        cur_sent_word_t = -1
        for sent_word_len in sent_word_lens:
            cur_sent_word_t += 1
            span_out[cur_sent_word_t] = 1
            cur_sent_word_t += sent_word_len
            span_out[cur_sent_word_t] = 1

        span = torch.where(span_out == self.span2id['break'],
                           self.one.expand(span_out.size()[0]),
                           self.zero.expand(span_out.size()[0]))
        type_out = self.classify_chunk(h, span)
        type_out[:, PAD_id] = -10000.0   # pad のタグが選ばれないようにする
        NE_type = torch.argmax(type_out, dim=1)
        cur_span_t = 0
        cur_type_t = 0
        newlines = []
        for sent in input_batch:
            newline = ''
            prev_span = 'break'
            cur_span_t += 1
            for word in sent:
                span_name = self.id2span[span_out[cur_span_t].item()]
                type_name = self.id2tag[NE_type[cur_type_t].item()]
                if type_name == 'None':
                    NE_tag = 'O'
                else:
                    if prev_span == 'break':
                        NE_tag = 'B-' + type_name
                    elif prev_span == 'tie':
                        NE_tag = 'I-' + type_name
                newline += word + '</>' + NE_tag + ' '
                if span_name == 'break':
                    cur_type_t += 1
                cur_span_t += 1
                prev_span = span_name
            newline = newline[:-1]
            newlines.append(newline)
            cur_type_t += 1
        return newlines

    def calc_prob_given_span_and_type(self, input_batch_w, input_batch_ch,
                                      mask, target_span_tags,
                                      target_type_tags):
        # # padding
        # max_sent_len = max([len(sent) for sent in input_batch_ch])
        # for b in range(len(input_batch_ch)):
        #     pad_size = max_sent_len - len(input_batch_ch[b])
        #     input_batch_ch[b] += [PAD_id] * pad_size
        #     input_batch_w[b] += [PAD_id] * pad_size
        #     mask[b] += [0] * pad_size

        # input_batch_w = torch.tensor(
        #     input_batch_w, dtype=torch.long, device=self.device)
        # input_batch_ch = torch.tensor(
        #     input_batch_ch, dtype=torch.long, device=self.device)
        # mask = torch.tensor(mask, dtype=torch.uint8, device=self.device)

        h = self.forward(input_batch_w, input_batch_ch, mask)
        span_out = self.detect_span(h)

        target_span = []
        for flatten_item in target_span_tags:
            flatten_item = list(map(lambda a: self.span2id[a], flatten_item))
            flatten_item = [self.break_id] + flatten_item
            target_span.extend(flatten_item)
        target_span = torch.tensor(target_span, device=self.device)

        mask_span = torch.where(target_span == self.span2id['break'],
                                self.one.expand(span_out.size()[0]),
                                self.zero.expand(span_out.size()[0]))

        # mask_tie = torch.where(target_span == self.span2id['tie'],
        #                        self.one.expand(span_out.size()[0]),
        #                        self.zero.expand(span_out.size()[0]))

        # calc span_prob
        span_prob_item = torch.softmax(span_out, dim=1)
        span_prob = span_prob_item.masked_select(
            mask_span.unsqueeze(1).expand(-1, len(self.span2id)))
        span_prob = span_prob.view(-1, len(self.span2id))
        span_prob = span_prob[:-1, self.span2id['break']
                              ] * span_prob[1:, self.span2id['break']]
        # tieを考慮
        cur_NE_index = -1
        for i, span_tag in enumerate(target_span):
            if span_tag == self.span2id['break']:
                cur_NE_index += 1
            elif span_tag == self.span2id['tie']:
                span_prob[cur_NE_index] *= span_prob_item[i][self.span2id['tie']]

        # calc type_prob
        # TODO 複数のクラスを考慮
        type_out = self.classify_chunk(h, mask_span)
        type_out = torch.softmax(type_out, dim=1)
        target_type = []
        for flatten_item in target_type_tags:
            target_type.extend(map(lambda a: self.tag2id[a], flatten_item))
            target_type.append(PAD_id)
        target_type = torch.tensor(target_type[:-1], device=self.device)

        type_prob = type_out[torch.arange(type_out.size()[0]), target_type]
        total_prob = span_prob * type_prob

        # mask処理によって、NEの確率だけを返す
        mask_NEs = torch.where(target_type != PAD_id,
                               self.one.expand(type_out.size()[0]),
                               self.zero.expand(type_out.size()[0]))
        total_prob = total_prob.masked_select(mask_NEs)
        return total_prob
