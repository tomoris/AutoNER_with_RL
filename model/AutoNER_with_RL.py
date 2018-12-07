#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn as nn
from torch.nn import functional as F

from model.AutoNER import AutoNER

PAD = '<pad>'
PAD_id = 0
UNK = '<unk>'
UNK_id = 1
BOS = '<bos>'
BOS_id = 2
EOS = '<eos>'
EOS_id = 3

space = ' '


class AutoNER_with_RL(nn.Module):
    def __init__(self, prams, w2id, id2w, ch2id, id2ch, tag2id, id2tag):
        super(AutoNER_with_RL, self).__init__()
        self.tagger = AutoNER(prams, w2id, id2w, ch2id, id2ch, tag2id, id2tag)
        self.SL_in_dim = prams['ch_rnn_dim']
        self.SL_h_dim = prams['SL_h_dim']
        self.NE_cnn_dim = prams['NE_cnn_dim']

        self.space_attention = nn.Linear(self.SL_in_dim, 1)

        self.fc_SL1 = nn.Linear(self.SL_in_dim + self.NE_cnn_dim,
                                self.SL_h_dim).float()
        self.fc_SL2 = nn.Linear(self.SL_h_dim, 1).float()

        self.NE_cnn = nn.Conv1d(prams['w_embed_dim'], self.NE_cnn_dim,
                                kernel_size=3, padding=1)

        self.one = torch.ones((1,), dtype=torch.uint8)
        self.zero = torch.zeros((1,), dtype=torch.uint8)

        self.all_select = prams['all_select']
        self.epsilon = 0.5
        self.device = None

    def to_gpu(self, cuda_device):
        self.device = cuda_device
        self.one = torch.ones((1,), dtype=torch.uint8, device=self.device)
        self.zero = torch.zeros((1,), dtype=torch.uint8, device=self.device)

        self.tagger.to_gpu(cuda_device)

    def load_pretraining_embeddings(self, embeddings):
        self.tagger.load_pretraining_embeddings(embeddings)

    def adjust_input(self, input_batch):
        return self.tagger.adjust_input(input_batch)

    def _select_sents(self, input_batch_w, input_batch_ch, mask,
                      target_span_tags, target_type_tags, NE_info):
        h = self.tagger.forward(input_batch_w, input_batch_ch, mask)
        h = h.detach()  # タガーのパラメータはRLでは更新しない

        RL_state = torch.zeros((len(input_batch_w), self.SL_in_dim),
                               dtype=torch.float32, device=self.device)

        attns = self.space_attention(h)
        sent_start_index = 0
        sent_end_index = 0
        for i, span_tags in enumerate(target_span_tags):
            sent_end_index += len(span_tags) + 1
            sent_attns = torch.softmax(
                attns[sent_start_index:sent_end_index], dim=0)
            sent_attns = sent_attns.expand(-1, self.SL_in_dim)
            RL_state[i] = torch.sum(
                h[sent_start_index:sent_end_index] * sent_attns, dim=0)
            sent_start_index = sent_end_index

        # NE's vector from cnn layer
        NE_list = NE_info[0]
        contain_NEs_in_sent_list = NE_info[1]
        try:
            max_NE_len = max([len(_) for NEs in NE_list for _ in NEs])
        except ValueError:
            max_NE_len = 0
        if max_NE_len != 0:
            NE_input_batch = []
            for NEs in NE_list:
                for NE in NEs:
                    pad_size = max_NE_len - len(NE)
                    NE_idx = NE + [PAD_id] * pad_size
                    NE_input_batch.append(NE_idx)
            NE_input_batch = torch.tensor(NE_input_batch, dtype=torch.long,
                                          device=self.device)
            NE_embs = self.tagger.w_embed(NE_input_batch)
            NE_embs = NE_embs.transpose(2, 1).contiguous()
            NE_cnn_out = self.NE_cnn(NE_embs)
            NE_cnn_out = F.max_pool1d(NE_cnn_out, NE_cnn_out.size(2))
            NE_cnn_out = NE_cnn_out.squeeze()

        NE_state = torch.zeros((RL_state.size(0), self.NE_cnn_dim),
                               device=self.device)
        idx = 0
        for b, contain_NEs_in_sent in enumerate(contain_NEs_in_sent_list):
            if contain_NEs_in_sent != 0:
                end = idx + contain_NEs_in_sent
                NE_cnn_state = torch.mean(NE_cnn_out[idx:end], dim=0)
                NE_state[b] = NE_cnn_state
                idx += contain_NEs_in_sent
        RL_state = torch.cat([RL_state, NE_state], dim=1)
        RL_action = self.fc_SL2(self.fc_SL1(RL_state))
        RL_action = torch.sigmoid(RL_action)

        size = RL_action.size()

        if self.all_select:
            action = self.one.expand(size[0], size[1])
        else:
            rand = torch.rand((1,)).item()
            rands = torch.randn((size[0], 1), device=self.device) + 0.5
            if self.epsilon < rand:
                action = torch.where(RL_action >= rands,
                                     self.one.expand(size[0], size[1]),
                                     self.zero.expand(size[0], size[1]))
            else:
                action = torch.where(RL_action < rands,
                                     self.one.expand(size[0], size[1]),
                                     self.zero.expand(size[0], size[1]))

        A = (action.float().detach() * RL_action) + \
            ((1.0 - action.float().detach()) * (1.0 - RL_action))

        sent_len = input_batch_ch.size()[1]
        selected_input_ch = input_batch_ch.masked_select(
            action.expand(-1, sent_len)).view(-1, sent_len)
        selected_input_w = input_batch_w.masked_select(
            action.expand(-1, sent_len)).view(-1, sent_len)
        selected_mask = mask.masked_select(
            action.expand(-1, sent_len)).view(-1, sent_len)
        selected_target_span_tags = []
        selected_target_type_tags = []
        for i, a in enumerate(action):
            if a == 1:
                selected_target_span_tags.append(target_span_tags[i])
                selected_target_type_tags.append(target_type_tags[i])
            else:
                pass
        selected_input = (selected_input_w, selected_input_ch,
                          selected_mask, selected_target_span_tags,
                          selected_target_type_tags)

        return selected_input, A

    def calc_tagger_loss(self, input_batch, target_span_tags,
                         target_type_tags, NE_info):
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

        selected_input, A = \
            self._select_sents(input_batch_w, input_batch_ch, mask,
                               target_span_tags, target_type_tags, NE_info)
        if len(selected_input[0]) == 0:
            loss = None
        else:
            loss = self.tagger.calc_tagger_loss(*(selected_input))

        select_num = len(selected_input[0])

        return loss, A, select_num

    def calc_RL_loss(self, action, valid_texts, valid_NEs):
        return

    def predict(self, input_batch):
        return self.tagger.predict(input_batch)

    def calc_prob_given_span_and_type(self, input_batch, target_span_tags,
                                      target_type_tags):
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

        reward = self.tagger.calc_prob_given_span_and_type(input_batch_w,
                                                           input_batch_ch,
                                                           mask,
                                                           target_span_tags,
                                                           target_type_tags)
        reward = reward.mean()
        reward = reward.detach()
        reward = reward.item()
        return reward
