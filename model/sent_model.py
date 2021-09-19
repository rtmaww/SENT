import numpy as np
import os
import torch
import torch.nn as nn
import math

NEAR_0 = 1e-10


class SENT_Model(nn.Module):
    def __init__(self,  options, vocab_file=None):
        super(SENT_Model, self).__init__()
        self.max_sent_len = options.max_len
        self.pos_emb_dim = 50
        self.ner_label_size = options.ner_label_size
        self.ner_emb_dim = 50
        self.vocab_size = options.vocab_size
        self.neg_sample_num = 10
        self.device = options.gpu
        self.n_device = options.n_gpu
        self.batch_size = options.batch_size
        self.label_size = options.label_size

        # initialize word embs
        weight_matrix = torch.from_numpy(np.load(vocab_file))
        self.emb_dim = weight_matrix.size(1)
        if options.random:
            self.word_embs = nn.Embedding(self.vocab_size, self.emb_dim)
        else:
            if self.vocab_size == weight_matrix.size(0)+2:
                unk = torch.randn(1, self.emb_dim) / math.sqrt(self.emb_dim)
                blk = torch.zeros(1, self.emb_dim)
                weight_matrix = torch.cat([weight_matrix, unk, blk], 0)
            self.word_embs = nn.Embedding(self.vocab_size,self.emb_dim,_weight=weight_matrix)

        # initialize position and ner embs
        self.pos1_emb = nn.Embedding(self.max_sent_len*2, self.pos_emb_dim)
        self.pos2_emb = nn.Embedding(self.max_sent_len*2, self.pos_emb_dim)
        self.ner_emb = nn.Embedding(self.ner_label_size, self.ner_emb_dim)

        self.input_dim = self.emb_dim + 2 * self.pos_emb_dim + self.ner_emb_dim
        self.encoder = nn.LSTM(input_size=self.input_dim, batch_first=True, hidden_size=256, bidirectional=True)
        self.decoder = nn.Sequential(nn.Linear(self.encoder.hidden_size*2*2, self.encoder.hidden_size*2), nn.Tanh(),
                                     nn.Linear(self.encoder.hidden_size*2, self.label_size))
        self.drop = nn.Dropout(0.5)

        self.loss_function_pos = nn.CrossEntropyLoss()
        self.loss_function_neg = nn.NLLLoss()


    def forward(self, train_batch, mode='train', negloss=True):

        head_pos, tail_pos, input_ids, input_masks, ori_token_masks, head_masks, tail_masks, ner_labels, labels = train_batch

        labels = labels.view(-1,)

        batch_size, seq_len = input_ids.size(0), input_ids.size(1)

        words = self.word_embs(input_ids)
        pos1 = self.pos1_emb(head_pos)
        pos2 = self.pos2_emb(tail_pos)
        ner = self.ner_emb(ner_labels)
        inputs = torch.cat([words, ner, pos1, pos2], dim=-1)
        inputs = self.drop(inputs)

        input_lens = input_masks.sum(-1)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lens, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.encoder(inputs)
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True, total_length=seq_len)
        hiddens = self.drop(hiddens)

        loss, preds, right, total, probs = self.compute_negloss(hiddens, labels, ori_token_masks, head_masks, tail_masks, negloss)

        label_probs = probs.gather(-1,labels.view(-1,1))  ## (b,)
        pos_right = ((preds == labels) & (labels != 0)).sum()

        return loss, preds, right, total, pos_right, label_probs, probs


    def compute_negloss(self, t, labels, ori_token_masks, head_masks, tail_masks, negloss=True):

        batch_size, seq_len, hidden_dim = t.size()
        head_masks = head_masks.bool().unsqueeze(-1).repeat(1, 1, hidden_dim)
        tail_masks = tail_masks.bool().unsqueeze(-1).repeat(1, 1, hidden_dim)

        ### creating positive sample
        heads_t = (t * head_masks).sum(dim=1, keepdim=True) / head_masks.sum(dim=1, keepdim=True) # (b, 1, h)
        tails_t = (t * tail_masks).sum(dim=1, keepdim=True) / tail_masks.sum(dim=1, keepdim=True) # (b, 1, h)
        pos_sample = torch.cat([heads_t, tails_t], dim=-1)  # (b, 1, 2h)


        logits = self.decoder(pos_sample.view(batch_size, -1))

        loss = 0.
        if negloss:
            sample_num = self.neg_sample_num
            neg_probs =  torch.log(1. - torch.softmax(logits, dim=-1) + NEAR_0)  # (b, labelsize)
            labels_ = labels.view(-1, 1).repeat(1, sample_num).view(batch_size * sample_num, )
            neg_probs = neg_probs.unsqueeze(1).repeat(1,sample_num,1).view(batch_size*sample_num, neg_probs.size(-1))
            neg_label = (labels_
                          + torch.LongTensor(labels_.size()).cuda().random_(1, self.label_size)) % self.label_size

            loss = self.loss_function_neg(neg_probs.view(-1, self.label_size), neg_label.view(-1))

        else:
            loss = self.loss_function_pos(logits.view(-1, logits.size(-1)), labels.view(-1))

        preds = torch.argmax(logits, dim=-1)
        right = (torch.argmax(logits, dim=-1) == labels ).sum()
        total = (labels >= 0).sum()
        probs = torch.softmax(logits, dim=-1)

        return loss, preds, right, total, probs

