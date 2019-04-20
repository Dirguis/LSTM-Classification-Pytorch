import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu, attn_flag=True):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
        self.attn_flag = attn_flag

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state).squeeze(2)
        msk = (weights == 0)
        weights = weights.masked_fill(msk, torch.FloatTensor(np.array(-np.inf)))
        weights = torch.nn.functional.softmax(weights, dim=-1).unsqueeze(-1)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        attn_out = torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)
        return attn_out

    def forward(self, sentence, input_lengths):
        embeds = self.word_embeddings(sentence)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embeds, input_lengths, batch_first=True)
        lstm_out, self.hidden = self.lstm(packed_input, self.hidden)
        outp, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        if self.attn_flag:
            h_n, c_n = self.hidden
            outp = self.attention(outp, h_n)
        else:
            batch_vector = torch.LongTensor(np.arange(self.batch_size))
            outp = outp[batch_vector, input_lengths - 1, :]
        x = F.relu(outp)
        x = self.drop(x)
        y = self.hidden2label(x)
        return F.log_softmax(y, dim=-1)
