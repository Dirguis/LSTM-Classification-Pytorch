import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.drop = nn.Dropout(0.2)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence, input_lengths):
        batch_vector = torch.LongTensor(np.arange(self.batch_size))
        embeds = self.word_embeddings(sentence)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embeds, input_lengths)
        lstm_out, self.hidden = self.lstm(packed_input, self.hidden)
        outp, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        x = outp[input_lengths - 1, batch_vector, :]
        x = F.relu(x)
        x = self.drop(x)
        y = self.hidden2label(x)
        return F.log_softmax(y, dim=-1)
