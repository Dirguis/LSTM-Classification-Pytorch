import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import random


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, DATA_DIR, filenames):
        self.dictionary = Dictionary()
        self.data = self.tokenize(DATA_DIR, filenames)

    def tokenize(self, DATA_DIR, filenames):
        for filename in filenames:
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1

        return ids


class TxtDatasetProcessing(Dataset):
    def __init__(self, data_path, txt_path, txt_filename, label_filename, sen_len, corpus):
        self.txt_path = os.path.join(data_path, txt_path)
        # reading txt file from file
        txt_filepath = os.path.join(data_path, txt_filename)
        fp = open(txt_filepath, 'r')
        self.txt_filename = [x.strip() for x in fp]
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels
        self.corpus = corpus
        self.sen_len = sen_len

    def __getitem__(self, index):
        filename = os.path.join(self.txt_path, self.txt_filename[index])
        fp = open(filename, 'r')
        txt = []
        count = 0
        max_len = 0
        for words in fp:
            txt = [self.corpus.dictionary.word2idx[word.strip()]
                   for word in words.split()[:self.sen_len]
                   if word.strip() in self.corpus.dictionary.word2idx]
        return txt, self.label[index]

    def __len__(self):
        return len(self.txt_filename)


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nb_items = len(dataset)
        self.counter = 0

    @staticmethod
    def sort_batches(l):
        txts, labels, input_lengths = zip(*sorted(l, key=lambda x: x[2], reverse=True))
        batch_size = len(labels)
        train_inputs = torch.LongTensor(np.zeros((batch_size, max(input_lengths)), dtype=np.int64))
        inc = 0
        for txt in txts:
            train_inputs[inc, :input_lengths[inc]] = torch.LongTensor(txt)
            inc += 1
        return torch.LongTensor(train_inputs), torch.LongTensor(labels), torch.LongTensor(input_lengths)

    def __iter__(self):
        return self.__next__()

    def __next__(self):
        nb_items = len(self.dataset)
        cnt = 0
        batch = []
        if self.shuffle:
            np.random.seed(666)
            sample_idx = random.sample(range(nb_items), nb_items)
        else:
            sample_idx = range(nb_items)
        for ind in sample_idx:
            txt, label = self.dataset[ind]
            cnt += 1
            self.counter += 1
            batch.append((txt, label, len(txt)))
            if cnt == self.batch_size:
                cnt = 0
                yield self.sort_batches(batch)
                batch = []
        if len(batch) > 0:
            yield self.sort_batches(batch)
