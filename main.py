import os
import torch
import copy
import utils.DataProcessing as DP
import utils.LSTMClassifier as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import PlotFigure as PF
from utils.DataProcessing import DataLoader
import pickle
from datetime import datetime

use_plot = True
use_save = True

DATA_DIR = 'data'
TRAIN_DIR = 'train_txt'
TEST_DIR = 'test_txt'
TRAIN_FILE = 'train_txt.txt'
TEST_FILE = 'test_txt.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'

# parameter setting
epochs = 20
batch_size = 16
attn_flag = True
use_gpu = torch.cuda.is_available()
learning_rate = 0.005
torch.manual_seed(666)


if __name__ == '__main__':
    # parameter setting
    embedding_dim = 50
    hidden_dim = 150
    sentence_len = None
    train_file = os.path.join(DATA_DIR, TRAIN_FILE)
    test_file = os.path.join(DATA_DIR, TEST_FILE)
    fp_train = open(train_file, 'r')
    train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in fp_train]
    filenames = copy.deepcopy(train_filenames)
    fp_train.close()
    fp_test = open(test_file, 'r')
    test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in fp_test]
    fp_test.close()
    filenames.extend(test_filenames)

    corpus = DP.Corpus(DATA_DIR, filenames)
    nlabel = 8

    # create model
    model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(corpus.dictionary),
                                 label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu, attn_flag=attn_flag)
    if use_gpu:
        model = model.cuda()
    # data processing
    dtrain_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)

    train_loader = DataLoader(dtrain_set, batch_size=batch_size, shuffle=True)
    dtest_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus)

    test_loader = DataLoader(dtest_set, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2)

    loss_function = nn.NLLLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    # training procedure
    for epoch in range(epochs):

        model.train()
        # training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_loader):
            train_inputs, train_labels, train_lengths = traindata

            if use_gpu:
                train_inputs, train_labels = Variable(train_inputs.cuda()), train_labels.cuda()
            else:
                train_inputs = Variable(train_inputs)

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs, train_lengths)

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item()

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        # testing epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        model.eval()
        for iter, testdata in enumerate(test_loader):
            test_inputs, test_labels, test_lengths = testdata

            if use_gpu:
                test_inputs, test_labels = Variable(test_inputs.cuda()), test_labels.cuda()
            else:
                test_inputs = Variable(test_inputs)

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test_inputs, test_lengths)

            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == test_labels).sum()
            total += len(test_labels)
            total_loss += loss.item()
        test_loss_.append(total_loss / total)
        test_acc_.append(total_acc.item() / total)
        scheduler.step(test_loss_[-1])

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch + 1, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))

    param = {}
    param['lr'] = learning_rate
    param['batch size'] = batch_size
    param['embedding dim'] = embedding_dim
    param['hidden dim'] = hidden_dim
    param['sentence len'] = sentence_len

    result = {}
    result['train loss'] = train_loss_
    result['test loss'] = test_loss_
    result['train acc'] = train_acc_
    result['test acc'] = test_acc_
    result['param'] = param

    if use_plot:
        PF.PlotFigure(result, use_save)
    if use_save:
        if not os.path.exists('log'):
            os.mkdir('log')
        filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
        result['filename'] = filename

        fp = open(filename, 'wb')
        pickle.dump(result, fp)
        fp.close()
        print('File %s is saved.' % filename)
