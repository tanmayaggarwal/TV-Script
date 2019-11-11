# File: define the model

import numpy as np
import torch
import problem_unittests as tests
import helper
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        # initialize the PyTorch RNN module
        # param vocab_size: the number of input dimensions of the neural network (the size of the vocabulary)
        # param out_size: the number of output dimensions of the neural network
        # param embedding_dim: the size of embeddings
        # param hidden_dim: the size of the hidden layer outputs
        # param dropout: dropout to add in between LSTM layers
        super(RNN, self).__init__()
        # set class variables
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = dropout

        # define model layers
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, nn_input, hidden):
        # forward propagation function
        # param nn_input: the input to the neural network
        # param hidden: the hidden state
        # return: two tensors, output of the neural network and the latest hidden state
        batch_size = nn_input.size(0)
        x = self.embed(nn_input)
        lstm_output, hidden = self.lstm(x, hidden)
        out = lstm_output.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(out)
        out = self.fc(out)

        # reshape to be batch_size first
        output = out.view(batch_size, -1, self.output_size)
        output_words = output[:, -1] # get last batch of word scores

        return output_words, hidden

    def init_hidden(self, batch_size, train_on_gpu):
    # initialization function for the LSTM hidden state
    # param batch_size: the batch_size of the hidden state
    # return: hidden state of dimensions (n_layers, batch_size, hidden_dim)

        weight = next(self.parameters()).data

        if(train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
