# File: define the forward and back propagation

import numpy as np
import torch
import problem_unittests as tests
import helper
import torch.nn as nn
from RNN_model import RNN

def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden, train_on_gpu):
    # forward and backward propagation on the neural network
    # param decoder: PyTorch Module that holds the neural network
    # param decoder_optimizer: PyTorch optimizer for the neural network
    # param criterion: PyTorch loss function
    # param inp: batch of input to the neural network
    # param target: target output for the batch of input
    # return: the loss and the latest hidden state Tensor

    clip = 5 # gradient clipping

    if (train_on_gpu):
        inp, target = inp.cuda(), target.cuda()

    # creating new variables for the hidden state to avoid backprop through the entire training history
    h = tuple([each.data for each in hidden])

    # zero accumated gradients
    rnn.zero_grad()

    # get the output from the model
    output, h = rnn(inp, h)

    # calculate the loss and perform backprop
    loss = criterion(output.squeeze(), target)
    loss.backward()
    # clip gradient to prevent exploding gradient problem
    nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    optimizer.step()

    return loss.item(), h
