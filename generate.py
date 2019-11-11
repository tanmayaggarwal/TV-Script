import numpy as np
import torch
import problem_unittests as tests
import helper
import torch.nn as nn
import torch.nn.functional as F

def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, sequence_length, train_on_gpu, predict_len=100):
    # generates text using the neural network
    # param decoder: PyTorch module that holds the trained neural network
    # param prime_id: word id to start the first prediction
    # param int_to_vocab: dict of word id keys to word values
    # param token_dict: dict of punctuation tokens keys to punctuation values
    # param pad_value: the value used to pad a sequence
    # param predict_len: the length of text to generate
    # return: the generated text

    rnn.eval()

    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]

    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)

        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0), train_on_gpu)

        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)

        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu

        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()

        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())

        # retrieve the word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)

        # the generated word becomes the next 'current sequence' and the cycle continues
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i

    gen_sentences = ' '.join(predicted)

    # replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '""'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)

    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')

    # return all the sentences
    return gen_sentences
