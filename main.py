import numpy as np
import torch
import problem_unittests as tests
import helper
import torch.nn as nn

# load the data
from load_data import load_data
data_dir, text = load_data()

# pre-process the data - lookup tables
from pre_process import create_lookup_tables
vocab_to_int, int_to_vocab = create_lookup_tables(text)

# test create_lookup_tables
tests.test_create_lookup_tables(create_lookup_tables)

# test token_lookup
from pre_process import token_lookup
tests.test_tokenize(token_lookup)

# import batch data function
from batch_data import batch_data

# pre-process all the data and save it
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

# checkpoint
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

# check if GPU is available
from check_gpu import check_gpu
train_on_gpu = check_gpu()

# test the data loader
from batch_data import test_loader
test_loader()

# build the neural network
# test the model
from RNN_model import RNN
tests.test_rnn(RNN, train_on_gpu)

# test the forward and back propagation function
from forward_back_prop import forward_back_prop
tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)

# import training function
from train import train_rnn

# set the hyperparamaters
sequence_length = 131        # number of words in a sequence; total words: 892,110: factors are 30, 131, 227
batch_size = 128
train_loader = batch_data(int_text, sequence_length, batch_size)

# set the training parameters
num_epochs = 3
learning_rate = 0.01

# set the model parameters
vocab_size = len(vocab_to_int)
output_size = vocab_size
embedding_dim = 128
hidden_dim = 512
n_layers = 2

# show stats for every n number of batches
show_every_n_batches = 5

# train the model
# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# define the loss and optimization functions for Training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, train_on_gpu, train_loader, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model trained and saved')

# loading the saved model
from checkpoint import checkpoint
trained_rnn = checkpoint()

# generate TV script
from generate import generate
gen_length = 1000       # can be modified to the user's preference
prime_word = 'kramer'   # name for starting the script

pad_word = helper.SPECIAL_WORDS['PADDING']
generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], sequence_length, train_on_gpu, gen_length)

print(generated_script)

# save the script to a text file
f = open("generated_script_1.txt", "w")
f.write(generated_script)
f.close()
