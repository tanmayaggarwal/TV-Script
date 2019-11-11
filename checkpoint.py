import torch
import helper
import problem_unittests as tests

def checkpoint():
    _, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
    trained_rnn = helper.load_model('./save/trained_rnn')
    return trained_rnn
    

    
