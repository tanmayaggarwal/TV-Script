# File: check access to GPU
import torch

# check access to GPU
def check_gpu():
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('No GPU found. Please use a GPU to train the neural network.')
    return train_on_gpu


