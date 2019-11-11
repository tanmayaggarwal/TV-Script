
# File: input and batch the data
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# batching the data
def batch_data(words, sequence_length, batch_size):
    # batch the data using DataLoader
    # param words: the word ids of the TV script
    # param sequence_length: the sequence length of each batch
    # param batch_size: the size of each batch; the number of sequences in a batch
    # return: DataLoader with batched data

    feature_tensor = np.array(words).reshape((-1, sequence_length))
    target_tensor = np.zeros(len(feature_tensor), dtype=np.int64)
    for i in range(len(feature_tensor)):
        if i == len(feature_tensor)-1:
            target_tensor[i] = feature_tensor[0,0]
        else:
            target_tensor[i] = feature_tensor[i+1,0]

    data = TensorDataset(torch.from_numpy(feature_tensor), torch.from_numpy(target_tensor))
    loader = DataLoader(data, shuffle=True, batch_size=batch_size)
    return loader

# test dataloader
def test_loader():
    test_text = range(50)
    t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

    data_iter = iter(t_loader)
    sample_x, sample_y = data_iter.next()

    print(sample_x.shape)
    print(sample_x)
    print()
    print(sample_y.shape)
    print(sample_y)

    return
