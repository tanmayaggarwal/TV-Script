# File: Loading / exploring the dataset

import helper
import numpy as np

def load_data():
    # loading the dataset
    data_dir = './data/Seinfeld_Scripts.txt'
    text = helper.load_data(data_dir)

    # exploring the dataset
    view_line_range = (0, 10)

    print('Dataset Stats')
    print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
    lines = text.split('\n')
    print('Number of lines: {}'.format(len(lines)))
    word_count_line = [len(line.split()) for line in lines]
    print('Average number of words in each line: {}'.format(np.average(word_count_line)))

    print()
    print('The lines {} to {}:'.format(*view_line_range))
    print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))

    return data_dir, text

