This application generates its own TV scripts using RNNs.

The model has been trained using the Seinfeld dataset of scripts from 9 seasons.

The Neural Network generates a new, "fake" TV script, based on patterns it recognizes in this training data.

The Seinfeld dataset is from https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv

The application follows the following steps:
1. Loading / exploring the dataset
2. Implementing pre-processing functions (i.e., lookup tables, tokenizing punctuations)
3. Saving the pre-processed dataset
4. Building the neural network
5. Training the neural network
6. Saving the model
7. Generating TV scripts

Enjoy!
