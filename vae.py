from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import scipy
import tensorflow as tf
import numpy as np
import codecs
import csv
import os

EMBEDDING_PATH = "data/glove.6B.300d.txt"
TRAIN_PATH = "data/train.csv"
EMBED_DIM = 300
MAX_SEQ_LEN = 25
MAX_WORDS = 30000


# loading in sentences from train file
texts = []
with codecs.open(TRAIN_PATH, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        if len(values[3].split()) <= MAX_SEQ_LEN:
            texts.append(values[3])
        if len(values[4].split()) <= MAX_SEQ_LEN:
            texts.append(values[4])
print('Found %s texts in train.csv' % len(texts))
n_sents = len(texts)

# tokenize
tokenizer = Tokenizer(MAX_WORDS+1, oov_token='unk')
tokenizer.fit_on_texts(texts)
print('Found %s unique tokens' % len(tokenizer.word_index))
tokenizer.word_index = {
    e: i for e, i in tokenizer.word_index.items() if i <= MAX_WORDS
}
tokenizer.word_index[tokenizer.oov_token] = MAX_WORDS + 1
word_index = tokenizer.word_index
index2word = {v: k for k, v in word_index.items()}
sequences = tokenizer.texts_to_sequences(texts)
input_data = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)
validation_data = input_data[801000:807000]
print('Shape of input tensor:', input_data.shape)
print(tokenizer.num_words, len(word_index))
NB_WORDS = (min(tokenizer.num_words, len(word_index))+1)
