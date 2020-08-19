import pickle
import numpy as np
from keras.utils import np_utils

textFileName = 'lyricsText.txt'
raw_text = open(textFileName, encoding = 'UTF-8').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
int_chars = dict((i,c) for i, c in enumerate(chars))
chars_int = dict((i,c) for c, i in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

print(f'Total Characters: {n_chars}')
print(f'Total Vocab: {n_vocab}')

seq_len = 100
data_X = []
data_y = []

for i in range(0, n_chars - seq_len, 1):
    seq_in = raw_text[i:i+seq_len]
    seq_out = raw_text[i+seq_len]
    data_X.append([chars_int[char] for char in seq_in])
    data_y.append(chars_int[seq_out])
n_patterns = len(data_X)
print(f'Total Patterns: {n_patterns}')

X = np.reshape(data_X, (n_patterns, seq_len, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(data_y)

pickle.dump((X,y), open('clean_data', 'wb'))
pickle.dump((int_chars, n_vocab, data_X, X, y), open('lyric_generation_data', 'wb'))