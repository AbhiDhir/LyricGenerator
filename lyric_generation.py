from keras.models import Sequential
from keras.layers import LSTM, Activation, Flatten, Dropout, Dense, Embedding, TimeDistributed
import numpy as np
import pickle
import sys

(int_chars, n_vocab, data_X, X, y) = pickle.load(open('lyric_generation_data', 'rb'))
LSTM_layer_num = 4
layer_size = [256, 256, 256, 256]

model = Sequential()
model.add(LSTM(layer_size[0], input_shape=(X.shape[1],X.shape[2]), return_sequences=True))
for i in range(1, LSTM_layer_num):
    model.add(LSTM(layer_size[i], return_sequences=True))
model.add(Flatten())
model.add(Dense(y.shape[1]))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

weights_file = './Weights-LSTM-improvement-030-0.07350-bigger.hdf5'
model.load_weights(weights_file)
model.compile(loss='categorical_crossentropy', optimizer='adam')

start = np.random.randint(0, len(data_X)-1)
pattern = data_X[start]
print('Seed : ')
print("\"", ''.join([int_chars[value] for value in pattern]), "\"\n")

generated_characters = 3000

for i in range(generated_characters):
    x = np.reshape(pattern, (1,len(pattern),1))
    x = x / float(n_vocab)
    prediction = model.predict(x,verbose=0)
    index = np.argmax(prediction)
    result = int_chars[index]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print('\nDone')