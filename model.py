import sys
import pickle
from keras.models import Sequential
from keras.layers import LSTM, Activation, Flatten, Dropout, Dense, Embedding, TimeDistributed
from keras.callbacks import ModelCheckpoint

(X, y) = pickle.load(open('clean_data', 'rb'))

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

print(model.summary())

checkpoint_name = 'Weights-LSTM-improvement-{epoch:03d}-{loss:.5f}-bigger.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model_params = {'epochs': 30,
                'batch_size': 128,
                'callbacks': callbacks_list,
                'verbose': 1,
                'validation_split': 0.2,
                'validation_data': None,
                'shuffle': True,
                'initial_epoch': 0,
                'steps_per_epoch': None,
                'validation_steps': None}

model.fit(X, y,
        epochs=model_params['epochs'],
        batch_size=model_params['batch_size'],
        callbacks=model_params['callbacks'],
        verbose=model_params['verbose'],
        validation_split=model_params['validation_split'],
        validation_data=model_params['validation_data'],
        shuffle=model_params['shuffle'],
        initial_epoch=model_params['initial_epoch'],
        steps_per_epoch=model_params['steps_per_epoch'],
        validation_steps=model_params['validation_steps'])

