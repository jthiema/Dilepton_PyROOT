import time
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization, TimeDistributed
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
Y_train = np.load("Y_train.npy")
Y_test = np.load("Y_test.npy")

NAME = "Vanilla"
# NAME = "NodePerLayer-{},Bidirectional-{},LSTM-{},Dense-{},Time-{}".format(nodes_per_layer,bidirectional,LSTM_layer,dense_layer, int(time.time()))
print(NAME)

tensorboard = TensorBoard(log_dir = 'trash/{}'.format(NAME)) #'logs/{}''
model = Sequential()
model.add(TimeDistributed(Dense(440, activation='relu')))
for i in range(3):
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(88), return_sequences=True)))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(320, activation='relu')))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(90, activation='relu')))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(30, activation='relu')))
model.add(TimeDistributed(Dense(3, activation='linear'))) # output layer

nu_guesser.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics = ["accuracy"])
history = nu_guesser.fit(training_input, training_output, batch_size =997, epochs= 50, validation_data= (x_test,y_test), callbacks = [tensorboard])
# tf.keras.backend.clear_session()
# print("results shape: ", nu_guesser.predict(training_input).shape)
