import time
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization, TimeDistributed, LeakyReLU
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


X_train = np.load("X_train_normalized.npy")
X_test = np.load("X_test_normalized.npy")
Y_train = np.load("Y_train_normalized.npy")
Y_test = np.load("Y_test_normalized.npy")

# X_train = np.load("X_train.npy")
# X_test = np.load("X_test.npy")
# Y_train = np.load("Y_train.npy")
# Y_test = np.load("Y_test.npy")

NAME = "Vanilla_unnormalized_leaky"
# NAME = "Vanilla_normalized"
# NAME = "NodePerLayer-{},Bidirectional-{},LSTM-{},Dense-{},Time-{}".format(nodes_per_layer,bidirectional,LSTM_layer,dense_layer, int(time.time()))
print(NAME)

tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME)) #'logs/{}''
checkpoint_path = "checkpoints/{}.ckpt".format(NAME)
w_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_loss', verbose=1) # monitor vallidation loss to save if it's best it's seen

model = Sequential()
model.add(TimeDistributed(Dense(440, activation=LeakyReLU)))
for _ in range(3):
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(88, return_sequences=True)))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(320, activation=LeakyReLU)))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(90, activation=LeakyReLU)))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(30, activation=LeakyReLU)))
model.add(TimeDistributed(Dense(3, activation='linear'))) # output layer

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics = ["accuracy"])
history = model.fit(X_train, Y_train, epochs= 15, validation_data= (X_test,Y_test), callbacks = [w_callback, tensorboard])

