import time
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization, TimeDistributed
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

X_train = np.load("X_train_normalized.npy")
X_test = np.load("X_test_normalized.npy")
Y_train = np.load("Y_train_normalized.npy")
Y_test = np.load("Y_test_normalized.npy")

# X_train = np.load("X_train.npy")
# X_test = np.load("X_test.npy")
# Y_train = np.load("Y_train.npy")
# Y_test = np.load("Y_test.npy")

#NAME = "Vanilla_normalized_leaky"
NAME = "Vanilla_normalized"
# NAME = "NodePerLayer-{},Bidirectional-{},LSTM-{},Dense-{},Time-{}".format(nodes_per_layer,bidirectional,LSTM_layer,dense_layer, int(time.time()))
print(NAME)

tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME)) #'logs/{}''
#checkpoint_path = "checkpoints/{}/cp.ckpt".format(NAME)
checkpoint_path = "checkpoints/{}/cp.h5".format(NAME)
w_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_loss', verbose=1) # monitor vallidation loss to save if it's best it's seen

model = Sequential()
model.add(TimeDistributed(Dense(440, activation=tf.nn.relu)))
for _ in range(3):
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(88, return_sequences=True)))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(320, activation=tf.nn.relu)))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(90, activation=tf.nn.relu)))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(30, activation=tf.nn.relu)))
model.add(TimeDistributed(Dense(3, activation='linear'))) # output layer

model.compile(optimizer=tf.keras.optimizers.Adam(lr=5e-4), loss=tf.keras.losses.MeanSquaredError(), metrics = ["accuracy"])
#model.load_weights(checkpoint_path)
model=keras.models.load_model(checkpoint_path)
history = model.fit(X_train, Y_train, epochs= 35, validation_data= (X_test,Y_test), callbacks = [ tensorboard])
model.save(checkpoint_path)
#print("History keys: ", history.history.keys())
fig = plt.figure()
plt.plot(history.history['loss'], label="overall training loss")
plt.plot(history.history['val_loss'], label="overall validation loss")
plt.title('Training vs Validation Losses')
plt.legend(loc='upper left')
fig.savefig('Training vs Validation Losses')

