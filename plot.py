import time
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization, TimeDistributed
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

"""
Note: This program is made to test the performance of the trained model
"""

X_test = np.load("X_test_normalized.npy")
Y_test = np.load("Y_test_normalized.npy")
X_train = np.load("X_train_normalized.npy")
Y_train = np.load("Y_train_normalized.npy")
#NAME = "Vanilla_normalized_leaky"
NAME = "Vanilla_normalized"
checkpoint_path = "checkpoints/{}/cp.h5".format(NAME)
print("Y test shape: ",Y_test.shape)
model = Sequential()
model.add(TimeDistributed(Dense(440, activation=tf.nn.leaky_relu)))
for _ in range(3):
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(88, return_sequences=True)))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(320, activation=tf.nn.leaky_relu)))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(90, activation=tf.nn.leaky_relu)))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(30, activation=tf.nn.leaky_relu)))
model.add(TimeDistributed(Dense(3, activation='linear'))) # output layer

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.MeanSquaredError(), metrics = ["accuracy"])

#model.load_weights(checkpoint_path)
#model=keras.models.load_model(checkpoint_path, custom_objects={'leaky_relu': tf.nn.leaky_relu})
model=keras.models.load_model(checkpoint_path)

fig = plt.figure()
Yhat = model.predict(X_test)
print("Yhat test shape: ", Yhat.shape)
_, i_size, j_size =  Y_test.shape
#bins = np.linspace(-1.5, 1.5, 150)
bins = np.linspace(0.0, 1.0, 300)
for i in range(i_size):
    for j in range(j_size):
        
        plt.hist(Y_test[:,i,j], bins, alpha=0.5, label = "Y_test", figure = fig)
        plt.hist(Yhat[:,i,j], bins, alpha=0.5, label =" Yhat", figure = fig )
        plt.legend(loc='upper right')
        plt.title(f'Test_plot_on{i}and{j}', figure = fig) 
        fig.savefig(f'Test_plot_on{i}and{j}') 
        fig.clf()
        print("done one")
plt.close(fig)
print("Yhat train shape: ", Yhat.shape)
fig = plt.figure()
#now plot the with the train values
Yhat = model.predict(X_train[:167175,:,:])

for i in range(i_size):
    for j in range(j_size):
        #plt.legend(['Yhat'])
        plt.hist(Y_train[:167175,i,j], bins, alpha=0.5, label = "Y_train", figure = fig)
        #plt.legend(['Y test'])
        plt.hist(Yhat[:,i,j], bins, alpha=0.5, label =" Yhat", figure = fig)
        plt.legend(loc='upper right')
        plt.title(f'Train_plot_on{i}and{j}', figure = fig)
        fig.savefig(f'Train_plot_on{i}and{j}')
        fig.clf()
        print("done one")

