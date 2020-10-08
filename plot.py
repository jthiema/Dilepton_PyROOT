import time
import numpy as np
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
NAME = "Vanilla_normalized_leaky"
checkpoint_path = "checkpoints/{}/cp.ckpt".format(NAME)
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

model.load_weights(checkpoint_path)
Yhat = model(X_test)
_, i_size, j_size =  Y_test.shape
for i in range(i_size):
    for j in range(j_size):
        
        #plt.legend(['Yhat'])
        plt.hist(Y_test[:,i,j], label = "Y_test")
        #plt.legend(['Y test'])
        plt.hist(Yhat[:,i,j], label =" Yhat")
        plt.legend(loc='upper right')
        plt.title(f'Test_plot_on{i}and{j}') 
        plt.savefig(f'Test_plot_on{i}and{j}') 
        plt.clf()
        print("done one")
#now plot the with the train values
Yhat = model(X_train)
for i in range(i_size):
    for j in range(j_size):
        #plt.legend(['Yhat'])
        plt.hist(Y_train[:,i,j], label = "Y_test")
        #plt.legend(['Y test'])
        plt.hist(Yhat[:,i,j], label =" Yhat")
        plt.legend(loc='upper right')
        plt.title(f'Train_plot_on{i}and{j}')
        plt.savefig(f'Train_plot_on{i}and{j}')
        plt.clf()
        print("done one")

