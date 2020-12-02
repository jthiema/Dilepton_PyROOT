
import time
from transformer import train_loop
# from transformer import *
import torch
from torch.nn import Transformer
from transformer import evaluate_transformer, process_dataX, process_dataY
import numpy as np
import matplotlib.pyplot as plt

X_train = np.load("X_train_normalized.npy")
X_test = np.load("X_test_normalized.npy")
Y_train = np.load("Y_train_normalized.npy")
Y_test = np.load("Y_test_normalized.npy")

print(X_train.shape)
print(Y_train.shape)

# print(torch.cuda.get_device_name(0))
# N,S,E = X_train.shape
# X_train = X_train.reshape(S, N ,E)
# N,E,T = Y_train.shape
# Y_train = Y_train.reshape(T, N ,E)
# N,S,E = X_test.shape
# X_test = X_test.reshape(S, N ,E)
# N,E,T = Y_test.shape
# Y_test = Y_test.reshape(T, N ,E)
# print(X_train.shape)
# print((X_train[0,0,0]))

X_train = process_dataX(X_train)
Y_train = process_dataY(Y_train)
X_test = process_dataX(X_test)
Y_test = process_dataY(Y_test)

model = Transformer(d_model = 6, nhead = 6)

training_avg_losses, evaluating_avg_losses = train_loop(X_train, X_test, Y_train, Y_test, model, loop_n = 50)
print("training_avg_losses: ", training_avg_losses)
print("evaluating_avg_losses: ", evaluating_avg_losses)
#plot
epochs = [ i + 1 for i in range(len(training_avg_losses))]
plt.plot(epochs, training_avg_losses, label = "Training AVG Loss")
plt.legend(loc='upper right')
plt.title(f'Transformer Training Losses') 
plt.savefig(f'Train_Transformer_Losses') 
plt.clf()
plt.plot(epochs, evaluating_avg_losses, label = "Evaluating AVG Loss")
plt.legend(loc='upper right')
plt.title(f'Transformer Evaluating Losses') 
plt.savefig(f'Evaluating_Transformer_Losses') 
print("done")

"""
make a seperate plot.py to plot the output
"""
