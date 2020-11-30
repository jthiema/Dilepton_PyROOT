import time
import numpy as np

import matplotlib.pyplot as plt
from transformer import evaluate_transformer, process_dataX, process_dataY
import torch
from torch.nn import Transformer
"""
Note: This program is made to test the performance of the trained model
"""

X_test = np.load("X_test_normalized.npy")
Y_test = np.load("Y_test_normalized.npy")
X_train = np.load("X_train_normalized.npy")
Y_train = np.load("Y_train_normalized.npy")

X_test = process_dataX(X_test)
Y_test = process_dataY(Y_test)

# limit the test sizes as otherwise the memory take is too much
X_test = X_test[:1000,:,:]
Y_test = Y_test[:1000,:,:]
# reshape for transformer
# N,S,E = X_test.shape
# X_test = X_test.reshape(S, N ,E)
# N,E,T = Y_test.shape
# #Y_test = Y_test.reshape(T, N ,E)
# Y_test_input = np.zeros((T, N ,E))
Y_test_input = np.zeros(Y_test.shape)
print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape)

save_path = "./checkpoints/Transformer"

model = Transformer(d_model = 6, nhead = 6)
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
_, Yhat = evaluate_transformer(X_test, Y_test_input, model, optim, save_path = save_path, batch_size = None)
Yhat = Yhat.detach().numpy()
# Yhat = Yhat.detach().numpy().reshape(N, E, T)
print("Yhat shape: ", Yhat.shape)
fig = plt.figure()
bins = np.linspace(0.0, 1.0, 300)
for t in range(T):
    for e in range(E):
        #plt.legend(['Yhat'])
        plt.hist(Y_test[t,:,e], bins, alpha=0.5, label = "Y_test", figure = fig)
        # plt.hist(Y_test[:,e,t], bins, alpha=0.5, label = "Y_test", figure = fig)
        #plt.legend(['Y test'])
        plt.hist(Yhat[t,:,e], bins, alpha=0.5, label =" Yhat", figure = fig)
        # plt.hist(Yhat[:,e,t], bins, alpha=0.5, label = "Yhat", figure = fig)
        plt.legend(loc='upper right')
        plt.title(f'TransformerTest_plot_on{i}and{j}', figure = fig)
        fig.savefig(f'TransformerTest_plot_on{i}and{j}')
        fig.clf()
        print("done one")

