import time
import numpy as np

import matplotlib.pyplot as plt
from transformer import evaluate_transformer
import torch
from torch.nn import Transformer
"""
Note: This program is made to test the performance of the trained model
"""

X_test = np.load("X_test_normalized.npy")
Y_test = np.load("Y_test_normalized.npy")
X_train = np.load("X_train_normalized.npy")
Y_train = np.load("Y_train_normalized.npy")

# limit the test sizes as otherwise the memory take is too much
X_test = X_test[:1000,:,:]
Y_test = Y_test[:1000,:,:]
# reshape for transformer
N,S,E = X_test.shape
X_test = X_test.reshape(S, N ,E)
N,E,T = Y_test.shape
Y_test_reshape = Y_test.reshape(T, N ,E)

save_path = "./checkpoints/Transformer"
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
model = Transformer(d_model = 6, nhead = 6)
model.load_state_dict(torch.load(save_path))
model.eval()
_, Yhat = evaluate_transformer(X_test_reshape, Y_test_reshape, model, optim, save_path = save_path)
Yhat = Yhat.numpy().reshape(N,E,T)
fig = plt.figure()
bins = np.linspace(0.0, 1.0, 300)
for i in range(E):
    for j in range(T):
        #plt.legend(['Yhat'])
        plt.hist(Y_test, bins, alpha=0.5, label = "Y_test", figure = fig)
        #plt.legend(['Y test'])
        plt.hist(Yhat[:,i,j], bins, alpha=0.5, label =" Yhat", figure = fig)
        plt.legend(loc='upper right')
        plt.title(f'TrasformerTest_plot_on{i}and{j}', figure = fig)
        fig.savefig(f'TransformerTest_plot_on{i}and{j}')
        fig.clf()
        print("done one")

