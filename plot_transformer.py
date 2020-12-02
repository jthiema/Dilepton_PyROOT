import time
import numpy as np

import matplotlib.pyplot as plt
from transformer import evaluate_transformer, process_dataX, process_dataY
import torch
from torch.nn import Transformer
"""
Note: This program is made to test the performance of the trained model
"""


def make_plots(X, Y, model, title, save_path = "./checkpoints/Transformer"):
    X = process_dataX(X)
    Y = process_dataY(Y) 


    T, N, E = Y.shape
    batches = []
    start = 0
    while(start < N):
        end = start + 1000
        if end > N:
            end = N
        batches.append((X[:, start: end,:], Y[:, start: end,:]))
        start = end
    # optim = torch.optim.Adam(model.parameters(), lr=0.001)
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    Yhat = np.zeros((T, N, E ))
    start = 0
    for X_batch, Y_batch in batches:
        _, N_batch, _ = X_batch.shape
        end = start + N_batch
        _, Yhat_batch = evaluate_transformer(X_batch, Y_batch, model, optim, save_path = save_path, batch_size = None)
        Yhat_batch = Yhat_batch.detach().numpy()
        Yhat[:, start:end, :] = Yhat_batch
        start = end
    # Yhat = Yhat.detach().numpy().reshape(N, E, T)
    print("Yhat shape: ", Yhat.shape)
    fig = plt.figure()
    bins = np.linspace(0.0, 1.0, 300)
    for t in range(T):
        for e in range(E):
            #plt.legend(['Yhat'])
            plt.hist(Y[t,:,e], bins, alpha=0.5, label = "Y", figure = fig)
            # plt.hist(Y[:,e,t], bins, alpha=0.5, label = "Y", figure = fig)
            #plt.legend(['Y test'])
            plt.hist(Yhat[t,:,e], bins, alpha=0.5, label =" Yhat", figure = fig)
            # plt.hist(Yhat[:,e,t], bins, alpha=0.5, label = "Yhat", figure = fig)
            plt.legend(loc='upper right')
            plt.title(title + f'_plot_on{e}and{t}', figure = fig)
            fig.savefig(title + f'_plot_on{e}and{t}')
            fig.clf()
            print("done one")


X_test = np.load("X_test_normalized.npy")
Y_test = np.load("Y_test_normalized.npy")
X_train = np.load("X_train_normalized.npy")
Y_train = np.load("Y_train_normalized.npy")


# limit the test sizes as otherwise the memory take is too much
#X_test = X_test[:1000,:,:]
#Y_test = Y_test[:1000,:,:]



# reshape for transformer
# N,S,E = X_test.shape
# X_test = X_test.reshape(S, N ,E)
# N,E,T = Y_test.shape
# #Y_test = Y_test.reshape(T, N ,E)

print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape)

save_path = "./checkpoints/Transformer"
model = Transformer(d_model = 6, nhead = 6)
make_plots(X_test, Y_test, model, "TransformerTest", save_path = save_path)
make_plots(X_train, Y_train, model, "TransformerTrain", save_path = save_path)

