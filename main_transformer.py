
import time
from transformer import train_loop
# from transformer import *
import torch
from torch.nn import Transformer
from transformer import evaluate_transformer
import numpy as np
import matplotlib.pyplot as plt
"""
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
"""

X_train = np.load("X_train_normalized.npy")
X_test = np.load("X_test_normalized.npy")
Y_train = np.load("Y_train_normalized.npy")
Y_test = np.load("Y_test_normalized.npy")

print(X_train.shape)
print(Y_train.shape)
# data_iter = data_gen


# def data_gen(V, batch, nbatches):
#     "Generate random data for a src-tgt copy task."
#     for i in range(nbatches):
#         data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
#         # print("data shape: ", data.shape)
#         data[:, 0] = 1
#         src = Variable(data, requires_grad=False)
#         # print("src: ", src)
#         tgt = Variable(data, requires_grad=False)
#         yield Batch(src, tgt, 0)


# for i, batch in enumerate(data_iter):
#   print("i: ", i)
#   print("batch src shape: ", batch.src)
#   print("batch src_mask : ", batch.src_mask.shape)
#   print("batch trg_mask: ", batch.trg_mask.shape)
#   print("batch trg shape: ", batch.trg)
# # Train the simple copy task.
# V = 6
# criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
# model = make_model(V, V, N=2)
# model.eval()
# data = torch.from_numpy(np.random.randint(1, 3, size=(12, V, V)))
# model(data)

# model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
#         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# for epoch in range(10):
#     model.train()
#     run_epoch(data_gen(V, 30, 20), model, 
#               SimpleLossCompute(model.generator, criterion, model_opt))

#     model.eval()
#     print(run_epoch(data_gen(V, 30, 5), model, 
#                     SimpleLossCompute(model.generator, criterion, None)))


# print(torch.cuda.get_device_name(0))
N,S,E = X_train.shape
X_train = X_train.reshape(S, N ,E)
N,E,T = Y_train.shape
Y_train = Y_train.reshape(T, N ,E)
N,S,E = X_test.shape
X_test = X_test.reshape(S, N ,E)
N,E,T = Y_test.shape
Y_test = Y_test.reshape(T, N ,E)
print(X_train.shape)
# print((X_train[0,0,0]))
model = Transformer(d_model = 6, nhead = 6)

training_avg_losses, evaluating_avg_losses = train_loop(X_train, X_test, Y_train, Y_test, model, loop_n = 25)
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
