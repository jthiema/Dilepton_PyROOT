from transformer import process_dataX, process_dataY
import matplotlib.pyplot as plt
import numpy as np


Y_test = np.load("Y_test_normalized.npy")
Y_test = Y_test[:1000,:,:]
N, E, T = Y_test.shape

for t in range(T):
    for e in range(E):
        plt.hist(Y_test[:,e, t], bins, alpha=0.5, label = "Y_test", figure = fig)
        plt.legend(loc='upper right')
        plt.title(f'TransformerTesting{e}and{t}', figure = fig)
        fig.savefig(f'TransformerTesting{e}and{t}')
        fig.clf()
        print("done one")