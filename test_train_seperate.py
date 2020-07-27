import numpy as np 

X = np.load("X.npy")
Y = np.load("Y.npy")

np.random.shuffle(X)
np.random.shuffle(Y)

data_size = X.shape[0] # length of the data
ratio = 0.8 # fraction of the data that will work as training data

X_train = X[:int(data_size*ratio)]
X_test = X[int(data_size*ratio):]
Y_train = Y[:int(data_size*ratio)]
Y_test = Y[int(data_size*ratio):]

