import numpy as np 

X = np.load("X_raw.npy")
Y = np.load("Y_raw.npy")

np.random.shuffle(X)
np.random.shuffle(Y)

data_size = X.shape[0] # length of the data
ratio = 0.8 # fraction of the data that will work as training data

X_train = X[:int(data_size*ratio)]
X_test = X[int(data_size*ratio):]
Y_train = Y[:int(data_size*ratio)]
Y_test = Y[int(data_size*ratio):]

"""
Now we do normalization. Eta values are normalized to be eta_normalized = (eta +2.5) /5
phi values are normalized to be 
"""

np.save("X_train_normalized.npy", X_train)
np.save("X_test_normalized.npy", X_test)
np.save("Y_train_normalized.npy", Y_train)
np.save("Y_test_normalized.npy", Y_test)
