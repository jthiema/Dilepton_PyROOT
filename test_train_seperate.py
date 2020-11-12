import numpy as np 
import pickle
import sklearn
from sklearn.model_selection import train_test_split

X = np.load("X_raw.npy")
Y = np.load("Y_raw.npy")

scaler = sklearn.preprocessing.StandardScaler()
X_shape = X.shape
Y_shape = Y.shape
X = scaler.fit_transform(X.reshape(X_shape[0], np.prod(X_shape[1:])))
Y = scaler.fit_transform(Y.reshape(Y_shape[0], np.prod(Y_shape[1:])))
X = X.reshape(X_shape)
Y = Y.reshape(Y_shape)
#np.random.shuffle(X)
#np.random.shuffle(Y)


data_size = X.shape[0] # length of the data
#ratio = 0.2 # fraction of the data that will work as training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=765 )

'''
X_train = X[:int(data_size*ratio)]
X_test = X[int(data_size*ratio):]
Y_train = Y[:int(data_size*ratio)]
Y_test = Y[int(data_size*ratio):]


np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("Y_train.npy", Y_train)
np.save("Y_test.npy", Y_test)

"""
Now we do normalization. Eta values are normalized to be eta_normalized = (eta +2.5) /5
phi values are normalized to be phi_normalized = (phi + pi)/ (2*pi) = phi /(2*pi) + 0.5
for the rest, you take the maximum value of the category and divide all the values by the maximum value (leptons and jets together, but X and Y seperated)
This refers to Pt, energy and mass
"""
X_pt_max = np.max(X[:,0,:]) # get max pt value
X[:,0,:] = X[:,0,:] / X_pt_max # pt normalization
X[:,1,:] = (X[:,1,:] / (2*np.pi)) + 0.5 # phi normalization
X[:,2,:] = (X[:,2,:] + 2.5) / 5 # eta normalization
X_E_max = np.max(X[:,3,2:]) # get max E value for jets
X[:,3,2:] = X[:,3,2:] / X_E_max # E normalization (first two columns are for leptons)
X_MET_pt_max = np.max(X[:,4,:2]) # get max MET pt value
X[:,4,:2] = X[:,4,:2] / X_MET_pt_max # pt normalization (same as normal pt normalization)
X_M_max = np.max(X[:,4,2:]) # get max M value for jets
X[:,4,2:] = X[:,4,2:] / X_M_max # M normalization (first two columns are for leptons)
X[:,5,:2] = (X[:,5,:2] / (2*np.pi)) + 0.5 # MET phi normalization (same as normal phi normalization)

Y_pt_max = np.max(Y[:,:,0]) # get max pt value
Y[:,:,0] = Y[:,:,0] / Y_pt_max # pt normalization
Y[:,:,1] = (Y[:,:,1] / (2*np.pi)) + 0.5 # phi normalization
Y[:,:,2] = (Y[:,:,2] + 2.5) / 5 # eta normalization


X_train = X[:int(data_size*ratio)]
X_test = X[int(data_size*ratio):]
Y_train = Y[:int(data_size*ratio)]
Y_test = Y[int(data_size*ratio):]
'''
np.save("X_train_normalized.npy", X_train)
np.save("X_test_normalized.npy", X_test)
np.save("Y_train_normalized.npy", Y_train)
np.save("Y_test_normalized.npy", Y_test)

'''
max_values = {"X_pt_max": X_pt_max, "X_E_max": X_E_max, "X_MET_pt_max": X_MET_pt_max, "X_M_max": X_M_max, "Y_pt_max": Y_pt_max}
file = open("normalization_amplitude.pkl", "wb") #save value dictionary to pkl files
pickle.dump(max_values, file)
file.close()
'''
