from tensorflow import keras
import matplotlib.pyplot as plt
import sklearn.datasets as sk
import sklearn.model_selection
from keras.utils import to_categorical
# In
moonsdata, moonsclass = sk.make_moons(10000, noise = 0.15)
Train_X, Test_X, Train_Y, Test_Y = sklearn.model_selection.train_test_split(moonsdata, moonsclass)
fig1 = plt.figure(figsize= [10,10])
plt.scatter(Train_X[Train_Y == 0,0], Train_X[Train_Y == 0,1])
plt.scatter(Train_X[Train_Y == 1,0], Train_X[Train_Y == 1,1])
plt.legend(['First Class','Second Class'])
plt.grid()
plt.figure(figsize= [10,10])
plt.scatter(Test_X[Test_Y == 0,0], Test_X[Test_Y == 0,1])
plt.scatter(Test_X[Test_Y == 1,0], Test_X[Test_Y == 1,1])
plt.legend(['First Class','Second Class'])

# In
y_binary = to_categorical(Train_Y)
model = keras.Sequential()
model.add(keras.layers.Input(shape = (2,)))
model.add(keras.layers.Dense(10, activation='relu',kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
    bias_initializer=keras.initializers.Zeros(), kernel_regularizer = keras.regularizers.l2(10**-4), bias_regularizer = keras.regularizers.l1(10**-5)))
keras.layers.Dropout(30, noise_shape=None, seed=None)
model.add(keras.layers.Dense(5, activation='tanh',kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
    bias_initializer=keras.initializers.Zeros(), kernel_regularizer = keras.regularizers.l2(10**-4), bias_regularizer = keras.regularizers.l1(10**-5)))
keras.layers.Dropout(30, noise_shape=None, seed=None)
model.add(keras.layers.Dense(2, activation='sigmoid'))
model.compile(optimizer='sgd', loss=keras.losses.categorical_crossentropy)
# This builds the model for the first time:
model.fit(Train_X, y_binary, batch_size=25, epochs=100)
# In[]:
