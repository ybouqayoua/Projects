#!/usr/bin/env python
# coding: utf8
#from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
import numpy as np
import NNGeneration as nn 
import Genetic_Algorithm as GA
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import scipy
#import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed
import tensorflow as tf
import keras
import os
import multiprocessing

manager = multiprocessing.Manager()
#Global_Population = manager.list()


#config = tf.ConfigProto( device_count = {'GPU':1,  'CPU':8} )
#sess = tf.Session(config=config)
#keras.backend.set_session(sess)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU:0"


## load data
#boston_dataset = load_boston()
#
## use PCA
##pca = PCA(copy=True, iterated_power='auto',  random_state=None,
##  svd_solver='full', tol=0.0, whiten=True)
#
#x = boston_dataset['data']
#y = boston_dataset['target']
#y2 = rand.randint(0,10) * x[:,rand.randint(0,12)]+ rand.randint(0,10) * x[:, rand.randint(0,12)]+ rand.randint(0,10) * x[:, rand.randint(0,12)] 
#+ rand.randint(0,10) * x[:, rand.randint(0,12)]+ rand.randint(0,10) * x[:, rand.randint(0,12)]
#y2 = y2/100
#y_final = np.transpose(np.vstack((y, y2)))

#X_train, X_test, Y_train, Y_test = train_test_split(x, y_final, test_size = 0.2)

Data = pd.read_csv(r"/home/centos/fabian_files/ArbeitArbeit/DiffEvoData/out_FLV.csv", encoding = "utf8", header=None)

x = Data.iloc[:,1:-1]
y = []
for i in Data.iloc[:,-1]:
    y.append(float(i.replace(",",".")))

X_train, X_test, Y_train, Y_test = np.array(train_test_split(x, y, test_size = 0.2))
X_train = np.array(X_train)
X_test = np.array(X_test)

layers = [1, 2] 
#neurons = [32, 64, 128, 256]
#neurons = np.array(range(3, 26)) * 10
neurons = [16, 32, 64, 128]
#LR = [0.001, 0.01, 0.05, 0.1]
#LR = np.array(range(1,21))
#LR = np.interp(LR, (LR.min(), LR.max()), (0.005, 0.1))
LR = [0.01, 0.05, 0.1, 0.3]

#activations =  ['relu', 'elu', 'selu', 'softmax', 'softplus']     # ['relu', 'elu', 'tanh', 'sigmoid', 'selu', 'softmax', 'softplus']
#optimizers = ['rmsprop', 'adam', 'adagrad', 'adadelta', 'adamax', 'nadam']
activations = ['relu', 'selu']
optimizers = ['adam','adamax']
#batch_sizes = list(range(1,11))
#batch_sizes = [x*50 for x in batch_sizes]
batch_sizes = [50,100,150,200]
ModPerGen = 50
islands = 8
Population = []

all_combinations = list(itertools.product(layers, activations, optimizers, batch_sizes))

######### analysis part #####################
prob_plot = scipy.stats.probplot(y)
Kurtosis = scipy.stats.kurtosis(y)
Skewness = scipy.stats.skew(y)
#Distribution_plot = plt.hist(y)


print("alles laeuft")
from keras import backend as K
K.tensorflow_backend._get_available_gpus()


Parallel(n_jobs = 8)(delayed(GA.GA)(X_train, Y_train, X_test, Y_test, 6, layers, neurons, LR,
                    activations, optimizers, batch_sizes, ModPerGen, 0.02, 0.08, 0.9, island, []) for island in range(islands))



#    pop, Global_Population = GA.GA(X_train, Y_train, X_test, Y_test, 6, layers, neurons, LR, 
 #                   activations, optimizers, batch_sizes, ModPerGen, 0.1, 0.5, 0.4, island,list(Global_Population))
  #  Population.append(pop)


#solution = nn.load_models(Population, 5, X_test, Y_test)

#predictions = np.zeros(len(solution[1][0]))
#for i in range(len(solution[1])):
 #   predictions = predictions + solution[1][i][:,0]

#predictions = predictions/len(solution[1])
#mse(predictons, Y_test)

#MSE = nn.generate_nn(X_train, Y_train, X_test, Y_test, 2, 20, 13, 'relu', 'adam', 25)
#
## load json and create model
#json_file = open("C:\\Models\\2_20_relu_adam_25" + '.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("C:\\Models\\2_20_relu_adam_25" + ".h5")
#


#MSE2 = mse(loaded_model.predict(X_test),Y_test)
































