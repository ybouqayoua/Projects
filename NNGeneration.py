# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:49:10 2019

@author: ybouqayoua
"""
from keras.models import Sequential, model_from_json, load_model, model_from_json
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import time
from keras import callbacks
import keras
import os
import warnings
from rbflayer import RBFLayer, InitCentersRandom
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae



    
def generate_nn(Gen, X_train, Y_train, X_test, Y_test, layers, neurons, lr, inputSize, activation, optimizer, batch_size, pt):
	#with tf.device('GPU:0'):
	
	    model = Sequential()
	    
	    # set model layers
	    if activation == 'rbf':
		for i in range(layers):
		    ##### RBF Layer zum testen. Theoretisch gute Ergebnisse für Regression #####
		    if i==0:
		        rbflayer = RBFLayer(neurons,
		                              initializer=InitCentersRandom(X_train),
		                              betas=2.0,
		                              input_shape=(inputSize,))
		    else:
		         rbflayer = RBFLayer(neurons,
		                              betas=2.0,
		                              input_shape=(neurons,))
		    model.add(rbflayer)
		    ###########################################################################
	    else: 
		for i in range(layers):
		    if i==0:
		        layer = Dense(neurons, input_dim=inputSize, activation=activation)
		    else:
		         layer = Dense(neurons, activation=activation)
		         
		    model.add(layer)

	    # output Layer
	    model.add(Dense(1))
	    
	    
	    if optimizer=='rmsprop':
		opt = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
	    elif optimizer == 'adam':
		opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)  
	    elif optimizer == 'adagrad':
		opt = keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=0.0)
	    elif optimizer == 'adadelta':
		opt = keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
	    elif optimizer == 'adamax':
		opt = keras.optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
	    elif optimizer == 'nadam':
		opt = keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
	    
	    
	    path = pt + str(Gen) + "/" + str(Gen) + "_" + "_" + str(layers) + "_" + str(neurons) + "_" + str(lr) + "_" + activation + "_" + optimizer + "_" + str(batch_size)
	    model.compile(loss='mae', optimizer=opt, metrics=['mae'])
	    
	    #EarlyStopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
	    EarlyStopping = EarlyStoppingAtMinLoss(patience = 30, relative_tol = 0.001, monitor='val_mean_absolute_error')
	    mcp_save = ModelCheckpointYB(path + '.hdf5', save_best_only=True, monitor='val_mean_absolute_error', mode='min', Threshold = 70000)
	    T_on_NaN = callbacks.TerminateOnNaN() 
	    model.fit(X_train, Y_train, epochs=2000, batch_size=batch_size, callbacks = [EarlyStopping, mcp_save, T_on_NaN],validation_data=[X_test,Y_test],shuffle=True, verbose = 0)
	    
	    
    

	    try:
       		if os.path.exists(path + '.hdf5'):
            		print("Hallo")
            		model = load_model(path + '.hdf5')
            		MSE = mse(model.predict(X_test),Y_test)
            		MAE = mae(model.predict(X_test),Y_test)
           	else:
            		MSE = mse(model.predict(X_test),Y_test)
            		MAE = mae(model.predict(X_test),Y_test)
    	    except:
        	MSE = MAE = 1e+20
        	with open("C:\Models\EXCEPTION" + str(time.time()) + ".txt", 'w') as error_file:
                	        error_file.write(str(layers) + "_" + str(neurons) 
                        	    + "_" + str(round(lr,6)) + "_" + activation + "_" + optimizer + "_" 
                            		+ str(batch_size))



#    if os.path.exists(path + '.hdf5'):
#        print("Hallo")
#        model = load_model(path + '.hdf5')
#        MSE = mse(model.predict(X_test),Y_test)
#        MAE = mae(model.predict(X_test),Y_test)
#    elif isinstance(model.loss(), float):
#        MSE = mse(model.predict(X_test),Y_test)
#        MAE = mae(model.predict(X_test),Y_test)
#    else: 
#        MSE = 1e+20
#        MAE = 1e+20
#    
    # serialize model to JSON lo
#    model_json = model.to_json()
#    with open(path + ".json", "w") as json_file:
#        json_file.write(model_json)
#    # serialize weights to HDF5
            return MSE, MAE, path
    


def load_models(Population, num_mods, X_Val, Y_Val):
    models = []
    predictions = []
    errors = []
    for i in range(num_mods):
        with open(Population[i][6] + ".json",'r') as f:
            model = model_from_json(f.read())
    
        model.load_weights(Population[i][6] + ".h5")
        predictions.append(model.predict(X_Val))
        errors.append([mse(model.predict(X_Val), Y_Val), mae(model.predict(X_Val), Y_Val)])
        models.append(model)

    return [models, predictions, errors]





class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
  """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
      
      relative_tol: Relative Tolerance, which we want to break 
  """

  def __init__(self, patience=0, relative_tol = 0, monitor = 'val_loss'):
    super(EarlyStoppingAtMinLoss, self).__init__()

    self.patience = patience
    self.relative_tol = relative_tol
    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None
    self.monitor = monitor
  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get(self.monitor)
    if isinstance(current, float):
        if self.best/current >= 1+self.relative_tol:
          self.best = current
          self.wait = 0
          # Record the best weights if current results is better (less).
          self.best_weights = self.model.get_weights()
        else:
          self.wait += 1
          if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
      
      
class ModelCheckpointYB(tf.keras.callbacks.Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_mean_absolute_error', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, Threshold = 100000):
        super(ModelCheckpointYB, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.Threshold = Threshold
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.best <= self.Threshold:
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
