import os
import sys
sys.path.append(os.path.abspath("../"))
import env_settings as env
import GPy
import GPyOpt
import numpy as np
import pandas as pds
import random
from keras.layers import Activation, Dropout, BatchNormalization, Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
import common_util

# MNIST class
class MNIST():
    def __init__(self, first_input=784, last_output=10,
                 l1_out=512, 
                 l2_out=512, 
                 l1_drop=0.2, 
                 l2_drop=0.2, 
                 batch_size=100, 
                 epochs=10, 
                 validation_split=0.1):
        self.__first_input = first_input
        self.__last_output = last_output
        self.l1_out = l1_out
        self.l2_out = l2_out
        self.l1_drop = l1_drop
        self.l2_drop = l2_drop
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.mnist_data()
        self.__model = self.mnist_model()
        self.model_folder=env.model_file_root_path+"/gpyopt_minist_model_files"
        common_util.mkdirs(self.model_folder)

    # load mnist data from keras dataset
    def mnist_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)
        return X_train, X_test, Y_train, Y_test
    
    # mnist model
    def mnist_model(self):
        model = Sequential()
        model.add(Dense(self.l1_out, input_shape=(self.__first_input,)))
        model.add(Activation('relu'))
        model.add(Dropout(self.l1_drop))
        model.add(Dense(self.l2_out))
        model.add(Activation('relu'))
        model.add(Dropout(self.l2_drop))
        model.add(Dense(self.__last_output))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        return model
    
    # fit mnist model
    def mnist_fit(self):
        early_stopping = EarlyStopping(patience=0, verbose=1)
        self.__model.fit(self.__x_train, self.__y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=0,
                       validation_split=self.validation_split,
                       callbacks=[early_stopping])

    def best_model_train(self):
        early_stopping = EarlyStopping(patience=0, verbose=1)
        model_checker=ModelCheckpoint(filepath=self.model_folder+"/best_model.hdf5",monitor="val_loss",verbose=1,save_best_only=True, mode='auto')
        self.__model.fit(self.__x_train, self.__y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=0,
                       validation_split=self.validation_split,
                       callbacks=[early_stopping,model_checker])

    # evaluate mnist model
    def mnist_evaluate(self):
        self.mnist_fit()
        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=self.batch_size, verbose=0)
        return evaluation

    # function to run mnist class

def run_mnist(first_input=784, last_output=10,
              l1_out=512, l2_out=512, 
              l1_drop=0.2, l2_drop=0.2, 
              batch_size=100, epochs=10, validation_split=0.1):
    
    _mnist = MNIST(first_input=first_input, last_output=last_output,
                   l1_out=l1_out, l2_out=l2_out, 
                   l1_drop=l1_drop, l2_drop=l2_drop, 
                   batch_size=batch_size, epochs=epochs, 
                   validation_split=validation_split)
    mnist_evaluation = _mnist.mnist_evaluate()
    return mnist_evaluation


# function to optimize mnist model
def f(x):
    print("x:",x)
    print("x shape:",x.shape)
    evaluation = run_mnist(
        l1_drop = float(x[:,1]), 
        l2_drop = float(x[:,2]), 
        l1_out = int(x[:,3]),
        l2_out = int(x[:,4]), 
        batch_size = int(x[:,5]), 
        epochs = int(x[:,6]), 
        validation_split = float(x[:,0]))
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]

def gpyopt_opt(func_obj,max_iter_no,domain_name):
    opt_obj=GPyOpt.methods.BayesianOptimization(f=func_obj, domain=domain_name)
    opt_obj.run_optimization(max_iter=max_iter_no)
    print("x_opt:",opt_obj.x_opt)
    return opt_obj

def main():
    # bounds for hyper-parameters in mnist model
    # the bounds dict should be in order of continuous type and then discrete type
    bounds = [{'name': 'validation_split', 'type': 'continuous',  'domain': (0.0, 0.3)},
            {'name': 'l1_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
            {'name': 'l2_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
            {'name': 'l1_out',           'type': 'discrete',    'domain': (64, 128, 256, 512, 1024)},
            {'name': 'l2_out',           'type': 'discrete',    'domain': (64, 128, 256, 512, 1024)},
            {'name': 'batch_size',       'type': 'discrete',    'domain': (10, 100, 500)},
            {'name': 'epochs',           'type': 'discrete',    'domain': (5, 10, 20)}]

    # optimizer
    #opt_mnist = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
    opt_mnist=gpyopt_opt(f,15,bounds)
    # optimize mnist mode

    #opt_mnist.run_optimization(max_iter=10)
    print("""
    Optimized Parameters:
    \t{0}:\t{1}
    \t{2}:\t{3}
    \t{4}:\t{5}
    \t{6}:\t{7}
    \t{8}:\t{9}
    \t{10}:\t{11}
    \t{12}:\t{13}
    """.format(bounds[0]["name"],opt_mnist.x_opt[0],
            bounds[1]["name"],opt_mnist.x_opt[1],
            bounds[2]["name"],opt_mnist.x_opt[2],
            bounds[3]["name"],opt_mnist.x_opt[3],
            bounds[4]["name"],opt_mnist.x_opt[4],
            bounds[5]["name"],opt_mnist.x_opt[5],
            bounds[6]["name"],opt_mnist.x_opt[6]))
    print("optimized loss: {0}".format(opt_mnist.fx_opt))
    
    best_mnist = MNIST(l1_out=int(opt_mnist.x_opt[3]), l2_out=int(opt_mnist.x_opt[4]), 
                   l1_drop=float(opt_mnist.x_opt[1]), l2_drop=float(opt_mnist.x_opt[2]), 
                   batch_size=int(opt_mnist.x_opt[5]), 
                   epochs=int(opt_mnist.x_opt[6]), 
                   validation_split=float(opt_mnist.x_opt[0]))
    best_mnist.best_model_train()

if __name__=="__main__":
    main()
