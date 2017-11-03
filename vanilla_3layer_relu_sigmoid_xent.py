#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:29:25 2017

@author: amajidsinar
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys

dataset = pd.read_csv('Dataset/fashion-mnist_train.csv').values

y = dataset[:,0]
y = np.reshape(y, (len(y),1))
X = dataset[:,1:]

mask = (y == 0) | (y == 1)
index = np.where(mask)
index = index[0]

y = y[index]
X = X[index] / 255

def relu(z):
    return z * (z > 0)

def relu_grad(z):
    return 1 * (z > 0)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z) * (1-sigmoid(z))


# to ensure that generated random numbers are the same no matter how many times you run this
np.random.seed(1)

alpha = 0.01
batch = 50000
minibatch = 64

a0 = X 
hidden_neuron = 5

# make sure the mean of weight is close to zero and std is close to one -- a good practice according to Efficient Backprop paper
w1 = np.random.randn(784,hidden_neuron)
w2 = np.random.randn(hidden_neuron,1)

# backprop + minibatch SGD
for iter in range(batch):
    error = 0
#    only use minibatch number of data 
    for i in range(0,X.shape[0]//minibatch,minibatch):
        batch_x=X[i:i+minibatch]
        batch_y=y[i:i+minibatch]
            
        a0 = batch_x    
        z1 = np.dot(a0,w1)
        a1 = sigmoid(z1)
        z2 = np.dot(a1,w2)
        a2 = sigmoid(z2)
            
        #update w2 first
        err2 = ((a2-batch_y)/(minibatch*a2*(1-a2)))*sigmoid_grad(z2)
        w2 -= alpha * np.dot(a1.T,err2)
            
            #update w1
        err1 = np.dot(err2,w2.T)*relu_grad(z1)
        w1 -= alpha * np.dot(a0.T,err1)
        
        error += (np.sum(np.abs(err2)))
    sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
    if(iter % 100 == 99):
        print("")

#a1 = relu(np.dot(X,w1))
#a2 = sigmoid(np.dot(a1,w2))
#    




