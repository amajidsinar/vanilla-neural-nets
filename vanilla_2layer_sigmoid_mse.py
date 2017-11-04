#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 22:08:40 2017

@author: amajidsinar
"""

import numpy as np
import sys


X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,1,1,0]]).T

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z) * 1-sigmoid(z)

# to ensure that generated random numbers are the same no matter how many times you run this
np.random.seed(1)

alpha = 0.4
batch = 10000
minibatch = 2

a0 = X 

# make sure the mean of weight is close to zero and std is close to one -- a good practice according to Efficient Backprop paper
w1 = np.random.randn(a0.shape[1],y.shape[1])

# naive batch
#for i in range(batch):
#    z1 = np.dot(a0,w1)
#    a1 = sigmoid(z1)
#    err1 = ((a1-y)/batch)*sigmoid_prime(z1)
#    w1 -= alpha * np.dot(a0.T,err1)

#def mini():
#    arr = np.arange(X.shape[0])
#    np.random.shuffle(arr)
#    return arr
#
#    
# backprop + minibatch SGD
# backprop + minibatch SGD
for iter in range(batch):
    error = 0
#    only use minibatch number of data 
    for i in range(0,X.shape[0],minibatch):
        batch_x=X[i:i+minibatch]
        batch_y=y[i:i+minibatch]
            
        a0 = batch_x    
        z1 = np.dot(a0,w1)
        a1 = sigmoid(z1)

        #update w2 first
        err1 = ((a1-batch_y)/minibatch)*sigmoid(z1)
        w1 -= alpha * np.dot(a0.T,err1)
        
        error += (np.sum(np.abs(err1)))
    sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
    if(iter % 100 == 99):
        print("")
    


