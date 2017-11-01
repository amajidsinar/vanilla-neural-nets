#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 07:36:30 2017

@author: amajidsinar
"""

import numpy as np


X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,1,1,0]]).T

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
batch = 10000
minibatch = 2

a0 = X 
hidden_neuron = 5

# make sure the mean of weight is close to zero and std is close to one -- a good practice according to Efficient Backprop paper
w1 = np.random.randn(a0.shape[1],hidden_neuron)
w2 = np.random.randn(hidden_neuron,y.shape[1])

# backprop + minibatch SGD
for i in range(batch):
    #only use minibatch number of data 
    shuffle = np.random.permutation(a0.shape[0])
    a0=a0[shuffle]
    a0=a0[:minibatch]
    y=y[shuffle]
    y=y[:minibatch]
    
    
    z1 = np.dot(a0,w1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1,w2)
    a2 = sigmoid(z2)
    
    #update w2 first
    err2 = ((a2-y)/minibatch)*sigmoid_grad(z2)
    w2 -= alpha * np.dot(a1.T,err2)
    
    #update w1
    err1 = np.dot(err2,w2.T)*sigmoid_grad(z1)
    w1 -= alpha * np.dot(a0.T,err1)
    





