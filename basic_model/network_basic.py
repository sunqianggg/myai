#encoding:utf-8
import numpy as np

class Network(object):

    def __init__(self,initializer):
        self.num_layers=len(initializer['size'])
        self.size=initializer['size']
        self.biases=initializer['biases']
        self.weights=initializer['weights']

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a



def sigmoid(z):
    return(1/(1+np.exp(z)))
