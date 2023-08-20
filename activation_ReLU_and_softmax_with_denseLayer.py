import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Activation_softmax:

    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        self.output=exp_values/np.sum(exp_values,axis=1,keepdims=True)

class Activation_ReLU:

    def forward(self,inputs):
        self.output=np.maximum(inputs,0)

class Dense_layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights=np.random.randn(n_inputs,n_neurons)
        self.bias=np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.bias

X,y=spiral_data(samples=100,classes=3)
layer1=Dense_layer(2,3)
layer2=Dense_layer(3,3)
relu=Activation_ReLU()
softmax=Activation_softmax()
layer1.forward(X)
relu.forward(layer1.output)
layer2.forward(relu.output)
softmax.forward(layer2.output)
print(softmax.output[:5])
print(np.sum(softmax.output[:5],axis=1))
