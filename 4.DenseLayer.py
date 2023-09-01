import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Dense_layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights=np.random.randn(n_inputs,n_neurons)
        self.bias=np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output=np.dot(inputs,self.weights)+self.bias

X,y=spiral_data(samples=100,classes=3)
layer1=Dense_layer(2,3)
layer2=Dense_layer(3,4)
layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output[:5])
