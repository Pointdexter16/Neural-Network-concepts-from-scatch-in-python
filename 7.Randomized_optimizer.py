import numpy as np
import nnfs
from nnfs.datasets import vertical_data,spiral_data
import matplotlib.pyplot as plt

nnfs.init()

class Loss:

    def accuracy(self,y_pred,y_true):
        if(len(y_true.shape)==2):
            y_true=np.argmax(y_true,axis=1)
        y_pred=np.argmax(y_pred,axis=1)
        acc=np.mean(y_pred==y_true)
        return acc

    def calculate(self,output,y):
        
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss
    
class Categorical_Cross_Entropy(Loss):

    def forward(self,y_pred,y_true):
        samples=len(y_true)
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)

        if(len(y_true.shape)==2):  #changing one-hot to sparse
            y_true=np.argmax(y_true,axis=1)

        correct_confidence=y_pred_clipped[range(samples),y_true]
        neg_log=-np.log(correct_confidence)
        return neg_log


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

X,y=spiral_data(100,3)
# X,y=vertical_data(100,3)
# plt.scatter(X[:,0],X[:,1],c=y,s=20,cmap='brg')
# plt.show()

Dense1=Dense_layer(2,3)
Dense2=Dense_layer(3,3)
reLU1=Activation_ReLU()
softmax2=Activation_softmax()
crossEnt=Categorical_Cross_Entropy()

best_loss=1000
best_weight1=Dense1.weights.copy()
best_bias1=Dense1.bias.copy()
best_weight2=Dense2.weights.copy()
best_bias2=Dense2.bias.copy()


epochs=1000000
for epoch in range(epochs):

    Dense1.weights+=0.05*np.random.randn(2,3)
    Dense1.bias+=0.05*np.random.randn(1,3)
    Dense2.weights+=0.05*np.random.randn(3,3)
    Dense2.bias+=0.05*np.random.randn(1,3)

    Dense1.forward(X)
    reLU1.forward(Dense1.output)
    Dense2.forward(reLU1.output)
    softmax2.forward(Dense2.output)
    loss=crossEnt.calculate(softmax2.output,y)
    acc=crossEnt.accuracy(softmax2.output,y)

    if loss<best_loss:
        best_loss=loss
        best_weight1=Dense1.weights.copy()
        best_bias1=Dense1.bias.copy()
        best_weight2=Dense2.weights.copy()
        best_bias2=Dense2.bias.copy()
        print("loss: ",loss,'accuracy: ',acc,'epoch: ',epoch)
    else:
        Dense1.weights=best_weight1.copy()
        Dense1.bias=best_bias1.copy()
        Dense2.weights=best_weight2.copy()
        Dense2.bias=best_bias2.copy()






