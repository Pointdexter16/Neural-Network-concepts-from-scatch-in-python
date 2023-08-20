import numpy as np
import nnfs
from nnfs.datasets import spiral_data

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

X,y=spiral_data(samples=100,classes=3)
layer1=Dense_layer(2,3)
layer2=Dense_layer(3,3)
relu=Activation_ReLU()
softmax=Activation_softmax()
layer1.forward(X)
relu.forward(layer1.output)
layer2.forward(relu.output)
softmax.forward(layer2.output)
loss_func=Categorical_Cross_Entropy()
loss=loss_func.calculate(softmax.output,y)
acc=loss_func.accuracy(softmax.output,y)
print('loss: ',loss,
      'accuracy: ',acc)
#creating one-hot targets and trying loss and accuracy funcitons
one_hot_y=np.expand_dims(y,axis=1)
condlist=[one_hot_y==0,one_hot_y==1,one_hot_y==2]
choicelist=[[1,0,0],[0,1,0],[0,0,1]]
one_hot_y=np.select(condlist=condlist,choicelist=choicelist)
loss_func_oh=Categorical_Cross_Entropy()
loss_oh=loss_func.calculate(softmax.output,one_hot_y)
acc_oh=loss_func.accuracy(softmax.output,one_hot_y)
print('loss-one-hot: ',loss_oh,
      'accuracy-one-hot: ',acc_oh)