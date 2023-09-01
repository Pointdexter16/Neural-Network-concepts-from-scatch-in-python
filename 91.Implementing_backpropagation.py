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
    
    def backward(self,dvalues,y_true):

        samples=len(dvalues)
        labels=len(dvalues[0])

        if(len(y_true.shape)==1):
            y_true=np.eye(labels)[y_true]

        self.dinputs = - y_true/dvalues
        self.dinputs = self.dinputs/samples

class Activation_softmax:

    def forward(self,inputs):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        self.output=exp_values/np.sum(exp_values,axis=1,keepdims=True)
    
    def backward(self,dvalues):
        self.dinputs=np.empty_like(dvalues)
        for index,(single_output,single_dvalue) in enumerate(zip(self.output,dvalues)):
            single_output=single_output.reshape(-1,1)
            jacobian_matrix=np.diagflat(single_output)-np.dot(single_output,single_output.T)
            self.dinputs[index]=np.dot(jacobian_matrix,single_dvalue)

class Activation_Softmax_categorial_Cross_Entropy():
    def __init__(self):
        self.activation=Activation_softmax()
        self.loss=Categorical_Cross_Entropy()

    def forward(self,inputs,y):
        self.activation.forward(inputs)
        self.output=self.activation.output
        return self.loss.calculate(self.output,y)
    
    def backward(self,dvalues,y):
        samples=len(dvalues)
        if len(y.shape) == 2:
            y=np.argmax(y,axis=1)
        self.dinputs=dvalues.copy()
        self.dinputs[range(samples),y]-=1
        self.dinputs=self.dinputs/samples
        


class Activation_ReLU:

    def forward(self,inputs):
        self.output=np.maximum(inputs,0)

    def backward(self,dvalues):
        self.dinputs=dvalues.copy()
        self.dinputs[self.output<=0]=0

class Dense_layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights=np.random.randn(n_inputs,n_neurons)
        self.bias=np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.inputs=inputs.copy()
        self.output=np.dot(inputs,self.weights)+self.bias

    def backward(self,dvalues):
        self.dbiases=np.sum(dvalues,axis=0,keepdims=True)
        self.dweights=np.dot(self.inputs.T,dvalues)
        self.dinputs=np.dot(dvalues,self.weights.T)

X,y=spiral_data(samples=100,classes=3)
layer1=Dense_layer(2,3)
layer2=Dense_layer(3,3)
relu=Activation_ReLU()
# softmax=Activation_softmax()
layer1.forward(X)
relu.forward(layer1.output)
layer2.forward(relu.output)

# softmax.forward(layer2.output)
# loss_func=Categorical_Cross_Entropy()
# loss=loss_func.calculate(softmax.output,y)
# loss_func.backward(softmax.output,y)
# softmax.backward(loss_func.dinputs)
# print(loss_func.dinputs[0])
# print(softmax.dinputs[0])

softmax_cross_ent=Activation_Softmax_categorial_Cross_Entropy()
loss=softmax_cross_ent.forward(layer2.output,y)

softmax_cross_ent.backward(softmax_cross_ent.output,y)
layer2.backward(softmax_cross_ent.dinputs)
relu.backward(layer2.dinputs)
layer1.backward(relu.dinputs)
print(layer1.dweights,"\n\n",layer2.dweights)

# acc=loss_func.accuracy(softmax.output,y)
# print('loss: ',loss,
#       'accuracy: ',acc)
#creating one-hot targets and trying loss and accuracy funcitons
