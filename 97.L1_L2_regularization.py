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
    
    def regularization_loss(self,layer):
        regularization_layer_loss=0
        if layer.weight_regularizer_l1>0:
            regularization_layer_loss+=layer.weight_regularizer_l1*np.sum(np.abs(layer.weights))
        if layer.bias_regularizer_l1>0:
            regularization_layer_loss+=layer.bias_regularizer_l1*np.sum(np.abs(layer.bias))
        if layer.weight_regularizer_l2>0:
            regularization_layer_loss+=layer.weight_regularizer_l2*np.sum(layer.weights*layer.weights)
        if layer.bias_regularizer_l2>0:
            regularization_layer_loss+=layer.bias_regularizer_l2*np.sum(layer.bias*layer.bias)
        return regularization_layer_loss

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
        return self.loss.calculate(self.output,y),self.loss.accuracy(self.output,y)
    
    def backward(self,dvalues,y):
        samples=len(dvalues)
        if len(y.shape) == 2:
            y=np.argmax(y,axis=1)
        self.dinputs=dvalues.copy()
        self.dinputs[range(samples),y]-=1
        self.dinputs=self.dinputs/samples
        
class Optimizer_SGD:
    def __init__(self,learning_rate=1.0,decay=0.,momentum=0.):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iteration=0
        self.momentum=momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1/(1+self.decay*self.iteration))


    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer,"weight_momentum"):
                layer.weight_momentum=np.zeros_like(layer.weights)
                layer.bias_momentum=np.zeros_like(layer.bias)
        
            weights_update=self.momentum*layer.weight_momentum - self.current_learning_rate*layer.dweights
            layer.weights_momentum=weights_update
            bias_update=self.momentum*layer.bias_momentum - self.current_learning_rate*layer.dbias
            layer.bias_momentum=bias_update
        
        else:
            weights_update= -self.current_learning_rate*layer.dweights
            bias_update= -self.current_learning_rate*layer.dbias
        
        layer.weights+=weights_update
        layer.bias+=bias_update
    
    def post_update_params(self):
        self.iteration+=1

class Optimizer_adagrad:
    def __init__(self,learning_rate=1.0,decay=0.,epsilon=1e-7):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iteration=0
        self.epsilon=epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1/(1+self.decay*self.iteration))


    def update_params(self,layer):
        if not hasattr(layer,"weight_cache"):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.bias)
        
        layer.weight_cache+=layer.dweights**2
        layer.bias_cache+=layer.dbias**2

        layer.weights-=layer.dweights*self.current_learning_rate/(np.sqrt(layer.weight_cache)+ self.epsilon)
        layer.bias-=layer.dbias*self.current_learning_rate/(np.sqrt(layer.bias_cache)+ self.epsilon)

    
    def post_update_params(self):
        self.iteration+=1


class Optimizer_rmsprop:
    def __init__(self,learning_rate=0.001,decay=0.,epsilon=1e-7,rho=0.95):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iteration=0
        self.epsilon=epsilon
        self.rho=rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1/(1+self.decay*self.iteration))


    def update_params(self,layer):
        if not hasattr(layer,"weight_cache"):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.bias)
        
        layer.weight_cache = self.rho*layer.weight_cache + (1-self.rho)*layer.dweights**2
        layer.bias_cache = self.rho*layer.bias_cache + (1-self.rho)*layer.dbias**2

        layer.weights-=layer.dweights*self.current_learning_rate/(np.sqrt(layer.weight_cache)+ self.epsilon)
        layer.bias-=layer.dbias*self.current_learning_rate/(np.sqrt(layer.bias_cache)+ self.epsilon)

    
    def post_update_params(self):
        self.iteration+=1

class Optimizer_adam:
    def __init__(self,learning_rate=0.001,decay=5e-7,epsilon=1e-7,beta1=0.9,beta2=0.999):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iteration=0
        self.epsilon=epsilon
        self.beta1=beta1
        self.beta2=beta2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1/(1+self.decay*self.iteration))


    def update_params(self,layer):
        if not hasattr(layer,"weight_cache"):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.bias)
            layer.weight_momentum=np.zeros_like(layer.weights)
            layer.bias_momentum=np.zeros_like(layer.bias)
        
        layer.weight_momentum = self.beta1*layer.weight_momentum + (1-self.beta1)*layer.dweights
        layer.bias_momentum = self.beta1*layer.bias_momentum + (1-self.beta1)*layer.dbias
        
        layer.weight_momentum_correct=layer.weight_momentum/(1-self.beta1**(self.iteration+1))
        layer.bias_momentum_correct=layer.bias_momentum/(1-self.beta1**(self.iteration+1))

        layer.weight_cache= self.beta2*layer.weight_cache + (1-self.beta2)*layer.dweights**2
        layer.bias_cache= self.beta2*layer.bias_cache + (1-self.beta2)*layer.dbias**2

        layer.weight_cache_correct=layer.weight_cache/(1-self.beta2**(self.iteration+1))
        layer.bias_cache_correct=layer.bias_cache/(1-self.beta2**(self.iteration+1))

        layer.weights-=self.current_learning_rate*layer.weight_momentum_correct/(np.sqrt(layer.weight_cache_correct)+ self.epsilon)
        layer.bias-=self.current_learning_rate*layer.bias_momentum_correct/(np.sqrt(layer.bias_cache_correct)+ self.epsilon)

    
    def post_update_params(self):
        self.iteration+=1


class Activation_ReLU:

    def forward(self,inputs):
        self.output=np.maximum(inputs,0)

    def backward(self,dvalues):
        self.dinputs=dvalues.copy()
        self.dinputs[self.output<=0]=0

class Dense_layer:
    def __init__(self,n_inputs,n_neurons,weight_regularizer_l1=0,
                 bias_regularizer_l1=0,weight_regularizer_l2=0,
                 bias_regularizer_l2=0):
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.bias=np.zeros((1,n_neurons))
        self.weight_regularizer_l1=weight_regularizer_l1
        self.bias_regularizer_l1=bias_regularizer_l1
        self.weight_regularizer_l2=weight_regularizer_l2
        self.bias_regularizer_l2=bias_regularizer_l2

    def forward(self,inputs):
        self.inputs=inputs.copy()
        self.output=np.dot(inputs,self.weights)+self.bias

    def backward(self,dvalues):
        self.dbias=np.sum(dvalues,axis=0,keepdims=True)
        self.dweights=np.dot(self.inputs.T,dvalues)

        if self.weight_regularizer_l1>0:
            dL1=np.ones_like(self.weights)
            dL1[self.weights<0]=-1
            self.dweights+=self.weight_regularizer_l1*dL1

        if self.bias_regularizer_l1>0:
            dL1=np.ones_like(self.bias)
            dL1[self.bias<0]=-1
            self.dbias+=self.bais_regularizer_l1*dL1

        if self.weight_regularizer_l2>0:
            self.dweights+=2*self.weight_regularizer_l2*self.weights

        if self.bias_regularizer_l2>0:
            self.dbias+=2*self.bias_regularizer_l2*self.bias

        self.dinputs=np.dot(dvalues,self.weights.T)

X,y=spiral_data(samples=1000,classes=3)
x_test,y_test=spiral_data(samples=100,classes=3)

layer1=Dense_layer(2,512,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
relu=Activation_ReLU()
layer2=Dense_layer(512,3)
softmax_loss=Activation_Softmax_categorial_Cross_Entropy()
optimizer=Optimizer_adam(learning_rate=0.02,decay=5e-7)
EPOCHS=10001
for epoch in range(EPOCHS):
    layer1.forward(X)
    relu.forward(layer1.output)
    layer2.forward(relu.output)
    loss,acc=softmax_loss.forward(layer2.output,y)
    reg_loss=softmax_loss.loss.regularization_loss(layer1) + softmax_loss.loss.regularization_loss(layer2)
    data_loss= loss + reg_loss

    if not epoch%100:
        print(f'epoch: {epoch} ',
              f'loss: {loss:.3f} (data_loss: {data_loss:.3f} reg_loss: {reg_loss:.3f} ) ',
              f'acc: {acc:.3f} ',
              f'learning rate{optimizer.current_learning_rate}')

    softmax_loss.backward(softmax_loss.output,y)
    layer2.backward(softmax_loss.dinputs)
    relu.backward(layer2.dinputs)
    layer1.backward(relu.dinputs)
    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()

layer1.forward(x_test)
relu.forward(layer1.output)
layer2.forward(relu.output)
loss,acc=softmax_loss.forward(layer2.output,y_test)
print(f'validation, loss:{loss:.3f}, acc:{acc:.3f}')

