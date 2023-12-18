import numpy as np

class Model:
    def __init__(self):
        self.layers=[]

    def add(self,layer):
        self.layers.append(layer)
    
    def set(self,*,optimizer,loss,accuracy):
        self.loss=loss
        self.optimizer=optimizer
        self.accuracy=accuracy

    def train(self,X,y,*,epochs=1,print_every=1,validation_data=None):

        self.accuracy.init(y)

        for epoch in range(0,epochs+1):

            output=self.forward(X)
            
            data_loss,reg_loss=self.loss.calculate(output,y,include_regularization=True)
            loss=data_loss+reg_loss

            prediction=self.output_layer_activation.prediction(output)
            accuracy=self.accuracy.calculate(prediction,y)

            self.backward(output,y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch%print_every:
                print(f'epoch: {epoch} ',
                    f'loss: {loss:.3f} (data_loss: {data_loss:.3f} reg_loss: {reg_loss:.3f} ) ',
                    f'acc: {accuracy:.3f} ',
                    f'learning rate{self.optimizer.current_learning_rate}')

        if validation_data:
            X_val,y_val=validation_data[0],validation_data[1]

            output_val=self.forward(X_val,Training=False)

            loss=self.loss.calculate(output_val,y_val)

            prediction=self.output_layer_activation.prediction(output_val)
            accuracy=self.accuracy.calculate(prediction,y_val)

            print(('validation: ',f'loss: {loss:.3f}',
                f'acc: {accuracy:.3f}'))

    def finalize(self):

        self.input_layer=input_layer()
        layer_count=len(self.layers)
        self.trainable_layers=[]
        for i in range(layer_count):
            if i==0:
                self.layers[i].prev=self.input_layer
                self.layers[i].next=self.layers[i+1]
            elif i<layer_count-1:
                self.layers[i].prev=self.layers[i-1]
                self.layers[i].next=self.layers[i+1]
            else:
                self.layers[i].prev=self.layers[i-1]
                self.layers[i].next=self.loss
                self.output_layer_activation=self.layers[i]

            if(hasattr(self.layers[i],'weights')):
                self.trainable_layers.append(self.layers[i])

        self.softmax_classification_output=None

        if(isinstance(self.output_layer_activation,Activation_softmax) and isinstance(self.loss,Categorical_Cross_Entropy)):
            self.softmax_classification_output=Activation_Softmax_categorial_Cross_Entropy()
        
        self.loss.remember_trainable_layers(self.trainable_layers)
        

    def forward(self,X,Training=True):

        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output,Training)

        return layer.output
    
    def backward(self,output,y):

        if self.softmax_classification_output is not None: 
            self.softmax_classification_output.backward(output,y)
            self.layers[-1].dinputs=self.softmax_classification_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output,y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
class Accuracy:

    def calculate(self,predictions,y):

        comparison=self.comparison(predictions,y)
        return np.mean(comparison)


class Accuracy_regression(Accuracy):

    def __init__(self):
        self.precision=None

    def init(self,y,reinit=False):
        if self.precision is None or reinit:
            self.precision=np.std(y)/250

    def comparison(self,prediction,y):
        return np.abs(prediction-y)<self.precision


class Accuracy_categorical(Accuracy):

    def init(self,y):
        pass

    def comparison(self,prediction,y):
        if(len(y.shape)==2):
            y=np.argmax(y,axis=-1)
        return prediction==y

class Loss:

    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers=trainable_layers

        
    def calculate(self,output,y,include_regularization=False):
        
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        if not(include_regularization):
            return data_loss
        return data_loss,self.regularization_loss()
    
    def regularization_loss(self):
        regularization_layer_loss=0
        for layer in self.trainable_layers:
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

class binary_cross_entropy_loss(Loss):
    

    def forward(self,y_pred,y_true):
        y_pred_clip=np.clip(y_pred,1e-7,1-1e-7)
        sample_loss= -(y_true*(np.log(y_pred_clip))  +   (1-y_true)*np.log(1-y_pred_clip))
        sample_loss=np.mean(sample_loss,axis=-1)
        return sample_loss

    def backward(self,dvalues,y_true):
        samples=len(dvalues)
        outputs=len(dvalues[0])
        clipped_values=np.clip(dvalues,1e-7,1-1e-7)
        self.dinputs=-((y_true/clipped_values)-((1-y_true)/(1-clipped_values)))/outputs
        self.dinputs/=samples

class mean_squared_error(Loss):


    def forward(self,y_pred,y_true):

        sample_loss=np.mean((y_true-y_pred)**2,axis=-1)
        return sample_loss

    def backward(self,dvalues,y_true):

        samples=len(dvalues)
        outputs=len(dvalues[0])
        self.dinputs=-2*(y_true-dvalues)/outputs
        self.dinputs=self.dinputs/samples   

class mean_absolute_error(Loss):


    def forward(self,y_pred,y_true):

        sample_loss=np.mean(np.abs(y_true-y_pred),axis=-1)
        return sample_loss

    def backward(self,dvalues,y_true):
        
        samples=len(dvalues)
        outputs=len(dvalues[0])
        self.dinputs=np.sign(y_true-dvalues)/outputs
        self.dinputs=self.dinputs/samples


class Activation_softmax:

    def forward(self,inputs,training):
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        self.output=exp_values/np.sum(exp_values,axis=1,keepdims=True)
    
    def backward(self,dvalues):
        self.dinputs=np.empty_like(dvalues)
        for index,(single_output,single_dvalue) in enumerate(zip(self.output,dvalues)):
            single_output=single_output.reshape(-1,1)
            jacobian_matrix=np.diagflat(single_output)-np.dot(single_output,single_output.T)
            self.dinputs[index]=np.dot(jacobian_matrix,single_dvalue)

    def prediction(self,outputs):
        
        return np.argmax(outputs,axis=-1)
    
class Activation_linear:
    
    def forward(self,inputs,training):
        self.output=inputs

    def backward(self,dvalues):
        self.dinputs=dvalues.copy()

    def prediction(self,outputs):

        return outputs

class Activation_Softmax_categorial_Cross_Entropy():
    
    def backward(self,dvalues,y):
        samples=len(dvalues)
        if len(y.shape) == 2:
            y=np.argmax(y,axis=1)
        self.dinputs=dvalues.copy()
        self.dinputs[range(samples),y]-=1
        self.dinputs=self.dinputs/samples

class Activation_sigmoid:
    
    def forward(self,inputs,training):
        self.output=1/(1+np.exp(-inputs))

    def backward(self,dvalues):
        self.dinputs=dvalues*self.output*(1-self.output)

    def prediction(self,outputs):
        return (outputs > 0.5)*1
        
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
    def __init__(self,learning_rate=0.05,decay=5e-7,epsilon=1e-7,beta1=0.9,beta2=0.999):
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


class Activation_Relu:

    def forward(self,inputs,training):
        self.output=np.maximum(inputs,0)

    def backward(self,dvalues):
        self.dinputs=dvalues.copy()
        self.dinputs[self.output<=0]=0

    def prediction(self,outputs):
        return outputs

class dropout_layer:
    def __init__(self,dropout_rate):
        self.acceptance_rate=1-dropout_rate

    def forward(self,inputs,training):
        if not training:
            self.output=inputs.copy()
            return 
        self.binary_mask=np.random.binomial(1,self.acceptance_rate,inputs.shape)/self.acceptance_rate
        self.output=inputs*self.binary_mask

    def backward(self,dvalues):
        self.dinputs=self.binary_mask*dvalues

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

    def forward(self,inputs,training):
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

class input_layer:

    def forward(self,inputs):
        self.output=inputs
