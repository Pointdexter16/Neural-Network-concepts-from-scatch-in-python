import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import NN_final as nn

def load_image_data(dataset,path):
    labels=os.listdir(os.path.join(path,dataset))
    X=[]
    y=[]
    for label in labels:
        for file in os.listdir(os.path.join(path,dataset,label)):
            image=cv2.imread(os.path.join(path,dataset,label,file),cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return np.array(X) , np.array(y).astype('uint8')

def create_data(path):
    X,y=load_image_data('train',path)
    X_test,y_test=load_image_data('test',path)
    return X,y,X_test,y_test

def scale_shuffle_data(data,y,shuffle=True):
    data=(data.astype('float32')).reshape(len(data),-1)
    data=(data-127.5)/127.5
    if shuffle:
        shuffle_index=np.array(range(len(data)))
        np.random.shuffle(shuffle_index)
        data=data[shuffle_index]
        y=y[shuffle_index]
    return data,y

fashion_mnist_labels={
    0:'T-shirt',
    1:'Trouser',
    2:'Pullover',
    3:'Dress',
    4:'Coat',
    5:'Sandal',
    6:'Shirt',
    7:'Sneaker',
    8:'Bad',
    9:'Ankle boot',
}

# X,y,X_test,y_test=create_data('fashion_image_folder')
# X,y=scale_shuffle_data(X,y)
# X_test,y_test=scale_shuffle_data(X_test,y_test,shuffle=False)

# model=nn.Model()
# model.add(nn.Dense_layer(X.shape[1],128))
# model.add(nn.Activation_Relu())
# model.add(nn.Dense_layer(128,128))
# model.add(nn.Activation_Relu())
# model.add(nn.Dense_layer(128,10))
# model.add(nn.Activation_softmax())

# model.set(optimizer=nn.Optimizer_adam(decay=1e-5),
#           loss=nn.Categorical_Cross_Entropy(),
#           accuracy=nn.Accuracy_categorical())

# model.finalize()

# model.train(X,y,epochs=10,batch_size=128,validation_data=(X_test,y_test),print_every=100)
# model.evalute(X_test,y_test)

model=nn.Model.load('fashion_model_para.model')
image_data=cv2.imread('tshirt.png',cv2.IMREAD_GRAYSCALE)
image_data=cv2.resize(image_data,(28,28))
image_data=255-image_data
data=(image_data.astype('float32')).reshape(1,-1)
data=(data-127.5)/127.5
confidance=model.predict(data)
prediction=model.output_layer_activation.prediction(confidance)
print(fashion_mnist_labels[prediction[0]])








    
