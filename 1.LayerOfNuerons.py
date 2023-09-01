import numpy as np
inputs=[[1.3,3.5,2.6],
        [3,4,1],
        [1.3,3.2,4.2]]

weights=[[2,3,4],
         [5,1,2],
         [7,2,2]]

biases=[1,2,3]

outputs=np.dot(inputs,np.array(weights).T)+biases    
print("output of neural network is\n", outputs)
