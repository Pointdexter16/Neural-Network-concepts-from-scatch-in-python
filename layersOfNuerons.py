import numpy as np

inputs1=[[1.3,2.4,3.2,6.3],
         [1.2,4.2,6.2,2.1],
         [4.2,1.2,5.2,5.1],
         [4.7,1.5,1.2,5.3],
         [1.2,2.2,7.2,3.1]]

weights2=[[1.4,1.3,6.3,5.3],
          [6.4,5.3,6.6,6.2],
          [3.4,1.5,6.1,2.1]]

bias2=[2,1,3]

weights3=[[2.1,5.3,1.6],
          [6.1,6.1,1.2]]

bias3=[5,2]

output2=np.dot(inputs1,np.array(weights2).T)+bias2
output3=np.dot(output2,np.array(weights3).T)+bias3

print(output3)