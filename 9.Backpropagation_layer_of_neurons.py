
#     O-
#   /
# O - O-
#   \ 
#     O-

import numpy as np

dvalues=np.array([[1.,1.,1.],
                  [2.,2.,2.],
                  [3.,3.,3.]])

weights=np.array([[1.2,-2.4,-1.3,-4.2],
                  [2.4,2.1,-5.2,4.8],
                  [-1.1,1.8,6.1,-2.3]]).T

dx0=sum(weights[0]*dvalues[0])
dx1=sum(weights[1]*dvalues[0])
dx2=sum(weights[2]*dvalues[0])
dx3=sum(weights[3]*dvalues[0])
dinputs=np.array([dx0,dx1,dx2,dx3])

# print(dinputs)

dinputs=np.dot(dvalues,weights.T)
# print("dinputs\n",dinputs)

inputs=np.array([[1.2,3.1,1.2,3.1],
                 [3.2,5.2,6.7,1.7],
                 [4.2,6.3,1.2,6.6]])

dweights=np.dot(inputs.T,dvalues)
# print("dweights\n",dweights)

dbias=np.sum(dvalues,axis=0,keepdims=True)
# print("dbias\n",dbias)

bias=[1,1,1]

z=np.dot(inputs,weights)+bias

drelu=np.zeros_like(z)
drelu[z>0]=1
drelu*=dvalues
# print("drelu\n",drelu)

# optimizing drelu calculation
drelu=dvalues.copy()
drelu[z<=0]=0
# print(drelu)

# MINIMIZING RELU OUTPUT
output=np.dot(inputs,weights)+bias
output=np.maximum(output,0)
print(output)
drelu=output.copy()
drelu[output<=0]=0
dinput=np.dot(drelu,weights.T)
dweights=np.dot(inputs.T,drelu)
dbias=np.sum(output,axis=0,keepdims=True)
weights -= 0.001*dweights
bias -= 0.001*dbias
print("minimized output\n",np.maximum(np.dot(inputs,weights)+bias,0))
