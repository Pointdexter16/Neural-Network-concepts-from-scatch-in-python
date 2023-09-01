import matplotlib.pyplot as plt
import numpy as np
import math

a=np.expand_dims(np.linspace(-1,1,100),axis=1)
b=np.concatenate((a,a),axis=1)
b=np.concatenate((a,b),axis=1)
weights1=[1,1,1]
weights2=[5,5,5]
bias1=0
bias2=5

X1=np.dot(b,weights1)+bias1
X2=np.dot(b,weights1)+bias2
X3=np.dot(b,weights2)+bias1

output1=1/(1+math.e**(-X1))
output2=1/(1+math.e**(-X2))
output3=1/(1+math.e**(-X3))


fig,ax=plt.subplots(3)
ax[0].plot(a,output1)
ax[1].plot(a,output2)
ax[2].plot(a,output3)
plt.show()
