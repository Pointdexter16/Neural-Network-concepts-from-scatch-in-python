import numpy as np
import matplotlib.pyplot as plt

x=np.array(range(5))

def exponential(x):
    return 2*x**2

def tangent_line(x):
    return derivative*x + b
x_array=np.arange(0,5,0.001)
y=exponential(x_array)
p_delta=0.0001
color=['r','g','b','y','k']
for x in range(5):
    x_delta=x+p_delta
    f_x=exponential(x)
    f_x_delta=exponential(x_delta)
    derivative=(f_x_delta-f_x)/p_delta
    b=f_x - derivative*x
    plt.scatter(x,f_x,c=color[x])
    to_plot=[x-0.9,x,x+0.9]
    plt.plot(to_plot,[tangent_line(i) for i in to_plot],c=color[x])
plt.plot(x_array,y)
plt.show()