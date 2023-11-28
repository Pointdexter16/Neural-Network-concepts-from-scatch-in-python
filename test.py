import numpy as np
import nnfs
from nnfs import datasets

example_output=[1,2,0,4,0,3,0,0]
ratio=0.5
while True:
    index=np.random.randint(0,len(example_output)-1)
    example_output[index]=0
    dropout_no=0
    for output in example_output:
        if output==0:
            dropout_no+=1
    if dropout_no/len(example_output)>=0.5:
        break
print(example_output)