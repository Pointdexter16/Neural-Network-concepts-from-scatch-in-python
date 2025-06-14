import numpy as np
# nxm

def max_pool(tensor1,feature_map_n,feature_map_m,max_pool_n,max_pool_m,stride,back=False):
    iteration_x=int(np.ceil(((feature_map_m - max_pool_m)/stride)+1))
    iteration_y=int(np.ceil(((feature_map_n - max_pool_n)/stride)+1))
    final=np.zeros((iteration_y,iteration_x))
    for y in range(iteration_y):
        for x in range(iteration_x):
            if back:
                final[y,x]=np.argmax(tensor1[y*stride:y*stride+max_pool_n,x*stride:x*stride+max_pool_m])
            else:
                final[y,x]=np.argmax(tensor1[y*stride:y*stride+max_pool_n,x*stride:x*stride +max_pool_m])
    
    return final if not back else final,iteration_x,iteration_y
            
def max_pool_back(tensor1,tensorgradient,feature_map_n,feature_map_m,max_pool_n,max_pool_m,stride):
    gradient=np.zeros((feature_map_n,feature_map_m))
    final,iteration_x,iteration_y=max_pool(tensor1,feature_map_n,feature_map_m,max_pool_n,max_pool_m,stride,back=True)
    for y in range(iteration_y):
        for x in range(iteration_x):
            gradient[y*stride:y*stride+max_pool_n,x*stride:x*stride +max_pool_m][int(final[y,x]//max_pool_n),int(final[y,x]%max_pool_m)]=tensorgradient[y,x]
    return gradient


feature_map_n=4 
feature_map_m=4 
max_pool_n=2
max_pool_m=2
stride=2

a=np.array([[1,2,3,6],
            [4,7,3,1],
            [6,4,3,7],
            [5,34,54,5]])

b=np.array([[1,2],
            [4,7]])
            
forward=max_pool(a,feature_map_n,feature_map_m,max_pool_n,max_pool_m,stride)
backward=max_pool_back(a,b,feature_map_n,feature_map_m,max_pool_n,max_pool_m,stride)
print(forward,backward)

