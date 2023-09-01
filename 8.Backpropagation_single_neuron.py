inputs=[-1.0,6.3,10.1]
weights=[2.,0.,10.3]
bias=2
z=inputs[0]*weights[0]+inputs[1]*weights[1]+inputs[2]*weights[2]+bias
relu= z if z>0 else 0

dvalue=1.0
drelu_dz=dvalue*(1 if z>0 else 0)
dsum_dxw0=1
dsum_dxw1=1
dsum_dxw2=1
dsum_db=1
drelu_dxw0=drelu_dz*dsum_dxw0
drelu_dxw1=drelu_dz*dsum_dxw1
drelu_dxw2=drelu_dz*dsum_dxw2
dmul_dx0=weights[0]
dmul_dx1=weights[1]
dmul_dx2=weights[2]
dmul_dw0=inputs[0]
dmul_dw1=inputs[1]
dmul_dw2=inputs[2]
drelu_dx0=drelu_dxw0*dmul_dx0
drelu_dx1=drelu_dxw1*dmul_dx1
drelu_dx2=drelu_dxw2*dmul_dx2
drelu_dw0=drelu_dxw0*dmul_dw0
drelu_dw1=drelu_dxw1*dmul_dw1
drelu_dw2=drelu_dxw2*dmul_dw2
drelu_db=drelu_db=drelu_dz*dsum_db

dw=[drelu_dw0,drelu_dw1,drelu_dw2]
dx=[drelu_dx0,drelu_dx1,drelu_dx2]
db=drelu_db

print("original output",relu)

weights[0]+=-0.001*dw[0]
weights[1]+=-0.001*dw[1]
weights[2]+=-0.001*dw[2]
bias+=-0.001*db

xw0=weights[0]*inputs[0]
xw1=weights[1]*inputs[1]
xw2=weights[2]*inputs[2]

z=xw0+xw1+xw2

relu=z if z>0 else 0

print("minimized output",relu)


# short-version
# drelu_dx0=drelu_dxw0 * dmul_dx0
# drelu_dx0=drelu_dz * dsum_dxw0 * weights[0]
# drelu_dx0=dvalue * (1 if z>0 else 0) * 1 * weights[0]

