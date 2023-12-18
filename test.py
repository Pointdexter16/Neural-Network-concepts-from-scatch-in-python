import numpy as np
import nnfs
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt
import cv2

image_data=cv2.imread('tshirt.png',cv2.IMREAD_GRAYSCALE)
image_data=cv2.resize(image_data,(28,28))
plt.imshow(image_data,cmap='gray')
plt.show()