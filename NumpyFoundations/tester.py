import numpy as np


array = np.arange(1,61).reshape(2,10,3)
#print(array)
print(array.shape)
slices = array[::2]

print(slices)