import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def get_lambda2(target_size):
    if target_size <=5:
        return 0.0001
    reduce = (0.000001)*(target_size-5)
    r = 0.0001 - reduce
    if r <=0:
        return 0
    return r

x = np.linspace(0,30,31)
print(x)

y = [get_lambda2(i) for i in x]
print(y)

plt.plot(x,y)
plt.show()

for i in range(30):
    print(f"{x[i]}\t{y[i]}")