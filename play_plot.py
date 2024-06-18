import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


x = torch.linspace(-1,1,100)
y = x.clone()
y[(y>0.6)&(y<0.8)] = y[(y>0.6)&(y<0.8)] / 1.5
y[y<0.6] = 0

plt.plot(x,y)
plt.show()