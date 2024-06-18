import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


x = torch.linspace(-1,1,100)

lrl = nn.LeakyReLU(2)

y = torch.where(x<0.8, lrl(x-0.8)+0.8 ,x)

plt.plot(x,y)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()