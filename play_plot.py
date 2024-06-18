import numpy as np
import matplotlib.pyplot as plt

k=0.3
d = 0.05

s = d/k

x = np.linspace(0,1,100)
y = x.copy()
y[np.abs(y)<k] = y[np.abs(y)<k]*s

plt.plot(x,y)
plt.show()
