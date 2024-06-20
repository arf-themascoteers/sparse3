import torch
import torch.nn as nn
import matplotlib.pyplot as plt

relu = nn.ReLU()
lrelu = nn.LeakyReLU()

def custom_regularization2(weights):
    threshold = 0.8
    m_weights = weights-threshold
    m_weights = relu(m_weights)
    m_weights = torch.where(m_weights==0,0, m_weights+threshold)
    return m_weights

def custom_regularization(weights):
    threshold = 0.8
    #weights = torch.where(weights<0.8,0, weights)
    weights = torch.where(weights<0.8,weights*0.2, weights)
    return weights


weights = torch.linspace(-2,2,100)
s = custom_regularization(weights)

for i in range(len(weights)):
    print(f"Weight: {weights[i]:.2f} LR: {s[i]:.2f}")

plt.plot(weights, s)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show()
