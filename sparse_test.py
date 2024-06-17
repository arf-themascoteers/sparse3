import torch
import matplotlib.pyplot as plt


def custom_regularization(weights):
    threshold = 0.8
    reg_loss = torch.where(weights < threshold, torch.abs(weights), (torch.abs(weights)-threshold)*.2+threshold)
    return reg_loss

weights = torch.linspace(-2,2,100)
s = custom_regularization(weights)

for i in range(len(weights)):
    print(f"Weight: {weights[i]:.2f} SCAD: {s[i]:.2f}")

plt.plot(weights, s)
plt.show()
