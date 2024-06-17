import torch
import matplotlib.pyplot as plt


def scad_penalty(weights, lambda_val=1, a=1):
    abs_weights = torch.abs(weights)
    scad_output = torch.zeros_like(abs_weights)

    case1 = abs_weights <= lambda_val
    scad_output[case1] = lambda_val * abs_weights[case1]

    case2 = (abs_weights > lambda_val) & (abs_weights <= a * lambda_val)
    scad_output[case2] = (2 * a * lambda_val * abs_weights[case2] - abs_weights[case2] ** 2 - lambda_val ** 2) / (
                2 * (a - 1))

    case3 = abs_weights > a * lambda_val
    scad_output[case3] = (a + 1) * lambda_val ** 2 / 2

    return scad_output


weights = torch.linspace(-2,2,100)
s = scad_penalty(weights)

for i in range(len(weights)):
    print(f"Weight: {weights[i]:.2f} SCAD: {s[i]:.2f}")

plt.plot(weights, s)
plt.show()
