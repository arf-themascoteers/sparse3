import torch
import torch.nn as nn

# Example 2D tensor with batch size 2 and 3 features
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Define LayerNorm with normalized shape matching the feature size
layer_norm = nn.LayerNorm(3)

# Apply LayerNorm
normalized_output = layer_norm(x)

# Apply exponential function to ensure all values are positive
positive_output = torch.exp(normalized_output)

print("Input tensor:")
print(x)
print("\nLayerNorm output:")
print(normalized_output)
print("\nPositive output after applying exponential function:")
print(positive_output)
