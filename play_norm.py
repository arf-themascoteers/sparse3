import torch

t1 = torch.tensor([1.0,2.0,3.0])*1

t1l1 = torch.norm(t1, p=1)
t1l2 = torch.norm(t1, p=2)

t1norm1 = t1l1 / t1l2
t1norm2 = t1l1 / (t1l2*t1l2)

print(t1l1)
print(1/t1l2)