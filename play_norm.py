import torch

t1 = torch.tensor([1.4,2.4,3.4])+10

t1l1 = torch.norm(t1, p=1)
t1l2 = torch.norm(t1, p=2)

t1norm1 = t1l1 / t1l2
t1norm2 = t1l1 / (t1l2*t1l2)

print(t1l1)
print(t1l2)
print(t1norm1)