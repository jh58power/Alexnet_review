import torch
import torch.nn.functional as F

a = torch.tensor([1,1,1,1]) * 0.1065
b = torch.tensor([2,2,2,2])


print(a)

c = F.softmax(torch.FloatTensor([2,4,2]),dim=0)
print(c)