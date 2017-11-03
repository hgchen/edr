import edr_net
import torch.nn as nn
import torch
from torch.autograd import Variable

input = Variable(torch.rand(1, 20,9,9))
hidden = Variable(torch.rand(1, 1,9,9))
output = []

model = edr_net.EDR(alpha=0.4, learnable=False)
output = model(input,hidden)

print(output)