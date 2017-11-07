import edr_net
import torch.nn as nn
import torch
from torch.autograd import Variable

input = Variable(torch.rand(1, 20,9,9))
hidden = Variable(torch.rand(1, 1,9,9))
output = []

model = edr_net.EDR(alpha=0.4, learnable=False, mu_on=0.05, mu_off=-0.1)
output = model(input,hidden)

print(output)