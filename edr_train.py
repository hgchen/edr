import edr_net
import torch.nn as nn
import torch
from torch.autograd import Variable

# model = edr_net.EMA(alpha=0.4, learnable=False)

# rnn = nn.RNNCell(10, 10)

input = Variable(torch.rand(1, 20,9,9))
hidden = Variable(torch.rand(1, 1,9,9))
output = []

model = edr_net.EDR(alpha=0.4, learnable=False)
# for i in range(9):
#     hidden = model(input[i], hidden)
#     output.append(hidden)

output = model(input,hidden)

output = torch.stack(output)

event = input / output

# output = []
# for i in range(6):
#     hx = rnn(input[i], hx)
#     output.append(hx)

print(output)

# out = model(input)
