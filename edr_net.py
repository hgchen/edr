import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class EMA(nn.Module):

    def __init__(self, alpha=0.3, learnable=False):
        super(EMA, self).__init__()
        alpha = torch.from_numpy(np.asarray([alpha], dtype=float))
        alpha = alpha.type(torch.FloatTensor)
        self.alpha = Variable(alpha, requires_grad=learnable)

    def forward(self, inp, hidden):

        hidden = torch.abs(self.alpha).expand_as(inp) * inp + \
                 (1 - torch.abs(self.alpha).expand_as(inp)) * hidden

        return hidden

    # def init_hidden(self):
    #     return Variable(torch.zeros(1, ))


class EMARecurrent(nn.Module):

    def __init__(self, alpha=0.4, learnable=False, recurrent_dim=2):
        super(EMARecurrent, self).__init__()
        self.ema = EMA(alpha, learnable)
        self.recurrent_dim = recurrent_dim
        # self.output = []

    def forward(self, inp, hidden):

        inp_l = torch.chunk(inp, inp.size(self.recurrent_dim), self.recurrent_dim)
        # hidden = hidden.transpose(0, 1)
        output = []

        for i in range(len(inp_l)):
            hidden = self.ema(inp_l[i], hidden)
            output.append(hidden)

        output = torch.cat(output, dim=self.recurrent_dim)

        return output


class EDR(nn.Module):
    def __init__(self, alpha=0.3, learnable=False):
        super(EDR, self).__init__()
        self.recurrent_dim = 2
        self.ema_recurrent = EMARecurrent(alpha, learnable, self.recurrent_dim)

    def forward(self, inp, hidden):

        x = inp
        ema = self.ema_recurrent(inp, hidden)

        x = x / (ema + 1e-5)
        x = torch.log(x)
        x = torch.tanh(x)
        # x = x * 20
        pos = F.relu(x - 0.05).unsqueeze(2)
        neg = F.relu(-(x - 0.1)).unsqueeze(2)

        x = torch.cat((pos, neg), dim=2)

        return x

