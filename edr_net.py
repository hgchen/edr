import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EMARecurrent(nn.Module):

    def __init__(self, alpha=0.4, learnable=False):
        super(EMARecurrent, self).__init__()
        self.ema = EMA(alpha, learnable)
        # self.output = []

    def forward(self, input, hidden):
        input = input.transpose(0, 1)
        hidden = hidden.transpose(0, 1)
        output = []

        for i in range(len(input)):
            hidden = self.ema(input[i], hidden)
            output.append(hidden)

        output = torch.stack(output)
        output = torch.squeeze(output, 1)
        output = output.transpose(0, 1)

        return output


class EMA(nn.Module):

    def __init__(self, alpha=0.3, learnable=False):
        super(EMA, self).__init__()
        alpha = torch.from_numpy(np.asarray([alpha], dtype=float))
        alpha = alpha.type(torch.FloatTensor)
        self.alpha = Variable(alpha, requires_grad=learnable)


    def forward(self, input, hidden):



        hidden = torch.abs(self.alpha).expand_as(input) * input + (1 - torch.abs(self.alpha).expand_as(input)) * hidden

        return hidden

    # def initHidden(self):
    #     return Variable(torch.zeros(1, ))


class EDR(nn.Module):
    def __init__(self, alpha=0.3, learnable=False):
        super(EDR, self).__init__()
        self.ema_recurrent = EMARecurrent(alpha, learnable)


    def forward(self, input, hidden):

        x = input
        ema = self.ema_recurrent(input, hidden)

        x = x / ema
        x = torch.log(x)
        x = x * 20
        pos = F.relu(x - 0.01).unsqueeze(2)
        neg = F.relu(-x - 0.01).unsqueeze(2)

        x = torch.cat((pos, neg), dim=2)

        return x

