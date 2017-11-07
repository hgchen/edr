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


    def forward(self, input, hidden):



        hidden = self.alpha.expand_as(input) * input + (1 - self.alpha).expand_as(input) * hidden

        return hidden

    # def initHidden(self):
    #     return Variable(torch.zeros(1, ))


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


class EDR(nn.Module):
    """
    alpha = 0.5 and 0.0166
    
    """
    def __init__(self, alpha=0.5, learnable=False, mu_on=0.05, mu_off=-0.1):
        super(EDR, self).__init__()
        self.ema_recurrent = EMARecurrent(alpha, learnable)
        self.mu_on = mu_on
        self.mu_off = mu_off


    def forward(self, input, hidden):

        x = input
        ema = self.ema_recurrent(input, hidden)

        x = x / ema
        x = torch.log(x)
        x = F.tanh(x)
        # x = x * 20
        pos = F.relu(x - self.mu_on).unsqueeze(2)
        neg = F.relu(-(x - self.mu_off)).unsqueeze(2)

        x = torch.cat((pos, neg), dim=2)

        return x

