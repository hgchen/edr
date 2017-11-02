import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def EMACell(input, hidden, alpha):
    # hy = F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh)
    ema = alpha * input + (1 - alpha) * hidden
    return ema

class RecurrentEMA(torch.nn.modules.rnn.RNNCell):
    def forward(self, input, hx, alpha):
        func = EMACell
        return func(
            input, hx,
            alpha
        )

class EDR(nn.Module):
    def __init__(self):
        super(EDR, self).__init__()
        self.rnn = RNNCellLinear(10,10, bias=False, nonlinearity=None)


    def forward(self, x, alpha):




class EMA(nn.Module):
    def __init__(self, alpha=0.3, learnable=False):
        super(EMA, self).__init__()

        self.alpha = Variable(alpha, require_grad=learnable)

    def forward(self, input, ema):
        RecurrentEMA(input, ema, self.alpha)

        # ema = self.alpha * input + (1-self.alpha) * ema
        # return ema