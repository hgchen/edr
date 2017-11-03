import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


# def EMACell(input, hidden, alpha):
#     # hy = F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh)
#     hidden = alpha * input + (1 - alpha) * hidden
#     return hidden
#
# # class RecurrentEMA(torch.nn.modules.rnn.RNNCell):
# #     def forward(self, input, hx, alpha):
# #         func = EMACell
# #         return func(
# #             input, hx,
# #             alpha
# #         )
#
# class RecurrentEMA(torch.nn.modules.rnn.RNNCellBase):
#
#     def __init__(self, input_size, hidden_size, alpha=0.3, learnable=False):
#         alpha = torch.from_numpy(np.asarray([alpha], dtype=float))
#         self.alpha = Variable(alpha, requires_grad=learnable)
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         # self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, hx):
#         def forward(self, input, hx):
#             func = EMACell
#             return func(
#                 input, hx,
#                 self.alpha
#             )
#
# # class EDR(nn.Module):
# #     def __init__(self):
# #         super(EDR, self).__init__()
# #         self.rnn = RNNCellLinear(10,10, bias=False, nonlinearity=None)
# #
# #
# #     def forward(self, x, alpha):
#
#
# class EMA(nn.Module):
#     def __init__(self, alpha=0.3, learnable=False):
#         super(EMA, self).__init__()
#         alpha = torch.from_numpy(np.asarray([alpha], dtype=float))
#         self.alpha = Variable(alpha, requires_grad=False)
#         self.recurrent_ema = RecurrentEMA
#
#     def forward(self, input, ema):
#         out = self.recurrent_ema(input, ema, self.alpha)
#
#         return out
#
#         # ema = self.alpha * input + (1-self.alpha) * ema
#         # return ema

# model = edr_net.EMA(alpha=0.4, learnable=False)
# for i in range(9):
#     hidden = model(input[i], hidden)
#     output.append(hidden)
#
# output = torch.stack(output)

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

