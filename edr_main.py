import edr_net
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# input = Variable(torch.rand(5, 6, 224, 224))
# hidden = Variable(torch.rand(5, 1, 224, 224))

input = np.load('input_view.np.npy')
input_tensor = torch.from_numpy(input)
hidden = input_tensor[:, :, 1, :, :, :].unsqueeze(dim=2)

input_tensor = input_tensor / 255
hidden = hidden / 255

input_tensor = Variable(input_tensor)
hidden = Variable(hidden)

model = edr_net.EDR(alpha=0.4, learnable=False)
output = model(input_tensor, hidden)

output_img_all = output.data.cpu().numpy()
for i in range(5):
    output_img = output_img_all[0, 0, :, i, 0, :, :].transpose(1,2,0)
    output_img_edr = np.zeros((224, 224, 3))
    output_img_edr[:, :, 0:2] = output_img
    plt.imshow(output_img_edr)
    plt.pause(0.5)


# print(output)