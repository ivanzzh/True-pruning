import torch
import torch.nn as nn
from lib import *
import time
from memory_profiler import profile


class TestCNN(nn.Module):
    def __init__(self):
        super(TestCNN, self).__init__()

        self.layer1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(5, 5), stride=(2, 2))
        self.layer2 = nn.Conv2d(in_channels=10, out_channels=64, kernel_size=(3, 3), stride=(2, 2))
        self.layer3 = nn.Conv2d(in_channels=64, out_channels=12, kernel_size=(3, 3), stride=(1, 1))
        self.hidden = nn.ReLU()

    def forward(self, x):
        s = time.time()
        x = self.layer1(x)
        # x = self.hidden(x)
        x = self.layer2(x)
        # x = self.hidden(x)
        x = self.layer3(x)
        e = time.time()
        print("time consumption: {:.6f}s".format(e - s))
        # print(x.shape)

        return x


class SpReTestCNN(nn.Module):
    def __init__(self, state_dict, device, row_size, col_size):
        super().__init__()
        self.layer1 = ReConv(device, in_channels=3, out_channels=10, kernel_size=5, stride=2, col_size=col_size,
                             row_size=row_size,
                             weight_state_dict=state_dict['layer1.weight'], bias_state_dict=state_dict['layer1.bias'])
        row_size = int((row_size - 5) / 2 + 1)
        col_size = int((col_size - 5) / 2 + 1)
        self.layer2 = ReConv(device, in_channels=10, out_channels=64, kernel_size=3, stride=2, col_size=col_size,
                             row_size=row_size,
                             weight_state_dict=state_dict['layer2.weight'], bias_state_dict=state_dict['layer2.bias'])
        row_size = int((row_size - 3) / 2 + 1)
        col_size = int((col_size - 3) / 2 + 1)
        self.layer3 = ReConv(device, in_channels=64, out_channels=12, kernel_size=3, stride=1, col_size=col_size,
                             row_size=row_size,
                             weight_state_dict=state_dict['layer3.weight'], bias_state_dict=state_dict['layer3.bias'])
        self.hidden = nn.ReLU()

    def forward(self, x):
        s = time.time()
        x = self.layer1(x)
        # x = self.hidden(x)
        x = self.layer2(x)
        # x = self.hidden(x)
        x = self.layer3(x)
        e = time.time()
        print("Reconstructed Network time consumption: {:.6f}s".format(e - s))

        return x
