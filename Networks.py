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
        x = self.hidden(x)
        x = self.layer2(x)
        x = self.hidden(x)
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
        x = self.hidden(x)
        x = self.layer2(x)
        x = self.hidden(x)
        x = self.layer3(x)
        e = time.time()
        print("Reconstructed Network time consumption: {:.6f}s".format(e - s))

        return x


class SpReTestCNNTorch(nn.Module):
    def __init__(self, batches, state_dict, device, row_size, col_size):
        super().__init__()
        self.layer1 = ReConv_torch(batches, device, in_channels=3, out_channels=10, kernel_size=5, stride=2,
                                   col_size=col_size,
                                   row_size=row_size,
                                   weight_state_dict=state_dict['layer1.weight'],
                                   bias_state_dict=state_dict['layer1.bias'])
        row_size = int((row_size - 5) / 2 + 1)
        col_size = int((col_size - 5) / 2 + 1)
        self.layer2 = ReConv_torch(batches, device, in_channels=10, out_channels=64, kernel_size=3, stride=2,
                                   col_size=col_size,
                                   row_size=row_size,
                                   weight_state_dict=state_dict['layer2.weight'],
                                   bias_state_dict=state_dict['layer2.bias'])
        row_size = int((row_size - 3) / 2 + 1)
        col_size = int((col_size - 3) / 2 + 1)
        self.layer3 = ReConv_torch(batches, device, in_channels=64, out_channels=12, kernel_size=3, stride=1,
                                   col_size=col_size,
                                   row_size=row_size,
                                   weight_state_dict=state_dict['layer3.weight'],
                                   bias_state_dict=state_dict['layer3.bias'])
        self.hidden = nn.ReLU()

    def forward(self, x):
        s = time.time()
        x = self.layer1(x)
        x = self.hidden(x)
        x = self.layer2(x)
        x = self.hidden(x)
        x = self.layer3(x)
        e = time.time()
        print("Reconstructed Network time consumption: {:.6f}s".format(e - s))

        return x


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(10, 10), stride=(4, 4))
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(5, 5), stride=(1, 1))
        self.layer3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1))
        self.layer4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1))
        self.layer5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1))
        self.Rel = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

    def forward(self, x):
        s = time.time()
        x = self.layer1(x)
        x = self.Rel(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.Rel(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.Rel(x)
        x = self.layer4(x)
        x = self.Rel(x)
        x = self.layer5(x)
        x = self.Rel(x)
        x = self.maxpool(x)
        e = time.time()
        print("Original Network time consumption: {:.6f}s".format(e - s))
        return x


class SpReAlexNetTorch(nn.Module):
    def __init__(self, batches, state_dict, device, row_size, col_size):
        super().__init__()
        row_size1, col_size1 = row_size, col_size
        self.layer1 = ReConv_torch(batches, device, in_channels=3, out_channels=64, kernel_size=10, stride=4,
                                   col_size=col_size1, row_size=row_size1,
                                   weight_state_dict=state_dict['layer1.weight'],
                                   bias_state_dict=state_dict['layer1.bias'])
        row_size1 = int((row_size1 - 10) / 4 + 1)
        col_size1 = int((col_size1 - 10) / 4 + 1)
        row_size1 = int((row_size1 - 3) / 2 + 1)
        col_size1 = int((col_size1 - 3) / 2 + 1)
        self.layer2 = ReConv_torch(batches, device, in_channels=64, out_channels=192, kernel_size=5, stride=1,
                                   col_size=col_size1, row_size=row_size1,
                                   weight_state_dict=state_dict['layer2.weight'],
                                   bias_state_dict=state_dict['layer2.bias'])
        row_size1 = int((row_size1 - 5) / 1 + 1)
        col_size1 = int((col_size1 - 5) / 1 + 1)
        row_size1 = int((row_size1 - 3) / 2 + 1)
        col_size1 = int((col_size1 - 3) / 2 + 1)
        self.layer3 = ReConv_torch(batches, device, in_channels=192, out_channels=384, kernel_size=3, stride=1,
                                   col_size=col_size1, row_size=row_size1,
                                   weight_state_dict=state_dict['layer3.weight'],
                                   bias_state_dict=state_dict['layer3.bias'])
        row_size1 = int((row_size1 - 3) / 1 + 1)
        col_size1 = int((col_size1 - 3) / 1 + 1)
        self.layer4 = ReConv_torch(batches, device, in_channels=384, out_channels=256, kernel_size=3, stride=1,
                                   col_size=col_size1, row_size=row_size1,
                                   weight_state_dict=state_dict['layer4.weight'],
                                   bias_state_dict=state_dict['layer4.bias'])
        row_size1 = int((row_size1 - 3) / 1 + 1)
        col_size1 = int((col_size1 - 3) / 1 + 1)
        self.layer5 = ReConv_torch(batches, device, in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                   col_size=col_size1, row_size=row_size1,
                                   weight_state_dict=state_dict['layer5.weight'],
                                   bias_state_dict=state_dict['layer5.bias'])
        self.Rel = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

    def forward(self, x):
        s = time.time()
        x = self.layer1(x)
        x = self.Rel(x)
        x1 = self.maxpool(x)
        x = self.layer2(x1)
        x = self.Rel(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.Rel(x)
        x = self.layer4(x)
        x = self.Rel(x)
        x = self.layer5(x)
        x = self.Rel(x)
        x = self.maxpool(x)
        e = time.time()
        print("Reconstructed Network time consumption: {:.6f}s".format(e - s))
        return x


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1))
        self.layer4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1))
        self.layer5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1))
        self.layer6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1))
        self.layer7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1))
        self.layer8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1))
        self.layer9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1))
        self.layer10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1))
        self.layer11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1))
        self.layer12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1))
        self.layer13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        s = time.time()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.maxpool(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.maxpool(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.maxpool(x)
        e = time.time()
        print("Original Network time consumption: {:.6f}s".format(e - s))
        return x


class SpReVGG16Torch(nn.Module):
    def __init__(self, batches, state_dict, device, row_size, col_size):
        super().__init__()
        self.layer1 = ReConv_torch(batches, device, in_channels=3, out_channels=64, kernel_size=3, stride=1,
                                   col_size=col_size,
                                   row_size=row_size,
                                   weight_state_dict=state_dict['layer1.weight'],
                                   bias_state_dict=state_dict['layer1.bias'])
        row_size1, col_size1 = transfer(row_size, col_size, 3, 1)
        self.layer2 = ReConv_torch(batches, device, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                   col_size=col_size1,
                                   row_size=row_size1, weight_state_dict=state_dict['layer2.weight'],
                                   bias_state_dict=state_dict['layer2.bias'])
        row_size2, col_size2 = transfer(row_size1, col_size1, 3, 1)
        row_size2, col_size2 = transfer(row_size2, col_size2, 2, 2)
        self.layer3 = ReConv_torch(batches, device, in_channels=64, out_channels=128, kernel_size=3, stride=1,
                                   col_size=col_size2,
                                   row_size=row_size2, weight_state_dict=state_dict['layer3.weight'],
                                   bias_state_dict=state_dict['layer3.bias'])
        row_size3, col_size3 = transfer(row_size2, col_size2, 3, 1)
        self.layer4 = ReConv_torch(batches, device, in_channels=128, out_channels=128, kernel_size=3, stride=1,
                                   col_size=col_size3,
                                   row_size=row_size3, weight_state_dict=state_dict['layer4.weight'],
                                   bias_state_dict=state_dict['layer4.bias'])
        row_size4, col_size4 = transfer(row_size3, col_size3, 3, 1)
        row_size4, col_size4 = transfer(row_size4, col_size4, 2, 2)
        self.layer5 = ReConv_torch(batches, device, in_channels=128, out_channels=256, kernel_size=3, stride=1,
                                   col_size=col_size4,
                                   row_size=row_size4, weight_state_dict=state_dict['layer5.weight'],
                                   bias_state_dict=state_dict['layer5.bias'])
        row_size5, col_size5 = transfer(row_size4, col_size4, 3, 1)
        self.layer6 = ReConv_torch(batches, device, in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                   col_size=col_size5,
                                   row_size=row_size5, weight_state_dict=state_dict['layer6.weight'],
                                   bias_state_dict=state_dict['layer6.bias'])
        row_size5, col_size5 = transfer(row_size5, col_size5, 3, 1)
        self.layer7 = ReConv_torch(batches, device, in_channels=256, out_channels=256, kernel_size=3, stride=1,
                                   col_size=col_size5,
                                   row_size=row_size5, weight_state_dict=state_dict['layer7.weight'],
                                   bias_state_dict=state_dict['layer7.bias'])
        row_size5, col_size5 = transfer(row_size5, col_size5, 3, 1)
        row_size5, col_size5 = transfer(row_size5, col_size5, 2, 2)
        self.layer8 = ReConv_torch(batches, device, in_channels=256, out_channels=512, kernel_size=3, stride=1,
                                   col_size=col_size5,
                                   row_size=row_size5, weight_state_dict=state_dict['layer8.weight'],
                                   bias_state_dict=state_dict['layer8.bias'])
        row_size5, col_size5 = transfer(row_size5, col_size5, 3, 1)
        self.layer9 = ReConv_torch(batches, device, in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                   col_size=col_size5,
                                   row_size=row_size5, weight_state_dict=state_dict['layer9.weight'],
                                   bias_state_dict=state_dict['layer9.bias'])
        row_size5, col_size5 = transfer(row_size5, col_size5, 3, 1)
        self.layer10 = ReConv_torch(batches, device, in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                    col_size=col_size5,
                                    row_size=row_size5, weight_state_dict=state_dict['layer10.weight'],
                                    bias_state_dict=state_dict['layer10.bias'])
        row_size5, col_size5 = transfer(row_size5, col_size5, 3, 1)
        row_size5, col_size5 = transfer(row_size5, col_size5, 2, 2)
        self.layer11 = ReConv_torch(batches, device, in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                    col_size=col_size5,
                                    row_size=row_size5, weight_state_dict=state_dict['layer11.weight'],
                                    bias_state_dict=state_dict['layer11.bias'])
        row_size5, col_size5 = transfer(row_size5, col_size5, 3, 1)
        self.layer12 = ReConv_torch(batches, device, in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                    col_size=col_size5,
                                    row_size=row_size5, weight_state_dict=state_dict['layer12.weight'],
                                    bias_state_dict=state_dict['layer12.bias'])
        row_size5, col_size5 = transfer(row_size5, col_size5, 3, 1)
        self.layer13 = ReConv_torch(batches, device, in_channels=512, out_channels=512, kernel_size=3, stride=1,
                                    col_size=col_size5,
                                    row_size=row_size5, weight_state_dict=state_dict['layer13.weight'],
                                    bias_state_dict=state_dict['layer13.bias'])
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        s = time.time()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.maxpool(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.maxpool(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.maxpool(x)
        e = time.time()
        print("Reconstructed Network time consumption: {:.6f}s".format(e - s))
        return x


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        # nn.BatchNorm2d(places, affine=False),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                # nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


def ResNet50():
    return ResNet([3, 4, 6, 3])


def ResNet101():
    return ResNet([3, 4, 23, 3])


def ResNet152():
    return ResNet([3, 8, 36, 3])