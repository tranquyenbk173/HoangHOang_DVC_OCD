# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes, bias=True):
        super(MLP, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):

            if i < (len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))

                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.n_features = 0
        self._name = "BaseModule"

    def forward(self, x):
        return x

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def init_weights(self, std=0.01):
        print("Initialize weights of %s with normal dist: mean=0, std=%0.2f" % (type(self), std))
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif type(m) == nn.Conv2d:
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    m.bias.data.zero_()

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        print("BIAS IS", bias)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.contiguous().view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.linear(x)
        return x

    def forward_DVC(self, x):
        out = self.features(x)
        logits = self.logits(out)
        return logits,
    
class QNet(BaseModule):
    def __init__(self,
                 n_units,
                 n_classes):
        super(QNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2 * n_classes, n_units),
            nn.ReLU(True),
            nn.Linear(n_units, n_classes),
        )

    def forward(self, zcat):
        zzt = self.model(zcat)
        return zzt


class DVCNet(BaseModule):
    def __init__(self,
                 backbone,
                 n_units,
                 n_classes,
                 has_mi_qnet=True):
        super(DVCNet, self).__init__()

        self.backbone = backbone
        self.has_mi_qnet = has_mi_qnet

        if has_mi_qnet:
            self.qnet = QNet(n_units=n_units,
                             n_classes=n_classes)

    def forward(self, x, xt):
        size = x.size(0)
        xx = torch.cat((x, xt))
        zz,fea = self.backbone.forward_DVC(xx)
        z = zz[0:size]
        zt = zz[size:]

        fea_z = fea[0:size]
        fea_zt = fea[size:]

        if not self.has_mi_qnet:
            return z, zt, None

        zcat = torch.cat((z, zt), dim=1)
        zzt = self.qnet(zcat)

        return z, zt, zzt,[torch.sum(torch.abs(fea_z), 1).reshape(-1, 1),torch.sum(torch.abs(fea_zt), 1).reshape(-1, 1)]


def Reduced_ResNet18_DVC(nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    backnone = ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)
    return DVCNet(backbone=backnone,n_units=128,n_classes=nclasses,has_mi_qnet=True)


def ResNet18(nclasses, nf=20, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)
