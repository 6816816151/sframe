# coding=UTF-8

import numpy as np


class SoftmaxLossLayer(object):
    def __init__(self):
        self.outputs = 0
        self.labels = np.zeros(0)

    def setup(self, inputs_shape):
        return 1

    def setLabels(self, labels):
        self.labels = labels

    # 计算模型的前向传播损失
    def forward(self, inputs):
        pass
