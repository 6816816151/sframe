# coding=UTF-8

import numpy as np


class ReLULayer(object):
    def __init__(self):
        self.outputs = []

    def setup(self, inputs_shape):
        # 根据输入数据计算当前层次输出结果的大小
        #
        # 输入数据以4维数组的形式组织，具体格式为：
        #     [batch, width, height, channel]
        self.outputs = np.zeros(inputs_shape)
        return self.outputs.shape

    # 计算前向传播结果，返回max(x, 0)
    def forward(self, inputs):
        return np.maximum(inputs, 0)
