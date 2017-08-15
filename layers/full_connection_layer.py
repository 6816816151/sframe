# coding=UTF-8

import numpy as np


class FullConnectionLayer(object):
    def __init__(self, neuron_num):
        self.neuron_num = neuron_num

        self.weights = []
        self.outputs = []

    def setup(self, inputs_shape):
        # 使用均值为0、标准差为1的高斯分布初始化全连接层各神经元的权重
        self.weights = np.random.normal(0, 0.01, np.prod(inputs_shape[1:]) * self.neuron_num)

        # 根据输入数据计算当前层次输出结果的大小
        #
        # 输入数据以4维数组的形式组织，具体格式为：
        #     [batch, width, height, channel]
        #
        # 前一层为全连接层时，输入数据格式为：
        #     [batch, 1, 1, channel]
        self.outputs = np.zeros((inputs_shape[0], 1, 1, self.neuron_num))
        return self.outputs.shape

    # 计算前向传播结果
    def forward(self, inputs):
        pass
