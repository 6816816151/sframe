# coding=UTF-8

from sframe.layers.convolution_layer import ConvolutionLayer
from sframe.layers.max_pooling_layer import MaxPoolingLayer
from sframe.layers.full_connection_layer import FullConnectionLayer
from sframe.layers.relu_layer import ReLULayer
from sframe.layers.softmax_loss_layer import SoftmaxLossLayer
from sframe.configuration.base_config import BaseConfig
from sframe.dataset.minibatch import MiniBatch
import numpy as np


class Classifier(object):
    def __init__(self, images, labels):
        # 当前模型所含的网络结构
        self.layers = []

        self.base_config = BaseConfig()
        self.minibatch = MiniBatch(images, labels, self.base_config.batch_size)

    # 在当前网络结构的末尾插入一层
    def append_layer(self, layer):
        self.layers.append(layer)

    # 初始化各层权重，确定输入数据与输出数据的规模
    def setup(self):
        inputs_shape = np.array([self.base_config.batch_size, self.base_config.img_width,
                                 self.base_config.img_height, 1])

        for i in range(len(self.layers)):
            inputs_shape = self.layers[i].setup(inputs_shape)

    # 根据给定训练数据训练模型
    def train(self):
        self.setup()

        # 获得当前迭代的minibatch
        inputs, label_batches = self.minibatch.get_train_batch()

        # 更新Softmax损失层中minibatch的样本标签
        self.layers[-1].setLabels(label_batches)

        # 在网络中前向传播数据
        for i in range(len(self.layers)):
            inputs = self.layers[i].forward(inputs)

    def test(self, images, labels):
        pass


# 基于LeNet网络结构，构建深度学习模型
def get_LeNet_model(images, labels):
    classifier = Classifier(images, labels)

    # 逐层构建LeNet卷积网络
    classifier.append_layer(ConvolutionLayer(filter_size=5, stride=1, filter_num=20))
    classifier.append_layer(MaxPoolingLayer(filter_size=2, stride=2))

    classifier.append_layer(ConvolutionLayer(filter_size=5, stride=1, filter_num=50))
    classifier.append_layer(MaxPoolingLayer(filter_size=2, stride=2))

    classifier.append_layer(FullConnectionLayer(neuron_num=500))
    classifier.append_layer(ReLULayer())
    classifier.append_layer(FullConnectionLayer(neuron_num=10))

    classifier.append_layer(SoftmaxLossLayer())

    return classifier
