# coding=UTF-8

import os
import numpy as np
from sframe.dataset.imdb import loadImageSet, loadLabelSet
from sframe.configuration.classifier import get_LeNet_model

# 加载训练数据
minist_image_path = '/Users/dawner/Documents/MNIST dataset/train-images.idx3-ubyte'
minist_label_path = '/Users/dawner/Documents/MNIST dataset/train-labels-idx1-ubyte'

images = loadImageSet(minist_image_path)
labels = loadLabelSet(minist_label_path)

# 获得LeNet卷积神经网络结构
classifier = get_LeNet_model(images, labels)

# 启动模型训练
classifier.train()

# 加载测试数据
# minist_test_image_path = '/Users/dawner/Documents/MNIST dataset/t10k-images-idx3-ubyte'
# minist_test_label_path = '/Users/dawner/Documents/MNIST dataset/t10k-labels-idx1-ubyte'
#
# test_images = loadImageSet(minist_test_image_path)
# test_labels = loadLabelSet(minist_test_label_path)

# 测试模型准确率
# classifier.test(test_images, test_labels)