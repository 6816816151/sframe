# coding=UTF-8

import numpy as np


class MiniBatch(object):
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

        # images_num记录训练数据集中的图像总数
        # remain记录当前剩余的样本batch
        self.images_num = self.images.shape[0]
        self.remain = self.organize_simple_batch()

    # 以batch_size为大小，划分训练数据集
    #
    # 返回结果的格式为：
    #     [n, batch_size]
    def organize_simple_batch(self):
        shuffled_ids = np.random.permutation(self.images_num)
        total_batch = self.images_num / self.batch_size

        return shuffled_ids[self.images_num % self.batch_size:].reshape(total_batch, self.batch_size)

    # 返回当前模型训练所需样本的id
    def next_batch(self):
        if self.remain.shape[0] == 0:
            self.remain = self.organize_simple_batch()

        batch_ids = self.remain[-1, :]
        self.remain = self.remain[:-1, :]

        return batch_ids

    # 返回当前模型训练所需的minibatch
    def get_train_batch(self):
        batch_ids = self.next_batch()
        img_batches = self.images[batch_ids, :, :] \
            .reshape(self.batch_size, self.images.shape[1], self.images.shape[2], 1)
        label_batches = self.labels[batch_ids]

        return img_batches, label_batches
