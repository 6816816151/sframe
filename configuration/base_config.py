# coding=UTF-8

class BaseConfig(object):
    # 基本学习率
    base_lr = 0.01

    # 最大迭代次数
    max_iters = 20000

    # minibatch大小
    batch_size = 64

    # 模型默认大小
    img_width = 28
    img_height = 28
