# coding=UTF-8

import numpy as np
import struct


def loadImageSet(minist_image_path):
    binfile = open(minist_image_path, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width, height])
    return imgs


def loadLabelSet(minist_label_path):
    binfile = open(minist_label_path, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    return labels
