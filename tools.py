import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import initializer, XavierUniform, HeNormal, Constant


def weights_init(m):
    if isinstance(m, nn.Dense):
        init = initializer(XavierUniform(), m.weight.shape, mindspore.float32)
        m.weight = init
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        init = initializer(HeNormal(), m.weight.shape, mindspore.float32)
        m.weight = init
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        init_w = initializer(Constant(), m.weight.shape, mindspore.float32)
        init_b = initializer(Constant(), m.bias.shape, mindspore.float32)
        m.weight = init_w
        m.bias = init_b
        #nn.init.constant_(m.weight, 1)
        #nn.init.constant_(m.bias, 0)
    