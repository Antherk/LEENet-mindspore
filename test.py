from decom import DecomNet
from tools import weights_init
import mindspore.nn as nn 


if __name__=='__main__':
    net = nn.Conv2d(3, 3, 2, pad_mode='pad', padding=1)
    print(id(net.weight))
    for m in net.cells_and_names():
        print(id(m[1].weight))
        weights_init(m[1])
    
