import mindspore.nn as nn
from mindspore import ops


class AjustNet(nn.Cell):
    def __init__(self, channel=64, kernel_size=3):
        super(AjustNet, self).__init__()
        self.relu       = nn.ReLU()
        
        self.conv0_1    = nn.Conv2d(3, channel, kernel_size, pad_mode='pad', padding=1)

        self.conv1_1    = nn.Conv2d(channel, channel, kernel_size, stride=2, pad_mode='pad', padding=1)
        self.conv1_2    = nn.Conv2d(channel, channel, kernel_size, stride=2, pad_mode='pad', padding=1)
        self.conv1_3    = nn.Conv2d(channel, channel, kernel_size, stride=2, pad_mode='pad', padding=1)
        
        self.deconv1_1  = nn.Conv2d(channel*2, channel, kernel_size, pad_mode='pad', padding=1)
        self.deconv1_2  = nn.Conv2d(channel*2, channel, kernel_size, pad_mode='pad', padding=1)
        self.deconv1_3  = nn.Conv2d(channel*2, channel, kernel_size, pad_mode='pad', padding=1)
        
        self.fusion     = nn.Conv2d(channel*3, channel, 1, pad_mode='pad', padding=1)
        self.output_put = nn.Conv2d(channel, 3, kernel_size, pad_mode='pad', padding=0)

    def construct(self, I, delta):
        input           = ops.OnesLike(I)

        for i in range(I.shape[0]):
            input[i,:,:,:] = I[i,:,:,:]*delta[i]
        output_0        = self.conv0_1(input)
        output_1        = self.relu(self.conv1_1(output_0))
        output_2        = self.relu(self.conv1_2(output_1))
        output_3        = self.relu(self.conv1_3(output_2))
        
        output_1_up     = nn.ResizeBilinear(output_3,size=(output_2.shape[2], output_2.shape[3]))
        deconv1         = self.relu(self.deconv1_1(ops.Concat((output_1_up, output_2), axis=1)))
        output_2_up     = nn.ResizeBilinear(deconv1,size=(output_1.shape[2], output_1.shape[3]))
        deconv2         = self.relu(self.deconv1_2(ops.Concat((output_2_up, output_1), axis=1)))
        output_3_up     = nn.ResizeBilinear(deconv2,size=(output_0.shape[2], output_0.shape[3]))
        deconv3         = self.relu(self.deconv1_3(ops.Concat((output_3_up, output_0), axis=1)))


        deconv1_rs      = nn.ResizeBilinear(deconv1,size=(I.shape[2], I.shape[3]))
        deconv2_rs      = nn.ResizeBilinear(deconv2,size=(I.shape[2], I.shape[3]))
        feats_concate   = ops.Concat((deconv1_rs, deconv2_rs, deconv3), axis=1)
        
        feats_fusion    = self.fusion(feats_concate)
        output          = self.output_put(feats_fusion)
        
        return output