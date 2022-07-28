import mindspore.nn as nn

class DecomNet(nn.Cell):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.net_shallow = nn.Conv2d(3, channel, kernel_size * 3, pad_mode='pad', padding=4)
        self.net_deep = nn.SequentialCell([ nn.Conv2d(channel, channel, kernel_size, pad_mode='pad', padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(channel, channel, kernel_size, pad_mode='pad', padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(channel, channel, kernel_size, pad_mode='pad', padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(channel, channel, kernel_size, pad_mode='pad', padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(channel, channel, kernel_size, pad_mode='pad', padding=1)])
        self.net_recon =   nn.Conv2d(channel, 6, kernel_size, pad_mode='pad', padding=1)

    def construct(self, input):
        feat_shallow    = self.net_shallow(input)
        feat_deep       = self.net_deep(feat_shallow)
        feat_output_put = self.net_recon(feat_deep)
        BRDF            = nn.Sigmoid(feat_output_put[:, :3, :, :])
        F_delta         = nn.Sigmoid(feat_output_put[:, 3:, :, :])
        return BRDF, F_delta
