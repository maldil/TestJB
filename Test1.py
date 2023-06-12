def __init__(self, in_ch, num_filter, bias=False):
    super(FeatureExtrator, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=num_filter // 2,
                           kernel_size=7, dilation=(1, 1), padding=3, bias=bias)
    self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=num_filter // 2,
                           kernel_size=7, dilation=(2, 2), padding=6, bias=bias)
    self.conv3 = nn.Conv2d(in_channels=in_ch, out_channels=num_filter // 2,
                           kernel_size=7, dilation=(4, 4), padding=12, bias=bias)

    self.bn1 = nn.BatchNorm2d(int(1.5 * num_filter))
    self.bn2 = nn.BatchNorm2d(num_filter)
    self.relu = nn.ReLU(inplace=False)

    self.conv4 = nn.Conv2d(in_channels=int(1.5 * num_filter), out_channels=num_filter,
                           kernel_size=7, stride=1, padding=3, bias=bias)
    self.conv5 = nn.Conv2d(in_channels=in_ch, out_channels=num_filter,
                           kernel_size=1, stride=1, padding=0, bias=bias)
