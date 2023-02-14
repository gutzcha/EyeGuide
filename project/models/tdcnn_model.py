import torch
import torch.nn as nn


class TDCNN(nn.Module):
    def __init__(self, filter_widths, num_features, dropout=0.25, channels=64):
        super().__init__()

        for fw in filter_widths:
            assert fw % 2 != 0, 'Please input an odd filter width'

        self.num_features = num_features
        self.filter_widths = filter_widths
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.sigmoid = nn.Sigmoid()
        self.expand_conv = nn.Conv1d(num_features * 2, channels, filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)

            layers_conv.append(nn.Conv1d(channels,
                                         channels,
                                         filter_widths[i],
                                         dilation=next_dilation,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.conv_layers = nn.ModuleList(layers_conv)
        self.bn_layers = nn.ModuleList(layers_bn)
        self.eval()

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum


    def forward(self, x):
        x = x.type(torch.float32)

        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            res = x[:, :, pad: x.shape[2] - pad]

            x = self.dropout(self.relu(self.bn_layers[2 * i](self.conv_layers[2 * i](x))))
            x = res + self.dropout(self.relu(self.bn_layers[2 * i + 1](self.conv_layers[2 * i + 1](x))))

        return x

# class PatchEmbed(nn.Module):
#     """
#     Embed patches using tdcnn
#     """
#     def __len__(self, n_landmarks, in_chans, conv_layers, embed_dim):
#

if __name__ == '__main__':
    n_frames = 32
    n_landmarks = 478
    dims = 2
    test_input = torch.rand(1,n_frames,n_landmarks*dims)
    filter_widths = [3, 3, 3, 3]
    dropout = 0.25
    channels = 64
    model = TDCNN(filter_widths=filter_widths, num_features=n_landmarks,channels=channels)
    ret = model(test_input)
    print(ret)