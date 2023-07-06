'''
Code from https://raw.githubusercontent.com/facebookresearch/VideoPose3D/1afb1ca0f1237776518469876342fc8669d3f6a9/common/model.py
I added options to change number of out frames, and out dim
'''
#
import torch
import torch.nn as nn
from utils.constants import TRAINED_LANDMARKS

class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels, out_dim):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        self.out_dim = out_dim
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)

        shrink_out = num_joints_out * out_dim
        # shrink_out = 128

        self.shrink = nn.Conv1d(channels,shrink_out , 1)
        # self.conv_out = nn.Conv2d(channels, )

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = self._forward_blocks(x)
        # print(x.shape)
        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, self.out_dim)
        x = x.mean(dim=1)
        return x


class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False, out_dim=32):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout,
                         channels, out_dim)

        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        return x


class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, out_dim=32):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels, out_dim)

        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0],
                                     bias=False)

        # self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0], stride=filter_widths[0],
        #                              bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)

            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2:: self.filter_widths[i + 1]]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, in_chans, embed_dim, channels=64, filter_widths=None, dense=False):
        super().__init__()
        self.img_size = img_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.embed_channles = channels
        if filter_widths is None:
            self.filter_widths = [3, 3, 3]
        else:
            self.filter_widths = filter_widths

        self.n_frames, self.n_landmarks = img_size
        if dense:
            self.tdcnn_model = TemporalModel(self.n_landmarks, in_features=self.in_chans,
                                             num_joints_out=self.n_landmarks,
                                             filter_widths=self.filter_widths, causal=False, dropout=0.25,
                                             channels=self.embed_channles,
                                             out_dim=self.embed_dim, dense=True)

        else:
            self.tdcnn_model = TemporalModelOptimized1f(self.n_landmarks, in_features=self.in_chans,
                                                        num_joints_out=self.n_landmarks,
                                                        filter_widths=self.filter_widths, causal=False, dropout=0.25,
                                                        channels=self.embed_channles,
                                                        out_dim=self.embed_dim)

    def forward(self, x):
        return self.tdcnn_model(x)


if __name__ == '__main__':
    # n_frames = 32 * 2

    n_frames = 32
    # n_landmarks = 478
    n_landmarks = len(TRAINED_LANDMARKS)
    dims = 2
    test_input = torch.rand(1, n_frames, n_landmarks, dims)
    filter_widths = [3, 3, 3]
    dropout = 0.25
    channels = 64
    out_dim = 32
    out_frames = 1
    # model = TemporalModel(num_joints_in=n_landmarks, in_features=2, num_joints_out=n_landmarks,
    #              filter_widths=filter_widths, causal=False, dropout=0.25, channels=channels, dense=False,
    #                       out_dim=out_dim)

    # model = TemporalModelOptimized1f(num_joints_in=n_landmarks, in_features=2, num_joints_out=n_landmarks,
    #                                  filter_widths=filter_widths, causal=False, dropout=0.25, channels=channels,
    #                                  out_dim=out_dim)

    # ret = model(test_input)
    # print(ret.shape)
    # print(ret)

    ####
    img_size = (n_frames, n_landmarks)
    in_chans = dims
    embed_dim = out_dim
    model = PatchEmbed(img_size, in_chans, embed_dim, channels=128, filter_widths=[3, 3, 3], dense=True)
    x = model(test_input)
    print(test_input.shape)
    print(x.shape)
