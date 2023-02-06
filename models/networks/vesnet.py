import torch
import torch.nn as nn
from models.modules.convgru import ConvGRU
from models.networks.utils import MultiAttentionBlock
from models.modules.cbam import ChannelGate
from torch.autograd import Variable
import numpy as np

from models.networks_other import init_weights


def batch_norm_and_prelu(filters):
    return nn.Sequential(
        nn.BatchNorm2d(num_features=filters),
        nn.PReLU()
    )


def conv_batch_norm_prelu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=5, padding=2),
        nn.modules.BatchNorm2d(num_features=out_channels),
        nn.PReLU()
    )


def conv2_prelu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=2, stride=2),
        nn.PReLU()
    )


class SpatialChannelAttentionModule(nn.Module):
    def __init__(self, filters):
        super(SpatialChannelAttentionModule, self).__init__()
        self.filters = filters
        self.attentionMode = 'concatenation'
        self.attention_dsample = (2, 2)
        self.selfSpatialAttentionHigherResolutionSpace = ChannelGate(
            gate_channels=self.filters[0])
        self.multiScaleSpatialAttention = MultiAttentionBlock(in_size=self.filters[0], gate_size=self.filters[1],
                                                              inter_size=self.filters[0],
                                                              nonlocal_mode=self.attentionMode,
                                                              sub_sample_factor=self.attention_dsample)
        self.selfSpatialAttentionLowerResolutionSpace = ChannelGate(
            gate_channels=self.filters[1])

    def forward(self, input, gating_signal):
        x_A = self.selfSpatialAttentionHigherResolutionSpace(input)
        g_A = self.selfSpatialAttentionLowerResolutionSpace(gating_signal)
        return self.multiScaleSpatialAttention(x_A, g_A)[0]


class Encoder(nn.Module):

    def __init__(self, filters):
        super(Encoder, self).__init__()
        self.filters = filters
        self.encoder_1 = conv_batch_norm_prelu(self.filters, self.filters)
        self.encoder_2 = nn.Conv2d(
            self.filters, self.filters, kernel_size=5, padding=2)
        self.encoder_3 = batch_norm_and_prelu(self.filters)

    def forward(self, x):
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        return self.encoder_3(torch.add(x, x2))


class Decoder(nn.Module):

    def __init__(self, filters):
        super(Decoder, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2d(
            self.filters * 2, self.filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.filters, self.filters,
                               kernel_size=3, padding=1)
        self.batchNormAndPrelu = batch_norm_and_prelu(self.filters)

    def forward(self, input_skipConnection, input_UpSampling):
        x1 = self.conv1(
            torch.cat([input_skipConnection, input_UpSampling], dim=1))
        x2 = self.conv2(x1)
        return self.batchNormAndPrelu(torch.add(input_UpSampling, x2))


class ResizeUpConvolution(nn.Module):

    def __init__(self, filters):
        super(ResizeUpConvolution, self).__init__()
        self.filters = filters
        self.resizeUp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=self.filters[1], out_channels=self.filters[0], kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = self.resizeUp(x)
        return self.conv(x)


class VesNet(nn.Module):

    def __init__(self, in_channels=2, feature_scale=8):
        super(VesNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.bs = -1
        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor  # computation in GPU
            self.device = torch.device('cuda')
        else:
            self.dtype = torch.FloatTensor
            self.device = torch.device('cpu')

        self.filters = [64, 128, 256, 512]
        self.filters = [int(x / self.feature_scale) for x in self.filters]

        self.imagePrep = conv_batch_norm_prelu(self.in_channels, self.filters[0])
        self.encoder1 = Encoder(self.filters[0])
        self.pool1 = conv2_prelu(self.filters[0], self.filters[1])

        self.encoder2 = Encoder(self.filters[1])
        self.pool2 = conv2_prelu(self.filters[1], self.filters[2])

        self.encoder3 = Encoder(self.filters[2])
        self.pool3 = conv2_prelu(self.filters[2], self.filters[3])

        self.encoder4 = Encoder(self.filters[3])

        # skip connections with Conv GRU
        self.convGru1 = ConvGRU(input_size=self.filters[0], hidden_sizes=[self.filters[0]],
                                      kernel_sizes=[3], n_layers=1)
        self.convGru2 = ConvGRU(input_size=self.filters[1], hidden_sizes=[self.filters[1]],
                                      kernel_sizes=[1], n_layers=1)
        self.spatialChannelAttention3 = SpatialChannelAttentionModule(
            filters=self.filters[2:4])
        self.convGru3 = ConvGRU(input_size=self.filters[2], hidden_sizes=[self.filters[2]],
                                kernel_sizes=[3], n_layers=1)
        self.spatialChannelAttention2 = SpatialChannelAttentionModule(
            filters=self.filters[1:3])
        self.convGru4 = ConvGRU(input_size=self.filters[3], hidden_sizes=[self.filters[3]],
                                kernel_sizes=[3], n_layers=1)
        self.spatialChannelAttention1 = SpatialChannelAttentionModule(
            filters=self.filters[:2])

        self.resizeUp4 = ResizeUpConvolution(filters=self.filters[2:4])

        self.decoder3 = Decoder(self.filters[2])
        self.resizeUp3 = ResizeUpConvolution(filters=self.filters[1:3])

        self.decoder2 = Decoder(self.filters[1])
        self.resizeUp2 = ResizeUpConvolution(filters=self.filters[:2])

        self.decoder1 = Decoder(self.filters[0])

        self.conv_out = nn.Conv2d(self.filters[0], 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, input, hidden):

        new_hidden = [None] * 4
        if hidden is None:
            hidden = self.init_hidden(input.size(0), input.size(2))
        # encoding path
        resImagePrep = self.imagePrep(input)

        resEncoder1 = self.encoder1(resImagePrep)
        resPool1 = self.pool1(resEncoder1)

        resEncoder2 = self.encoder2(resPool1)
        resPool2 = self.pool2(resEncoder2)

        resEncoder3 = self.encoder3(resPool2)
        resPool3 = self.pool3(resEncoder3)

        resEncoder4 = self.encoder4(resPool3)

        # intermediate Steps
        # temporal attention units
        # returns output and hidden state in two lists
        resConvGru4 = self.convGru4(resEncoder4, hidden[3])
        resConvGru3 = self.convGru3(resEncoder3, hidden[2])
        resConvGru2 = self.convGru2(resEncoder2, hidden[1])
        resConvGru1 = self.convGru1(resEncoder1, hidden[0])

        # store for next calculation
        new_hidden[3] = resConvGru4
        new_hidden[2] = resConvGru3
        new_hidden[1] = resConvGru2
        new_hidden[0] = resConvGru1
        # decoder
        resSpatialChAtt3 = self.spatialChannelAttention3(
            resConvGru3[-1], resConvGru4[-1])
        resUp4 = self.resizeUp4(resConvGru4[-1])
        resDecoder3 = self.decoder3(resSpatialChAtt3, resUp4)

        resSpatialChAtt2 = self.spatialChannelAttention2(
            resConvGru2[-1], resDecoder3)
        resUp3 = self.resizeUp3(resConvGru3[-1])
        resDecoder2 = self.decoder2(resSpatialChAtt2, resUp3)

        resSpatialChAtt1 = self.spatialChannelAttention1(
            resConvGru1[-1], resDecoder2)
        resUp2 = self.resizeUp2(resConvGru2[-1])
        resDecoder1 = self.decoder1(resSpatialChAtt1, resUp2)

        return self.conv_out(resDecoder1), new_hidden

    @staticmethod
    def apply_sigmoid(pred):
        log_p = torch.sigmoid(pred)

        return log_p

    def init_hidden(self, batch_size, input_size):
        hidden_states = [None, None, None, None]
        return hidden_states


if __name__ == '__main__':
    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor  # computation in GPU
        device = torch.device('cuda')
    else:
        dtype = torch.FloatTensor
        device = device = torch.device('cpu')

    # batch size, sequence length (how many previous images are we putting in), channels, width, height
    image = torch.rand((5, 4, 2, 320, 320), device=device)

    model = VesNet().to(device)

    model(image)
