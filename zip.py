import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed_b, pe_embed_l):
        super(PositionalEncoding, self).__init__()
        if pe_embed_b == 0:
            self.embed_length = 1
            self.pe_embed = False
        else:
            self.lbase = float(pe_embed_b)
            self.levels = float(pe_embed_l)
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels
            self.pe_embed = True

    def __repr__(self):
        return f"Positional Encoder: pos_b={self.lbase}, pos_l={self.levels}, embed_length={self.embed_length}, to_embed={self.pe_embed}"

    def forward(self, pos):
        if self.pe_embed is False:
            return pos[:, None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase ** (i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1)


def MLP(dim_list, act='ReLU', bias=True):
    act = getattr(torch.nn, act)()
    mlp_list = []
    for i in range(len(dim_list) - 1):
        mlp_list += [nn.Linear(dim_list[i], dim_list[i + 1], bias=bias), act]
    return nn.Sequential(*mlp_list)


class CustomConv2d(nn.Module):
    def __init__(self, conv_type: str, in_channels: int, out_channels: int, stride: int, bias: bool):
        super(CustomConv2d, self).__init__()

        self.conv_type = conv_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bias = bias

        if self.conv_type == 'pixelshuffle':
            self.op0 = nn.Conv2d(in_channels, out_channels * stride * stride, 3, 1, 1, bias=bias)
            self.op1 = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.op0 = nn.ConvTranspose2d(in_channels, out_channels, stride, stride)
            self.op1 = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.op0 = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.op1 = nn.Conv2d(in_channels, out_channels, 2 * stride + 1, 1, stride, bias=bias)
        elif self.conv_type == 'deconv_conv':
            self.op0 = nn.ConvTranspose2d(in_channels, out_channels, stride, stride)
            self.op1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        else:
            raise NotImplementedError(f"conv_type: {conv_type} is not implemented.")

    def __repr__(self):
        return f"CustomConv2d(conv_type={self.conv_type}, in_channels={self.in_channels}, out_channels={self.out_channels}, stride={self.stride}, bias={self.bias})"

    def forward(self, x):
        return self.op1(self.op0(x))


class Block(nn.Module):
    def __init__(self, conv_type: str, in_channels: int, out_channels: int, stride: int, bias: bool, norm: str, act: str):
        super().__init__()

        self.conv = CustomConv2d(conv_type=conv_type, in_channels=in_channels, out_channels=out_channels, stride=stride, bias=bias)
        self.act = getattr(torch.nn, act)()
        self.norm = getattr(torch.nn, norm)()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvUpBlock(nn.Module):
    def __init__(self, conv_type: str, in_channels: int, out_channels: int, stride: int, bias: bool, norm: str, act: str):
        super().__init__()

        if in_channels <= out_channels:
            factor = 4
            self.conv1 = CustomConv2d(conv_type=conv_type, in_channels=in_channels, out_channels=in_channels // factor, stride=stride, bias=bias)
            self.conv2 = nn.Conv2d(in_channels // factor, out_channels, 3, 1, 1, bias=bias)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
            self.conv2 = CustomConv2d(conv_type=conv_type, in_channels=out_channels, out_channels=out_channels, stride=stride, bias=bias)

        self.act = getattr(torch.nn, act)()
        self.norm = getattr(torch.nn, norm)()

    def forward(self, x):
        return self.act(self.norm(self.conv2(self.conv1(x))))


class ZIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # temporal model
        self.pe = PositionalEncoding(
            pe_embed_b=cfg['pos_b'], pe_embed_l=cfg['pos_l']
        )

        stem_dim_list = [int(x) for x in cfg['stem_dim'].split('_')]
        self.feat_h, self.feat_w, self.feat_c = [int(x) for x in cfg['feat_hwc'].split('_')]
        # self.block_dim = cfg['block_dim']

        stem_mlp_dim_list = [self.pe.embed_length] + stem_dim_list + [self.feat_h * self.feat_w * self.feat_c]
        self.stem = MLP(dim_list=stem_mlp_dim_list, act=cfg['act'])

        # feat up

        conv_layers, norm_layers = [nn.ModuleList() for _ in range(2)]
        in_channels = self.feat_c
        for i, stride in enumerate(cfg['stride_list']):
            if i == 0:
                # expand channel width at first stage
                out_channels = int(in_channels * cfg['expansion'])
            else:
                # change the channel width for each stage
                out_channels = max(in_channels // (1 if stride == 1 else cfg['reduction']), cfg['lower_width'])

            norm_layers.append(nn.InstanceNorm2d(in_channels, affine=False))

            if i == 0:
                conv_layers.append(
                    ConvUpBlock(conv_type=cfg['conv_type'], in_channels=in_channels, out_channels=out_channels, stride=stride,
                                bias=cfg['bias'], norm=cfg['norm'], act=cfg['act']))
            else:
                conv_layers.append(
                    Block(in_channels=in_channels, out_channels=out_channels, stride=stride, bias=cfg['bias'],
                          norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
            in_channels = out_channels

        # to feat up
        decoder = []
        for norm_layer, conv_layer in zip(norm_layers, conv_layers):
            decoder.append(norm_layer)
            decoder.append(conv_layer)

        self.decoder = nn.Sequential(*decoder)

        # out
        self.out_conv = nn.Conv2d(in_channels, cfg['out_channel'], 1, 1, bias=cfg['bias'])
        self.out_act = getattr(torch.nn, cfg['out_act'])()

    def forward(self, idx):
        """
        :param idx: temporal indices, [N, 1]
        :return: reconstruct image, shape: [N, C, H, W]
        """
        emb = self.stem(self.pe(idx))  # [N, 1] -> [N, L] -> [N, chw]
        emb = emb.reshape(emb.shape[0], self.feat_c, self.feat_h, self.feat_w)  # [N, c, h, w]
        emb = self.decoder(emb)
        emb = self.out_conv(emb)
        emb = self.out_act(emb)
        return emb