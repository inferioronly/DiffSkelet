import torch
import torch.nn as nn

from .resnet import SkeletResnetBlock2D

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(UpConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DoubleConv2d, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionGroup(nn.Module):
    def __init__(self, num_channels):
        super(AttentionGroup, self).__init__()
        self.conv1 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv_1x1 = nn.Conv2d(num_channels, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        s = torch.softmax(self.conv_1x1(x), dim=1)

        att = s[:,0,:,:].unsqueeze(1) * x1 + s[:,1,:,:].unsqueeze(1) * x2 \
            + s[:,2,:,:].unsqueeze(1) * x3

        return x + att


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=True),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=True),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out
    
class SkeletDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, dropout: float = 0.0,
                 resnet_eps: float = 1e-6, resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish", resnet_groups: int = 32, 
                 resnet_pre_norm: bool = True, use_pooling=True):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2) if use_pooling else None
        self.resnet = SkeletResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, 
                                         groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, 
                                         non_linearity=resnet_act_fn, pre_norm=resnet_pre_norm)
        self.attention = AttentionGroup(out_channels)

    def forward(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor):
        hidden_states = self.pooling(hidden_states) if self.pooling else hidden_states
        hidden_states = self.resnet(hidden_states, temb)
        if self.attention:
            hidden_states = self.attention(hidden_states)
        return hidden_states
        

class SkeletMidBlock(nn.Module):
    def __init__(self, in_channels: int, temb_channels: int, dropout: float = 0.0, resnet_eps: float = 1e-6, 
                 resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish", resnet_groups: int = 32, 
                 resnet_pre_norm: bool = True):
        super().__init__()

        blocks = []
        for _ in range(4):
            block = ResnetBlock(in_channels, 2)
            blocks.append(block)
        self.mid = nn.Sequential(*blocks)

        self.conv = SkeletResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, 
                                         groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, 
                                         non_linearity=resnet_act_fn, pre_norm=resnet_pre_norm)
        self.attn = AttentionGroup(in_channels)
    def forward(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor):

        hidden_states = self.mid(hidden_states)
        hidden_states = self.conv(hidden_states, temb)
        hidden_states = self.attn(hidden_states)

        return hidden_states


class SkeletUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, dropout: float = 0.0,
                 resnet_eps: float = 1e-6, resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish", resnet_groups: int = 32, 
                 resnet_pre_norm: bool = True, use_up=True):
        super().__init__()
        self.up = UpConv2d(in_channels, out_channels, kernel_size=2, stride=2) if use_up else None
        self.resnet = SkeletResnetBlock2D(in_channels=out_channels*2, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, 
                                         groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, 
                                         non_linearity=resnet_act_fn, pre_norm=resnet_pre_norm)
        self.channel_attn = ChannelAttention(out_channels)
        self.spatial_attn = SpatialAttention()

    def forward(self, hidden_states, temb, out, index):
        out = list(reversed(out))
        if self.up:
            hidden_states = self.up(hidden_states)
        hidden_states = torch.cat([hidden_states, out[index]], dim=1)
        hidden_states = self.resnet(hidden_states, temb)
        hidden_states = self.channel_attn(hidden_states) * hidden_states
        hidden_states = self.spatial_attn(hidden_states) * hidden_states

        return hidden_states