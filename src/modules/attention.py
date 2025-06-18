import torch
from torch import nn
import torch.nn.functional as F

class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # nn.ReLU(inplace=True),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class ChannelAttnBlock(nn.Module):
    """
    Channel Attention Block.

    Parameters:
        in_channels (:obj:`int`): The number of channels in the input and output.
        out_channels (:obj:`int`): The number of channels in the output.
        groups (:obj:`int`, *optional*, defaults to 32): The number of groups to use for group normalization.
    """
    def __init__(self, in_channels, out_channels, groups=32, groups_out=None, eps=1e-6, non_linearity="swish", channel_attn=False, reduction=32):
        super().__init__()

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        
        self.channel_attn = channel_attn
        if self.channel_attn:
            self.se_channel_attn = SELayer(channel=in_channels, reduction=reduction)

        # Down channel: Use the conv1*1 to down the channel wise
        self.norm3 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.down_channel = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) # conv1*1

    def forward(self, hidden_states):
        input_feature = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.channel_attn:
            hidden_states = self.se_channel_attn(hidden_states)
            hidden_states = hidden_states + input_feature

        # Down channel
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.down_channel(hidden_states)

        return hidden_states
