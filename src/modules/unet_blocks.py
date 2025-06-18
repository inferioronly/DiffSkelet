import torch
from torch import nn

from .attention import ChannelAttnBlock
from .resnet import (Downsample2D, ResnetBlock2D, Upsample2D)


def get_down_block(down_block_type, num_layers, in_channels, out_channels, temb_channels, add_downsample, 
                   resnet_eps, resnet_act_fn, attn_num_head_channels, resnet_groups=None,
                   downsample_padding=None, channel_attn=False, reduction=32):

    if down_block_type == "DownBlock2D":
        return DownBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, 
                           add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, 
                           resnet_groups=resnet_groups, downsample_padding=downsample_padding)
    elif down_block_type == "SkeletDownBlock2D":
        return SkeletDownBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, channel_attn=channel_attn, 
                              temb_channels=temb_channels, add_downsample=add_downsample, resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn, 
                              resnet_groups=resnet_groups, downsample_padding=downsample_padding, 
                              attn_num_head_channels=attn_num_head_channels, reduction=reduction)
    else:
        raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(up_block_type, num_layers, in_channels, out_channels, prev_output_channel, temb_channels, add_upsample, resnet_eps,
                 resnet_act_fn, resnet_groups=None):

    if up_block_type == "UpBlock2D":
        return UpBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, 
                         temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, 
                         resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups)
    elif up_block_type == "SkeletUpBlock2D":
        return SkeletUpBlock2D(num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, prev_output_channel=prev_output_channel, 
                         temb_channels=temb_channels, add_upsample=add_upsample, resnet_eps=resnet_eps, 
                         resnet_act_fn=resnet_act_fn, resnet_groups=resnet_groups)
    else:
        raise ValueError(f"{up_block_type} does not exist.")



############## all channel attention

class UNetMidSkeletBlock2D(nn.Module):
    def __init__(self, in_channels: int, temb_channels: int, channel_attn: bool = False, dropout: float = 0.0, num_layers: int = 1, resnet_eps: float = 1e-6,
                 resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish", resnet_groups: int = 32, resnet_pre_norm: bool = True, 
                 attn_num_head_channels=1, attention_type="default", reduction=32, **kwargs):
        super().__init__()

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnets = [ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, 
                                 groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, 
                                 non_linearity=resnet_act_fn, pre_norm=resnet_pre_norm)]

        attentions = []

        for _ in range(num_layers):
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=temb_channels, eps=resnet_eps, 
                                         groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, 
                                         non_linearity=resnet_act_fn, pre_norm=resnet_pre_norm))
            attentions.append(ChannelAttnBlock(in_channels=in_channels, out_channels=in_channels, 
                                               non_linearity=resnet_act_fn, channel_attn=channel_attn, reduction=reduction))

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet, attn in zip(self.resnets[1:], self.attentions):
            
            # self attention
            hidden_states = attn(hidden_states)
            # t_embed
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class SkeletDownBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, dropout: float = 0.0, channel_attn: bool = False, num_layers: int = 1, 
                 resnet_eps: float = 1e-6, resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish", resnet_groups: int = 32, 
                 resnet_pre_norm: bool = True, attn_num_head_channels=1, attention_type="default", downsample_padding=1, 
                 add_downsample=True, reduction=32):
        super().__init__()
        attentions = []
        resnets = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            attentions.append(ChannelAttnBlock(in_channels=in_channels, out_channels=in_channels, groups=resnet_groups, 
                                                       non_linearity=resnet_act_fn, channel_attn=channel_attn, reduction=reduction))
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, 
                                         groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, 
                                         non_linearity=resnet_act_fn, pre_norm=resnet_pre_norm))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if num_layers == 1:
            in_channels = out_channels
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op")])
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for attn, resnet in zip(self.attentions, self.resnets):
            
            # self attention
            hidden_states = attn(hidden_states)
            # t_embed
            hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)


        return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, dropout: float = 0.0, num_layers: int = 1, 
                 resnet_eps: float = 1e-6, resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish", 
                 resnet_groups: int = 32, resnet_pre_norm: bool = True, add_downsample=True, downsample_padding=1):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, eps=resnet_eps, 
                                         groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift, 
                                         non_linearity=resnet_act_fn, pre_norm=resnet_pre_norm))

        self.resnets = nn.ModuleList(resnets)

        if num_layers == 1:
            in_channels = out_channels
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(in_channels, use_conv=True, out_channels=out_channels, 
                                                            padding=downsample_padding, name="op")])
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

class UpBlock2D(nn.Module):
    def __init__(self, in_channels: int, prev_output_channel: int, out_channels: int, temb_channels: int, dropout: float = 0.0, num_layers: int = 1,
                 resnet_eps: float = 1e-6, resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish", resnet_groups: int = 32,
                 resnet_pre_norm: bool = True, add_upsample=True):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_channels=resnet_in_channels+res_skip_channels, out_channels=out_channels, temb_channels=temb_channels,
                                         eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift,
                                         non_linearity=resnet_act_fn, pre_norm=resnet_pre_norm))

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class SkeletUpBlock2D(nn.Module):
    def __init__(self, in_channels: int, prev_output_channel: int, out_channels: int, temb_channels: int, dropout: float = 0.0, num_layers: int = 1,
                 resnet_eps: float = 1e-6, resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish", resnet_groups: int = 32,
                 resnet_pre_norm: bool = True, add_upsample=True):
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            attentions.append(ChannelAttnBlock(in_channels=resnet_in_channels+res_skip_channels, out_channels=out_channels, groups=resnet_groups,
                                                       non_linearity=resnet_act_fn, channel_attn=True, reduction=32))
            resnets.append(ResnetBlock2D(in_channels=out_channels, out_channels=out_channels, temb_channels=temb_channels,
                                         eps=resnet_eps, groups=resnet_groups, dropout=dropout, time_embedding_norm=resnet_time_scale_shift,
                                         non_linearity=resnet_act_fn, pre_norm=resnet_pre_norm))

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
