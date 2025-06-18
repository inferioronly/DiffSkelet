from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, register_to_config)
from diffusers.utils import BaseOutput, logging

from .embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import (DownBlock2D, UNetMidSkeletBlock2D, UpBlock2D, get_down_block, get_up_block)
from .skelet_block import SkeletDownBlock, SkeletUpBlock, SkeletMidBlock
from .MAFI import MAFI

logger = logging.get_logger(__name__)

@dataclass
class UNetOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, sample_size: Optional[int] = None, in_channels: int = 4, out_channels: int = 4, flip_sin_to_cos: bool = True, freq_shift: int = 0,
                 down_block_types: Tuple[str] = None, up_block_types: Tuple[str] = None, block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
                 layers_per_block: int = 1, downsample_padding: int = 1, act_fn: str = "silu", norm_num_groups: int = 32,
                 norm_eps: float = 1e-5, attention_head_dim: int = 8, channel_attn: bool = False, reduction: int = 32):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.skelet_down_blocks = nn.ModuleList([])
        self.feature_interaction_down = nn.ModuleList([])
        self.mid_block = None
        self.skelet_mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.skelet_up_blocks = nn.ModuleList([])
        self.feature_interaction_up = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = (i == len(block_out_channels) - 1)

            print("Load the down block ", down_block_type)
            down_block = get_down_block(down_block_type, num_layers=layers_per_block, in_channels=input_channel, out_channels=output_channel, temb_channels=time_embed_dim,
                                        add_downsample=not is_final_block, resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=norm_num_groups,
                                        attn_num_head_channels=attention_head_dim, downsample_padding=downsample_padding, reduction=reduction, channel_attn=channel_attn)
            self.down_blocks.append(down_block)
            self.skelet_down_blocks.append(SkeletDownBlock(3, 64, temb_channels=time_embed_dim, resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=norm_num_groups, use_pooling=False) 
                                           if i == 0 else 
                                           SkeletDownBlock(64*(2**(i-1)), 64*(2**i), temb_channels=time_embed_dim, resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=norm_num_groups, use_pooling=True))
            if i == 1 or i == 2:
                self.feature_interaction_down.append(MAFI((32//i, 32//i), 8, 4*i, 32, 32, 0.1, 0.1, position_block='down'))

        # mid
        self.mid_block = UNetMidSkeletBlock2D(in_channels=block_out_channels[-1], temb_channels=time_embed_dim, channel_attn=channel_attn, resnet_eps=norm_eps,
                                           resnet_act_fn=act_fn, resnet_time_scale_shift="default", resnet_groups=norm_num_groups, reduction=reduction)
        self.skelet_mid_block = SkeletMidBlock(512, temb_channels=time_embed_dim, resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=norm_num_groups)
        self.feature_interaction_mid = (MAFI((16, 16), 8, 16, 32, 32, 0.1, 0.1, position_block='mid'))

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            print("Load the up block ", up_block_type)
            up_block = get_up_block(up_block_type, num_layers=layers_per_block + 1,  # larger 1 than the down block
                                    in_channels=input_channel, out_channels=output_channel, prev_output_channel=prev_output_channel, 
                                    temb_channels=time_embed_dim, add_upsample=add_upsample, resnet_eps=norm_eps, resnet_act_fn=act_fn, 
                                    resnet_groups=norm_num_groups)
            self.up_blocks.append(up_block)
            self.skelet_up_blocks.append(SkeletUpBlock(512, 512, temb_channels=time_embed_dim, resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=norm_num_groups, use_up=False)
                                         if i == 0 else
                                         SkeletUpBlock(512//(2**(i-1)), 512//(2**i), temb_channels=time_embed_dim, resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=norm_num_groups, use_up=True))
            if i == 1 or i == 2:
                self.feature_interaction_up.append(MAFI((64*i, 64*i), 8, 8//i, 32, 32, 0.1, 0.1, position_block='up'))

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)
        self.skelet_conv_out = nn.Conv2d(block_out_channels[0], 1, kernel_size=1, stride=1, padding=0, bias=True)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.config.attention_head_dim % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.config.attention_head_dim}"
            )
        if slice_size is not None and slice_size > self.config.attention_head_dim:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.config.attention_head_dim}"
            )

        for block in self.down_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_attention_slice(slice_size)

        self.mid_block.set_attention_slice(slice_size)

        for block in self.up_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_attention_slice(slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (DownBlock2D, UpBlock2D)):
            module.gradient_checkpointing = value

    def forward(self, sample, timestep, input_image) -> Union[UNetOutput, Tuple]:
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time
        timesteps = timestep   # only one time
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)  # projection

        # 2. pre-process
        # print("shape: ", sample.shape)
        sample_skelet = input_image.clone()
        sample = self.conv_in(torch.cat([input_image, sample], dim=1))
        # print("after pre process: ", sample.shape)

        # 3. down
        down_block_res_samples = (sample,)
        down_block_skelet_res_samples = []
        for i, (downsample_block, skelet_downblock) in enumerate(zip(self.down_blocks, self.skelet_down_blocks)):
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)  
            sample_skelet = skelet_downblock(hidden_states=sample_skelet, temb=emb) 
            # sample_skelet = skelet_downblock(sample_skelet)

            if (i==1 or i==2) and self.feature_interaction_down[i-1] is not None:   # add feature interaction
                sample, sample_skelet = self.feature_interaction_down[i-1](sample, sample_skelet)
            res_samples = res_samples[:-1] + (sample,)                              # adjust the feature map of the last down block

            # print("after down block: ", sample.shape)
            down_block_skelet_res_samples.append(sample_skelet)
            down_block_res_samples += res_samples

        # 4. mid
        # print("before mid block: ", sample.shape)
        if self.mid_block and self.skelet_mid_block is not None:
            sample = self.mid_block(sample, emb)
            sample_skelet = self.skelet_mid_block(sample_skelet, emb)
            # sample_skelet = self.skelet_mid_block(sample_skelet)
            sample, sample_skelet = self.feature_interaction_mid(sample, sample_skelet)

        # print("after mid block: ", sample.shape)

        # 5. up
        for i, (upsample_block, skelet_upblock) in enumerate(zip(self.up_blocks, self.skelet_up_blocks)):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            # print("before up block: ", sample.shape)
            sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size)
            # sample_skelet = skelet_upblock(sample_skelet, down_block_skelet_res_samples, i)
            sample_skelet = skelet_upblock(sample_skelet, emb, down_block_skelet_res_samples, i)
            if (i==1 or i==2) and self.feature_interaction_up[i-1] is not None:
                sample, sample_skelet = self.feature_interaction_up[i-1](sample, sample_skelet)
            # print("after up block: ", sample.shape)

        # 6. post-process
        # print("before post process: ", sample.shape)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        sample_skelet = self.skelet_conv_out(sample_skelet)
        # print("after post process: ", sample.shape)

        return sample, sample_skelet
