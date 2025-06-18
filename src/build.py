import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src import UNet


def build_unet(args):
    unet = UNet(
        sample_size=args.resolution,
        in_channels=6,
        out_channels=3,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=('DownBlock2D', 
                          'SkeletDownBlock2D',
                          'SkeletDownBlock2D', 
                          'DownBlock2D'),
        up_block_types=('UpBlock2D', 
                        'SkeletUpBlock2D',
                        'SkeletUpBlock2D', 
                        'UpBlock2D'),
        block_out_channels=args.unet_channels, 
        layers_per_block=1,
        downsample_padding=1,
        act_fn='silu',
        norm_num_groups=32,
        norm_eps=1e-05,
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        reduction=32)
    
    return unet


def build_ddpm_scheduler(args):
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="sample")
    return ddpm_scheduler

