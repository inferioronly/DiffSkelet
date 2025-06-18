import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import trunc_normal_
from .resnet import Downsample2D, Upsample2D


class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
    

class DAttentionBaseline(nn.Module):

    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_group_channels, attn_drop, proj_drop
    ):

        super().__init__()
        self.q_h, self.q_w = q_size
        stride = self.q_h // kv_size
        ksize = stride + 1   # larger than stride
        pad_size = ksize // 2 if ksize != stride else 0

        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5

        self.n_heads = n_heads
        self.nc = self.n_head_channels * self.n_heads

        self.n_group_channels = n_group_channels
        self.n_groups = self.nc // self.n_group_channels
        self.n_group_heads = self.n_heads // self.n_groups

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, ksize, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1))
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

    def forward(self, x, y):

        assert x.size() == y.size(), "x and y must have the same size"

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # (b*g, 2, h//stride, w//stride)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        pos = (offset + reference).clamp(-1., +1.)

        y_sampled = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
                
        y_sampled = y_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(y_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(y_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)


        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        q_grid = self._get_q_grid(H, W, B, dtype, device)
        displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
        attn_bias = F.grid_sample(
            input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups),
            grid=displacement[..., (1, 0)],
            mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns

        attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
        attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        out = out.reshape(B, C, H, W)

        out = self.proj_drop(self.proj_out(out))

        return out, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


class MAFI(nn.Module):
    def __init__(self, q_size, kv_size, n_heads, n_head_channels, n_group_channels,
                    attn_drop, proj_drop, position_block='down'):
        super().__init__()

        dim = n_head_channels * n_heads
        if position_block == 'down':
            self.change_conv = Downsample2D(
            channels=dim,
            use_conv=True,
            out_channels=dim,
            padding=1,
            name='op'
        )
            self.return_conv = Upsample2D(
            channels=dim,
            use_conv=True,
            use_conv_transpose=False,
            out_channels=dim,
            name='conv'
        )
        elif position_block == 'up':
            self.change_conv = Upsample2D(
            channels=dim,
            use_conv=True,
            use_conv_transpose=False,
            out_channels=dim,
            name='conv'
        )
            self.return_conv = Downsample2D(
            channels=dim,
            use_conv=True,
            out_channels=dim,
            padding=1,
            name='op'
        )
        else:
            self.change_conv = nn.Identity()
            self.return_conv = nn.Identity()

        self.attn1 = DAttentionBaseline(q_size, kv_size, n_heads, n_head_channels, n_group_channels, attn_drop, proj_drop)
        self.attn2 = DAttentionBaseline(q_size, kv_size, n_heads, n_head_channels, n_group_channels, attn_drop, proj_drop)
        
    def forward(self, x1, x2):
        """
        x1: (B, C, H1, W1)
        x2: (B, C, H2, W2) 
        """
        x2 = self.change_conv(x2)
        out1, _, _ = self.attn1(x1, x2)
        out1 = out1 + x1

        out2, _, _ = self.attn2(x2, x1)
        out2 = out2 + x2
        out2 = self.return_conv(out2)

        return out1, out2