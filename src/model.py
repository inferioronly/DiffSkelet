import torch
from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, register_to_config)


class DiffSkeletModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
    
    def forward(self, x_t, timesteps, input_images):

        out, skelet_out = self.config.unet(x_t, timesteps, input_images)
        
        return out, skelet_out