import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
import os
from src import build_unet, DiffSkeletModel
from utils import make_ddim_sampling_parameters, make_ddim_timesteps, make_beta_schedule, noise_like, extract_into_tensor, default


class DiffSkelet_pipline(object):
    def __init__(self, unet, device="cuda:0", ddim_num_steps=1):
        super(DiffSkelet_pipline, self).__init__()

        self.device = torch.device(device)
        self.unet = unet.to(self.device)

        self.ddpm_schedule_init(timesteps=1000, schedule="linear", linear_start=0.0001, linear_end=0.02, cosine_s=8e-3)                      
        self.ddim_schedule_init(ddim_num_steps=ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=False)
        
        self.model = DiffSkeletModel(unet=self.unet)


    def ddpm_schedule_init(self, timesteps, schedule, linear_start, linear_end, cosine_s):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = make_beta_schedule(schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])

        self.betas = to_torch(self.betas)
        self.alphas = to_torch(self.alphas)
        self.alphas_cumprod = to_torch(self.alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(self.alphas_cumprod_prev)

        timesteps, = self.betas.shape
        self.ddpm_num_timesteps = int(timesteps)

    def ddim_schedule_init(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)

        assert self.alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.betas))
        self.register_buffer('alphas_cumprod', to_torch(self.alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(self.alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - self.alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - self.alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / self.alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / self.alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod.cpu(),
                                                                                                ddim_timesteps=self.ddim_timesteps,
                                                                                                eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    def register_buffer(self, name, attr: torch.Tensor):
        if not isinstance(attr, torch.Tensor):
            attr = torch.tensor(attr)
        setattr(self, name, attr.to(self.device))

    @torch.no_grad()
    def p_sample(self, z_t, z_cond, t):
        
        out, _ = self.model(x_t=z_t, timesteps=t, input_images=z_cond)
        return out
    
    @torch.no_grad()
    def __call__(self, input_image_pil, in_size=(128, 128)):
        input_tensor = transforms.Compose([
            transforms.Resize(in_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])(input_image_pil).unsqueeze(0).to(self.device)

        # 初始噪声
        x_t = torch.randn_like(input_tensor, device=self.device)

        timesteps = self.ddim_timesteps
        total_steps = timesteps.shape[0]
        iterator = tqdm(reversed(timesteps), desc='DiffSkelet Sampling',
                        total=total_steps)

        for _ in iterator:
            t_step = torch.full((1,), 999, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, input_tensor, t_step)

        img = (x_t.detach().cpu() * 0.5 + 0.5
               ).clamp_(0., 1.) * 255.
        img_np = img.numpy().astype(np.uint8)[0]       # CHW
        return np.transpose(img_np, (1, 2, 0))         # HWC
    

def arg_parse():
    from configs.diffskelet import get_parser

    parser = get_parser()
    args = parser.parse_args()
    args.input_image_size = (args.resolution, args.resolution)
    args.target_image_size = (args.resolution, args.resolution)

    return args

def load_diffskelet_pipeline(args):
    unet = build_unet(args=args).to(args.device)
    unet.load_state_dict(
        torch.load(os.path.join(args.ckpt_dir, "unet.pth"),
                   map_location=args.device)
    )
    pipe = DiffSkelet_pipline(unet=unet, device=args.device, ddim_num_steps=args.ddim_num_steps)
    print(f"Loaded pipeline successfully!  |  ddim_num_steps: {args.ddim_num_steps}")

    return pipe
    

def main():
    args = arg_parse()
    pipe = load_diffskelet_pipeline(args)
    os.makedirs(args.save_image_dir, exist_ok=True)
    exts = ('.jpg', '.jpeg', '.png', '.bmp')

    input_files = [
        f for f in sorted(os.listdir(args.input_image_dir))
        if f.lower().endswith(exts)
    ]
    for fname in tqdm(input_files, desc="Images"):
        in_path = os.path.join(args.input_image_dir, fname)
        out_path = os.path.join(args.save_image_dir, fname)

        img_pil = Image.open(in_path).convert('RGB')
        output_np = pipe(img_pil, in_size=args.input_image_size)

        Image.fromarray(output_np, 'RGB').save(out_path)


if __name__ == "__main__":
    main()