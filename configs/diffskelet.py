import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Training config for DiffSkelet.")
    ################# Experience #################
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--experience_name", type=str, default="diffskelet_training")
    parser.add_argument("--data_root", type=str, default=None, help="The root path of dataset.",)
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="The output directory where the training checkpoints will be written.")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs", 
                        help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                              " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."))

    # Model
    parser.add_argument("--resolution", type=int, default=128, 
                        help="The resolution for input images, all the images in the train/validation \
                            dataset will be resized to this.")
    parser.add_argument("--unet_channels", type=tuple, default=(64, 128, 256, 512),
                        help="The channels of the UNet.")
    parser.add_argument("--channel_attn", type=bool, default=True, help="Whether to use the se attention.",)

    ## train batch size
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    ## loss coefficient
    parser.add_argument("--perceptual_coefficient", type=float, default=0.01)
    parser.add_argument("--skelet_coefficient", type=float, default=0.1)
    ## step
    parser.add_argument("--max_train_steps", type=int, default=200000, 
                        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--ckpt_interval", type=int,default=20000, help="The step begin to validate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--log_interval", type=int, default=100, help="The log interval of training.")
    ## learning rate
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False, 
                        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="linear", 
                        help="The scheduler type to use. Choose between 'linear', 'cosine', \
                            'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'")
    parser.add_argument("--lr_warmup_steps", type=int, default=10000, 
                        help="Number of steps for the warmup in the lr scheduler.")
    ## scheduler
    parser.add_argument("--beta_scheduler", type=str, default="scaled_linear", help="The beta scheduler for DDPM.")
    ## optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], 
                        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires \
                            PyTorch >= 1.10. and an Nvidia Ampere GPU.")
    ## sample
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path of checkpoint dir.")
    parser.add_argument("--input_image_dir", type=str, default=None, help="The path of input image dir.")
    parser.add_argument("--save_image_dir", type=str, default=None, help="The path of output image dir.")
    parser.add_argument("--ddim_num_steps", type=int, default=1, help="The number of sampling steps of DDIM")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    return parser
