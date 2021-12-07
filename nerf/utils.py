# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Utility functions."""
import collections
import os
import math
import argparse

import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm


class TrainState:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step: int = 0
    ):
        self.optimizer = optimizer
        self.step = step


class Stats:
    def __init__(
        self,
        loss: float = 0,
        psnr: float = 0,
        loss_c: float = 0,
        psnr_c: float = 0,
    ):
        self.loss = loss
        self.psnr = psnr
        self.loss_c = loss_c
        self.psnr_c = psnr_c


Rays = collections.namedtuple("Rays", ("origins", "directions", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


def define_flags():
    """Define flags for both training and evaluation modes."""
    parser = argparse.ArgumentParser(description='TorchNeRF.')
    parser.add_argument(
        "--train_dir", type=str, default=None,
        help="where to store ckpts and logs")
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="input data directory.")
    parser.add_argument(
        "--config", type=str, default=None,
        help="using config files to set hyperparameters.")

    # Dataset Flags
    parser.add_argument(
        "--dataset", type=str, default="blender", choices=["blender", "llff"],
        help="The type of dataset feed to nerf.")
    parser.add_argument(
        "--image_batching", type=bool, default=False,
        help="sample rays in a batch from different images.")
    parser.add_argument(
        "--white_bkgd", type=bool, default=True,
        help="using white color as default background. (used in the blender dataset only)")
    parser.add_argument(
        "--batch_size", type=int, default=1024,
        help="the number of rays in a mini-batch (for training).")
    parser.add_argument(
        "--factor", type=int, default=4,
        help="the downsample factor of images, 0 for no downsample.")
    parser.add_argument(
        "--spherify", type=bool, default=False,
        help="set for spherical 360 scenes.")
    parser.add_argument(
        "--render_path", type=bool, default=False,
        help="render generated path if set true. (used in the llff dataset only)")
    parser.add_argument(
        "--llffhold", type=int, default=8,
        help="will take every 1/N images as LLFF test set. (used in the llff dataset only)")

    # Model Flags
    parser.add_argument(
        "--model", type=str, default="nerf", help="name of model to use.")
    parser.add_argument(
        "--near", type=float, default=2.0, help="near clip of volumetric rendering.")
    parser.add_argument(
        "--far", type=float, default=6.0, help="far clip of volumentric rendering.")
    parser.add_argument(
        "--net_depth", type=int, default=8, help="depth of the first part of MLP.")
    parser.add_argument(
        "--net_width", type=int, default=256, help="width of the first part of MLP.")
    parser.add_argument(
        "--net_depth_condition", type=int, default=1, help="depth of the second part of MLP.")
    parser.add_argument(
        "--net_width_condition", type=int, default=128, help="width of the second part of MLP.")
    parser.add_argument(
        "--weight_decay_mult", type=float, default=0., help="The multiplier on weight decay")
    parser.add_argument(
        "--skip_layer", type=int, default=4,
        help="add a skip connection to the output vector of every skip_layer layers.")
    parser.add_argument(
        "--num_rgb_channels", type=int, default=3, help="the number of RGB channels.")
    parser.add_argument(
        "--num_sigma_channels", type=int, default=1, help="the number of density channels.")
    parser.add_argument(
        "--randomized", type=int, default=True, help="use randomized stratified sampling.")
    parser.add_argument(
        "--min_deg_point", type=int, default=0,
        help="Minimum degree of positional encoding for points.")
    parser.add_argument(
        "--max_deg_point", type=int, default=10,
        help="Maximum degree of positional encoding for points.")
    parser.add_argument(
        "--deg_view", type=int, default=4, help="Degree of positional encoding for viewdirs.")
    parser.add_argument(
        "--num_coarse_samples", type=int, default=64,
        help="the number of samples on each ray for the coarse model.")
    parser.add_argument(
        "--num_fine_samples", type=int, default=128,
        help="the number of samples on each ray for the fine model.")
    parser.add_argument(
        "--use_viewdirs", type=bool, default=True, help="use view directions as a condition.")
    parser.add_argument(
        "--noise_std", type=float, default=None,
        help="std dev of noise added to regularize sigma output. (used in the llff dataset only)")
    parser.add_argument(
        "--lindisp", type=bool, default=False,
        help="sampling linearly in disparity rather than depth.")
    parser.add_argument(
        "--net_activation", type=str, default="ReLU",
        help="activation function used within the MLP.")
    parser.add_argument(
        "--rgb_activation", type=str, default="Sigmoid",
        help="activation function used to produce RGB.")
    parser.add_argument(
        "--sigma_activation", type=str, default="Softplus",
        help="activation function used to produce density.")
    parser.add_argument(
        "--legacy_posenc_order", type=bool, default=False,
        help="If True, revert the positional encoding feature order to an older version of this codebase.")

    # Train Flags
    parser.add_argument(
        "--lr_init", type=float, default=5e-4, help="The initial learning rate.")
    parser.add_argument(
        "--lr_final", type=float, default=5e-6, help="The final learning rate.")
    parser.add_argument(
        "--lr_max_steps", type=int, default=1000000, help="(for a fair comparison)")
    parser.add_argument(
        "--lr_delay_steps", type=int, default=0,
        help="The number of steps at the beginning of training to reduce the learning rate by lr_delay_mult")
    parser.add_argument(
        "--lr_delay_mult", type=float, default=1.0,
        help="A multiplier on the learning rate when the step is < lr_delay_steps")
    parser.add_argument(
        "--max_steps", type=int, default=1000000, help="the number of optimization steps.")
    parser.add_argument(
        "--save_every", type=int, default=5000, help="the number of steps to save a checkpoint.")
    parser.add_argument(
        "--print_every", type=int, default=500, help="the number of steps between reports to tensorboard.")
    parser.add_argument(
        "--render_every", type=int, default=10000,
        help="the number of steps to render a test image better to be x00 for accurate step time record.")

    # Eval Flags
    parser.add_argument(
        "--eval_once", type=bool, default=True,
        help="evaluate the model only once if true, otherwise keeping evaluating new checkpoints if there's any.")
    parser.add_argument(
        "--save_output", type=bool, default=True, help="save predicted images to disk if True.")
    parser.add_argument(
        "--chunk", type=int, default=8192,
        help="the size of chunks for evaluation inferences, set to the value that"
        "fits your GPU/TPU memory.")
    return parser.parse_args()


def update_flags(args):
    """Update the flags in `args` with the contents of the config YAML file."""
    if args.config is None:
        return
    pth = os.path.join(args.config + ".yaml")
    with open(pth, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    # Only allow args to be updated if they already exist.
    invalid_args = list(set(configs.keys()) - set(dir(args)))
    if invalid_args:
        raise ValueError(f"Invalid args {invalid_args} in {pth}.")
    args.__dict__.update(configs)


def check_flags(args, require_data=True, require_batch_size_div=False):
    if args.train_dir is None:
        raise ValueError("train_dir must be set. None set now.")
    if require_data and args.data_dir is None:
        raise ValueError("data_dir must be set. None set now.")
    if require_batch_size_div and args.batch_size % torch.cuda.device_count() != 0:
        raise ValueError("Batch size must be divisible by the number of devices.")


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def render_image(render_fn, rays, normalize_disp, chunk=8192):
    """Render all the pixels of an image (in test mode).

    Args:
      render_fn: function, render function.
      rays: a `Rays` namedtuple, the rays to be rendered.
      normalize_disp: bool, if true then normalize `disp` to [0, 1].
      chunk: int, the size of chunks to render sequentially.

    Returns:
      rgb: torch.tensor, rendered color image.
      disp: torch.tensor, rendered disparity image.
      acc: torch.tensor, rendered accumulated weights per pixel.
    """
    height, width = rays[0].shape[:2]
    num_rays = height * width
    rays = namedtuple_map(lambda r: r.reshape((num_rays, -1)), rays)

    results = []
    for i in tqdm(range(0, num_rays, chunk)):
        # pylint: disable=cell-var-from-loop
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        chunk_results = render_fn(chunk_rays)[-1]
        results.append(chunk_results)
        # pylint: enable=cell-var-from-loop
    rgb, disp, acc = [torch.cat(r, dim=0) for r in zip(*results)]
    # Normalize disp for visualization for ndc_rays in llff front-facing scenes.
    if normalize_disp:
        disp = (disp - disp.min()) / (disp.max() - disp.min())
    return (
        rgb.view((height, width, -1)),
        disp.view((height, width, -1)),
        acc.view((height, width, -1)),
    )


def compute_psnr(*args):
    """Compute psnr value given mse (we assume the maximum pixel value is 1).

    Args:
      mse: float, mean square error of pixels.

    Returns:
      psnr: float, the psnr value.
    """
    if len(args) == 1:
        mse = args[0]
    else:
        img0, img1 = args[0], args[1]
        mse = (img0 - img1) ** 2
    return -10.0 * torch.log(mse) / np.log(10.0)


def compute_ssim(
    img0,
    img1,
    max_val=1.0,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    """Computes SSIM from two images.

    This function was modeled after tf.image.ssim, and should produce comparable
    output.

    Args:
      img0: torch.tensor. An image of size [..., width, height, num_channels].
      img1: torch.tensor. An image of size [..., width, height, num_channels].
      max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
      filter_size: int >= 1. Window size.
      filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
      k1: float > 0. One of the SSIM dampening parameters.
      k2: float > 0. One of the SSIM dampening parameters.
      return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned

    Returns:
      Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """
    device = img0.device
    ori_shape = img0.size()
    width, height, num_channels = ori_shape[-3:]
    img0 = img0.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    img1 = img1.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    batch_size = img0.shape[0]

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    filt_fn1 = lambda z: F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0], groups=num_channels)
    filt_fn2 = lambda z: F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw], groups=num_channels)

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z: filt_fn1(filt_fn2(z))
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.minimum(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = torch.mean(ssim_map.reshape([-1, num_channels*width*height]), dim=-1)
    return ssim_map if return_map else ssim


def save_img(img, pth):
    """Save an image to disk.

    Args:
      img: jnp.ndarry, [height, width, channels], img will be clipped to [0, 1]
        before saved to pth.
      pth: string, path to save the image to.
    """
    with open(pth, "wb") as imgout:
        Image.fromarray(
            np.array((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
        ).save(imgout, "PNG")


def learning_rate_decay(
    step, lr_init, lr_final, lr_max_steps, lr_delay_steps=0, lr_delay_mult=1
):
    """Continuous learning rate decay function.

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    Args:
      step: int, the current optimization step.
      lr_init: float, the initial learning rate.
      lr_final: float, the final learning rate.
      max_steps: int, the number of steps during optimization.
      lr_delay_steps: int, the number of steps to delay the full learning rate.
      lr_delay_mult: float, the multiplier on the rate when delaying it.

    Returns:
      lr: the learning for current step 'step'.
    """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / lr_max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp
