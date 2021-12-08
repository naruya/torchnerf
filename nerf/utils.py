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
import argparse
import dataclasses
import gin
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclasses.dataclass
class TrainState:
    optimizer: torch.optim.Optimizer
    step: int


@dataclasses.dataclass
class Stats:
    loss: float = 0
    psnr: float = 0
    loss_c: float = 0
    psnr_c: float = 0


Rays = collections.namedtuple("Rays", ("origins", "directions", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


def define_args():
    """Define flags for both training and evaluation modes."""
    parser = argparse.ArgumentParser(description='TorchNeRF.')
    parser.add_argument(
        "--train_dir", type=str, default=None, help="where to store ckpts and logs")
    parser.add_argument(
        "--data_dir", type=str, default=None, help="input data directory.")
    parser.add_argument(
        "--gin_file", type=str, default=None, help="path to the config file.")
    parser.add_argument(
        "--model", type=str, default="nerf", help="name of model to use.")
    parser.add_argument(
        "--batch_size", type=int, default=4096, help="the number of rays in a mini-batch.")
    parser.add_argument(
        "--max_steps", type=int, default=1000000, help="the number of optimization steps.")
    parser.add_argument(
        "--print_every", type=int, default=500, help="the number of steps between reports to tensorboard.")
    parser.add_argument(
        "--save_every", type=int, default=10000, help="the number of steps to save a checkpoint.")
    parser.add_argument(
        "--render_every", type=int, default=10000, help="the number of steps to render a test image.")
    # eval
    parser.add_argument(
        "--chunk", type=int, default=4000, help="the size of chunks for evaluation inferences.")
    parser.add_argument(
        '--eval_once', action='store_true',
        help="evaluate the model only once if true, otherwise keeping evaluating new checkpoints.")
    parser.add_argument(
        "--showcase_index", type=int, default=0, help="index of test view point to render.")
    return parser.parse_args()


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


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
        mse = ((img0 - img1) ** 2).mean()
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


@gin.configurable
def learning_rate_decay(
    step,
    lr_init=5e-4,
    lr_final=5e-6,
    lr_max_steps=1000000,
    lr_delay_steps=0,
    lr_delay_mult=1,
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
