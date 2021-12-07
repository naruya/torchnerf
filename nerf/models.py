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
"""Different model implementation plus a general port for all the models."""
import os
import glob
import inspect
from typing import Any, Callable

import torch
import torch.nn as nn

from nerf import model_utils
from nerf import utils


def get_model(args):
    """A helper function that wraps around a 'model zoo'."""
    model_dict = {
        "nerf": construct_nerf,
    }
    return model_dict[args.model](args)


def get_model_state(args, device="cpu", restore=True):
    """
    Helper for loading model with get_model & creating optimizer &
    optionally restoring checkpoint to reduce boilerplate
    """
    model = get_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.lr_init, 
                                 weight_decay=args.weight_decay_mult)
    state = utils.TrainState(optimizer=optimizer, step=0)
    if restore:
        model, state = restore_model_state(args, model, state)
    return model, state


def restore_model_state(args, model, state):
    """
    Helper for restoring checkpoint.
    """
    ckpt_paths = sorted(
        glob.glob(os.path.join(args.train_dir, "*.ckpt")))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt["model"])
        state.optimizer.load_state_dict(ckpt["optimizer"])
        state = utils.TrainState(optimizer=state.optimizer, step=ckpt["step"])
        print (f"* restore ckpt from {ckpt_path}.")
    return model, state

class NerfModel(nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""
    
    def __init__(
        self,
        num_coarse_samples: int = 64,  # The number of samples for the coarse nerf.
        num_fine_samples: int = 128,  # The number of samples for the fine nerf.
        use_viewdirs: bool = True,  # If True, use viewdirs as an input.
        near: float = 2.0,  # The distance to the near plane
        far: float = 6.0,  # The distance to the far plane
        noise_std: float = 0.0,  # The std dev of noise added to raw sigma.
        net_depth: int = 8,  # The depth of the first part of MLP.
        net_width: int = 256,  # The width of the first part of MLP.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        net_activation: Callable[Ellipsis, Any] = nn.ReLU(),  # MLP activation
        skip_layer: int = 4,  # How often to add skip connections.
        num_rgb_channels: int = 3,  # The number of RGB channels.
        num_sigma_channels: int = 1,  # The number of density channels.
        white_bkgd: bool = True,  # If True, use a white background.
        min_deg_point: int = 0,  # The minimum degree of positional encoding for positions.
        max_deg_point: int = 10,  # The maximum degree of positional encoding for positions.
        deg_view: int = 4,  # The degree of positional encoding for viewdirs.
        lindisp: bool = False,  # If True, sample linearly in disparity rather than in depth.
        rgb_activation: Callable[Ellipsis, Any] = nn.Sigmoid(),  # Output RGB activation.
        sigma_activation: Callable[Ellipsis, Any] = nn.ReLU(),  # Output sigma activation.
        legacy_posenc_order: bool = False,  # Keep the same ordering as the original tf code.
    ):
        super(NerfModel, self).__init__()
        self.num_coarse_samples = num_coarse_samples
        self.num_fine_samples = num_fine_samples
        self.use_viewdirs = use_viewdirs
        self.near = near
        self.far = far
        self.noise_std = noise_std
        self.net_depth = net_depth
        self.net_width = net_width
        self.net_depth_condition = net_depth_condition
        self.net_width_condition = net_width_condition
        self.net_activation = net_activation
        self.skip_layer = skip_layer
        self.num_rgb_channels = num_rgb_channels
        self.num_sigma_channels = num_sigma_channels
        self.white_bkgd = white_bkgd
        self.min_deg_point = min_deg_point
        self.max_deg_point = max_deg_point
        self.deg_view = deg_view
        self.lindisp = lindisp
        self.rgb_activation = rgb_activation
        self.sigma_activation = sigma_activation
        self.legacy_posenc_order = legacy_posenc_order
        # TODO calculate input_dim/condition_dim from args
        # Construct the "coarse" MLP. Weird name is for
        # compatibility with 'compact' version
        self.MLP_0 = model_utils.MLP(
            net_depth = self.net_depth,
            net_width = self.net_width,
            net_depth_condition = self.net_depth_condition,
            net_width_condition = self.net_width_condition,
            net_activation = self.net_activation,
            skip_layer = self.skip_layer,
            num_rgb_channels = self.num_rgb_channels,
            num_sigma_channels = self.num_sigma_channels,
            input_dim=63,
            condition_dim=27)
        # Construct the "fine" MLP.
        if self.num_fine_samples > 0:
            self.MLP_1 = model_utils.MLP(
                net_depth = self.net_depth,
                net_width = self.net_width,
                net_depth_condition = self.net_depth_condition,
                net_width_condition = self.net_width_condition,
                net_activation = self.net_activation,
                skip_layer = self.skip_layer,
                num_rgb_channels = self.num_rgb_channels,
                num_sigma_channels = self.num_sigma_channels,
                input_dim=63,
                condition_dim=27)

    def forward(self, rays, randomized):
        """Nerf Model.

        Args:
          rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
          randomized: bool, use randomized stratified sampling.

        Returns:
          ret: list, [(rgb_coarse, disp_coarse, acc_coarse), (rgb, disp, acc)]
        """
        # Stratified sampling along rays
        z_vals, samples = model_utils.sample_along_rays(
            rays.origins,
            rays.directions,
            self.num_coarse_samples,
            self.near,
            self.far,
            randomized,
            self.lindisp,
        )
        samples_enc = model_utils.posenc(
            samples,
            self.min_deg_point,
            self.max_deg_point,
            self.legacy_posenc_order,
        )

        # Point attribute predictions
        if self.use_viewdirs:
            viewdirs_enc = model_utils.posenc(
                rays.viewdirs,
                0,
                self.deg_view,
                self.legacy_posenc_order,
            )
            raw_rgb, raw_sigma = self.MLP_0(samples_enc, viewdirs_enc)
        else:
            raw_rgb, raw_sigma = self.MLP_0(samples_enc)
        # Add noises to regularize the density predictions if needed
        raw_sigma = model_utils.add_gaussian_noise(
            raw_sigma,
            self.noise_std,
            randomized,
        )

        rgb = self.rgb_activation(raw_rgb)
        sigma = self.sigma_activation(raw_sigma)

        # Volumetric rendering.
        comp_rgb, disp, acc, weights = model_utils.volumetric_rendering(
            rgb,
            sigma,
            z_vals,
            rays.directions,
            white_bkgd=self.white_bkgd,
        )
        ret = [
            (comp_rgb, disp, acc),
        ]
        # Hierarchical sampling based on coarse predictions
        if self.num_fine_samples > 0:
            z_vals_mid = 0.5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
            z_vals, samples = model_utils.sample_pdf(
                z_vals_mid,
                weights[Ellipsis, 1:-1],
                rays.origins,
                rays.directions,
                z_vals,
                self.num_fine_samples,
                randomized,
            )
            samples_enc = model_utils.posenc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
                self.legacy_posenc_order,
            )

            if self.use_viewdirs:
                raw_rgb, raw_sigma = self.MLP_1(samples_enc, viewdirs_enc)
            else:
                raw_rgb, raw_sigma = self.MLP_1(samples_enc)
            raw_sigma = model_utils.add_gaussian_noise(
                raw_sigma,
                self.noise_std,
                randomized,
            )

            rgb = self.rgb_activation(raw_rgb)
            sigma = self.sigma_activation(raw_sigma)
            comp_rgb, disp, acc, unused_weights = model_utils.volumetric_rendering(
                rgb,
                sigma,
                z_vals,
                rays.directions,
                white_bkgd=self.white_bkgd,
            )
            ret.append((comp_rgb, disp, acc))
        return ret


def construct_nerf(args):
    """Construct a Neural Radiance Field.

    Args:
      args: FLAGS class. Hyperparameters of nerf.

    Returns:
      model: nn.Model. Nerf model with parameters.
      state: flax.Module.state. Nerf model state for stateful parameters.
    """
    net_activation = getattr(nn, str(args.net_activation))
    if inspect.isclass(net_activation):
        net_activation = net_activation()
    rgb_activation = getattr(nn, str(args.rgb_activation))
    if inspect.isclass(rgb_activation):
        rgb_activation = rgb_activation()
    sigma_activation = getattr(nn, str(args.sigma_activation))
    if inspect.isclass(sigma_activation):
        sigma_activation = sigma_activation()

    model = NerfModel(
        min_deg_point=args.min_deg_point,
        max_deg_point=args.max_deg_point,
        deg_view=args.deg_view,
        num_coarse_samples=args.num_coarse_samples,
        num_fine_samples=args.num_fine_samples,
        use_viewdirs=args.use_viewdirs,
        near=args.near,
        far=args.far,
        noise_std=args.noise_std,
        white_bkgd=args.white_bkgd,
        net_depth=args.net_depth,
        net_width=args.net_width,
        net_depth_condition=args.net_depth_condition,
        net_width_condition=args.net_width_condition,
        skip_layer=args.skip_layer,
        num_rgb_channels=args.num_rgb_channels,
        num_sigma_channels=args.num_sigma_channels,
        lindisp=args.lindisp,
        net_activation=net_activation,
        rgb_activation=rgb_activation,
        sigma_activation=sigma_activation,
        legacy_posenc_order=args.legacy_posenc_order,
    )
    return model
