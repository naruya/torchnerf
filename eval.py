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
"""Evaluation script for Nerf."""
import os
import functools
import gin
import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from nerf import datasets
from nerf import models
from nerf import utils


def main(args):
    gin.parse_config_file(args.gin_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    utils.set_random_seed(20210222)
    torch.autograd.set_detect_anomaly(True)

    print('* Load test data')
    dataset = datasets.get_dataset("test", args)
    print('* Load model')
    model, state = models.get_model_state(args, device=device, restore=False)
    print('* Done loading model')

    last_step = 0
    out_dir = os.path.join(
        args.train_dir, "path_renders" if dataset.render_path else "test_preds"
    )
    if not args.eval_once:
        summary_writer = SummaryWriter(os.path.join(args.train_dir, "eval"))

    while True:
        try:
            model, state = models.restore_model_state(args, model, state)
        except Exception as e:  # train.py is saving state just now.
            print(e); time.sleep(30); continue
        step = int(state.step)
        if step <= last_step:
            time.sleep(30); continue
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        psnrs = []
        ssims = []
        for idx in range(dataset.size):
            print(f"Evaluating {idx+1}/{dataset.size}")
            batch = next(dataset)
            rays = utils.namedtuple_map(
                lambda z: torch.from_numpy(z.copy()).to(device), batch["rays"])
            # Rendering is forced to be deterministic even if training was randomized, as
            # this eliminates "speckle" artifacts.
            with torch.no_grad():
                pred_color, pred_disp, pred_acc = utils.render_image(
                    functools.partial(model, randomized=False),
                    rays,
                    'llff' in args.data_dir.lower(),
                    chunk=args.chunk,
                )
                if not dataset.render_path:
                    gt_color = torch.from_numpy(batch["pixels"]).to(device)
                    psnr = utils.compute_psnr(pred_color, gt_color).mean().cpu().item()
                    ssim = utils.compute_ssim(pred_color, gt_color).mean().cpu().item()
                    print(f"PSNR = {psnr:.4f}, SSIM = {ssim:.4f}")
                    psnrs.append(float(psnr))
                    ssims.append(float(ssim))
            pred_color = pred_color.cpu().numpy()
            pred_disp = pred_disp.cpu().numpy()
            pred_acc = pred_acc.cpu().numpy()
            if not args.eval_once and idx == args.showcase_index:
                showcase_color = pred_color
                showcase_disp = pred_disp
                showcase_acc = pred_acc
                if not dataset.render_path:
                    showcase_gt = batch["pixels"]
            utils.save_img(pred_color, os.path.join(out_dir, "{:03d}.png".format(idx)))
            utils.save_img(
                pred_disp[Ellipsis, 0],
                os.path.join(out_dir, "disp_{:03d}.png".format(idx)),
            )
        if not args.eval_once:
            summary_writer.add_image("pred_color", showcase_color, step, None, 'HWC')
            summary_writer.add_image("pred_disp", showcase_disp, step, None, 'HWC')
            summary_writer.add_image("pred_acc", showcase_acc, step, None, 'HWC')
            if not dataset.render_path:
                summary_writer.add_scalar("psnr", np.mean(np.array(psnrs)), step)
                summary_writer.add_scalar("ssim", np.mean(np.array(ssims)), step)
                summary_writer.add_image("target", showcase_gt, step, None, 'HWC')
        if not dataset.render_path:
            with open(os.path.join(out_dir, "psnr.txt"), "w") as pout:
                pout.write("{}".format(np.mean(np.array(psnrs))))
            with open(os.path.join(out_dir, "ssim.txt"), "w") as pout:
                pout.write("{}".format(np.mean(np.array(ssims))))
        if args.eval_once:
            break
        if int(step) >= args.max_steps:
            break
        last_step = step


if __name__ == "__main__":
    args = utils.define_args()
    main(args)