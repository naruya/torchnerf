import os
import sys

os.chdir('../')
sys.path.append('./')

import argparse
import gin
import types
import functools
import numpy as np
import torch
from tqdm import tqdm

# pip install --upgrade PyMCubes
import mcubes

from nerf import datasets
from nerf import models
from nerf import utils
from nerf import model_utils


def define_args(args=None):
    """Define flags for both training and evaluation modes."""
    parser = argparse.ArgumentParser(description='TorchNeRF.')
    parser.add_argument(
        "--train_dir", type=str, default=None, help="where to store ckpts and logs")
    parser.add_argument(
        "--data_dir", type=str, default=None, help="input data directory.")
    parser.add_argument(
        "--gin_files", nargs='*', default=None, help="path to the config files.")
    parser.add_argument(
        "--model", type=str, default="nerf", help="name of model to use.")
    # for gen_mesh
    parser.add_argument(
        "--reso", type=int, default=300, help="resolution")
    parser.add_argument(
        "--lim", type=int, default=5, help="limitation")
    parser.add_argument(
        "--iso", type=int, default=5, help="isosurface")
    parser.add_argument(
        "--chunk", type=int, default=1000000, help="chunk size of points for evaluation")
    return parser.parse_args(args)


@torch.no_grad()
def eval_points(self, points, viewdirs=None, eval_rgb=False, coarse=False):
    """
    Evaluate at points, returing rgb and sigma.
    If sh_order >= 0 then this will return spherical harmonic
    coeffs for RGB. Please see eval_points for alternate
    version which always returns RGB.
    Args:
      points: torch.tensor [B, 3]
      viewdirs: torch.tensor [B, 3]
      coarse: if true, uses coarse MLP
    Returns:
      raw_rgb: torch.tensor [B, 3 * (sh_order + 1)**2 or 3]
      raw_sigma: torch.tensor [B, 1]
    """
    if self.num_fine_samples > 0 and not coarse:
        mlp = self.MLP_1
    else:
        mlp = self.MLP_0

    points = points[None]
    points_enc = model_utils.posenc(
        points,
        self.min_deg_point,
        self.max_deg_point,
        self.legacy_posenc_order,
    )

    if self.use_viewdirs and viewdirs is not None:
        viewdirs = viewdirs[None]
        viewdirs_enc = model_utils.posenc(
            viewdirs,
            0,
            self.deg_view,
            self.legacy_posenc_order,
        )

    if not eval_rgb:
        raw_sigma = mlp(points_enc, no_rgb=True)
        sigma = self.sigma_activation(raw_sigma)
        return sigma[0]

    else:
        if self.use_viewdirs:
            raw_rgb, _ = mlp(points_enc, viewdirs_enc)
        else:
            raw_rgb, _ = mlp(points_enc)
        rgb = self.rgb_activation(raw_rgb)
        return rgb[0]


def marching_cubes(fn_sigma, fn_rgb, c1, c2, reso, isosurface, chunk):
    """
    Run marching cubes on network. Uses PyMCubes.
    Args:
      fn main NeRF type network
      c1: list corner 1 of marching cube bounds x,y,z
      c2: list corner 2 of marching cube bounds x,y,z (all > c1)
      reso: list resolutions of marching cubes x,y,z
      isosurface: float sigma-isosurface of marching cubes
    """
    points = np.vstack(
        np.meshgrid(
            *(np.linspace(lo, hi, sz, dtype=np.float32)
                for lo, hi, sz in zip(c1, c2, reso)),
            indexing="ij"
        )
    ).reshape(3, -1).T
    # It's properly centered.
    # print(points[300*300*150+300*150+150])  # [0.01672241 0.01672241  0.01672241]
    # print(points[300*300*150+300*150+149])  # [0.01672241 0.01672241 -0.01672241]
    # print(points.max(0), points.min(0))  # [5. 5. 5.] [-5. -5. -5.]
    points = torch.from_numpy(points).to(device)

    print("* Evaluating sigma @", points.shape[0], "points")
    num_points = points.shape[0]
    sigmas = []

    for i in tqdm(range(0, num_points, chunk)):
        chunk_points = points[i : i + chunk]
        sigma = fn_sigma(chunk_points)
        sigma = sigma.detach().cpu().numpy()
        sigmas.append(sigma)
    sigmas = np.concatenate(sigmas, axis=0)
    sigmas = sigmas.reshape(*reso)

    print("* Running marching cubes")
    vertices, triangles = mcubes.marching_cubes(sigmas, isosurface)
    # Scale
    c1, c2 = np.array(c1), np.array(c2)
    vertices = vertices * (c2 - c1) / np.array(reso) + c1

    norms = np.linalg.norm(vertices, axis=-1, keepdims=True)
    viewdirs = - vertices / norms
    # use_pixel_centers
    vertices2 = vertices+np.array([0.01435,0.01435,0.01435])
    # evaluate a little inside
    vertices2 *= 0.99
    points = torch.from_numpy((vertices2).astype(np.float32)).to(device)
    viewdirs = torch.from_numpy(viewdirs.astype(np.float32)).to(device)

    print("* Evaluating rgb @", points.shape[0], "points")
    num_points = points.shape[0]
    rgbs = []

    for i in tqdm(range(0, num_points, chunk)):
        chunk_points = points[i : i + chunk]
        chunk_viewdirs = viewdirs[i : i + chunk]
        rgb = fn_rgb(chunk_points, chunk_viewdirs)
        rgb = rgb.detach().cpu().numpy()
        rgbs.append(rgb)
    vert_rgb = np.concatenate(rgbs, axis=0)

    return vertices, triangles, vert_rgb


def save_obj(vertices, triangles, path, vert_rgb=None):
    """
    Save OBJ file, optionally with vertex colors.
    This version is faster than PyMCubes and supports color.
    Taken from PIFu.
    :param vertices (N, 3)
    :param triangles (N, 3)
    :param vert_rgb (N, 3) rgb
    """
    file = open(path, "w")
    if vert_rgb is None:
        # No color
        for v in vertices:
            file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
    else:
        # Color
        for v, c in zip(vertices, vert_rgb):
            file.write(
                "v %.4f %.4f %.4f %.4f %.4f %.4f\n"
                % (v[0], v[1], v[2], c[0], c[1], c[2])
            )
    for f in triangles:
        f_plus = f + 1
        file.write("f %d %d %d\n" % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


# python gen_mesh.py
if __name__ == "__main__":
    
    args = [
        "--gin_files", "./configs/unity.py", "./configs/c64f128.py",
        "--data_dir", "../data/unity/face1",
        "--train_dir", "../logs/face1b",
    ]
    args = define_args(args)
    device = "cuda:0"

    gin.parse_config_files_and_bindings(args.gin_files, None)

    print('* Creating model')
    model, state = models.get_model_state(args, device=device, restore=True)

    model.eval_points = types.MethodType(eval_points, model)

    verts, faces, vert_rgb = marching_cubes(
        functools.partial(model.eval_points, viewdirs=None, eval_rgb=False),
        functools.partial(model.eval_points, eval_rgb=True),
        c1=[-args.lim]*3, c2=[args.lim]*3, reso=[args.reso]*3, isosurface=args.iso, chunk=args.chunk
    )
    print(verts.shape, faces.shape, vert_rgb.shape)

    mesh_path = os.path.join(args.train_dir, 'mesh.obj')
    print('* Saving to', mesh_path)
    save_obj(verts, faces, mesh_path, vert_rgb)
