"""Camera trajectories and their Plücker-ray embedding, the input of the CameraCtrl pose encoder.

Conventions are those of RealEstate10K and of the authors' `inference.py`: camera axes are
x right, y down, z forward; a pose is a camera-to-world matrix; every trajectory is expressed
relative to its first frame, which is the identity.

The trajectories here are synthetic, one elementary motion each. Their names say what the
*camera* does: `pan-left` translates it to its left (the scene drifts right in the image),
`zoom-in` moves it forward (a dolly), `roll-cw` rotates it clockwise about the viewing axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

CAMERAS = ("static", "pan-left", "pan-right", "tilt-up", "tilt-down", "zoom-in", "zoom-out", "roll-cw", "roll-ccw")

# Focal length as a fraction of the frame height. The authors' sample pose files have
# fy = 0.9025 of a 720-pixel-high frame; `inference.py` rescales fx so that pixels stay square.
FOCAL = 0.9025

_TRANSLATE = {"pan-left": (-1, 0, 0), "pan-right": (1, 0, 0), "tilt-up": (0, -1, 0), "tilt-down": (0, 1, 0),
              "zoom-in": (0, 0, 1), "zoom-out": (0, 0, -1)}
_ROLL = {"roll-cw": 1.0, "roll-ccw": -1.0}


@dataclass(frozen=True)
class Trajectory:
    name: str
    c2w: torch.Tensor          # (F, 4, 4) camera-to-world, float32
    focal: float = FOCAL       # fraction of the frame height, same for x and y (square pixels)

    @property
    def frames(self) -> int:
        return self.c2w.shape[0]


def trajectory(name: str, frames: int = 16, step: float = 0.1, roll_deg: float = 2.0) -> Trajectory:
    """One elementary camera motion: `step` scene units per frame, or `roll_deg` degrees per frame."""
    if name not in CAMERAS:
        raise KeyError(f"unknown camera {name!r}; known: {CAMERAS}")
    c2w = torch.eye(4).repeat(frames, 1, 1)
    i = torch.arange(frames, dtype=torch.float32)
    if name in _TRANSLATE:
        c2w[:, :3, 3] = i[:, None] * step * torch.tensor(_TRANSLATE[name], dtype=torch.float32)
    elif name in _ROLL:
        a = i * math.radians(roll_deg) * _ROLL[name]
        c2w[:, 0, 0], c2w[:, 0, 1] = torch.cos(a), -torch.sin(a)
        c2w[:, 1, 0], c2w[:, 1, 1] = torch.sin(a), torch.cos(a)
    return Trajectory(name, c2w)


def relative_poses(c2w: torch.Tensor) -> torch.Tensor:
    """Poses relative to the first frame: frame 0 becomes the identity (the authors' `get_relative_pose`)."""
    rel = torch.linalg.inv(c2w[0])[None] @ c2w
    rel[0] = torch.eye(4, dtype=c2w.dtype)
    return rel


def plucker_embedding(traj: Trajectory, height: int, width: int) -> torch.Tensor:
    """(1, 6, F, H, W) float32: per pixel, the moment `o x d` and the unit direction `d` of its ray.

    Same arithmetic as `ray_condition` in the authors' `inference.py`.
    """
    c2w = relative_poses(traj.c2w.float())
    f = traj.focal * height
    j, i = torch.meshgrid(torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width), indexing="ij")
    xs = (i.reshape(-1) + 0.5 - width / 2) / f
    ys = (j.reshape(-1) + 0.5 - height / 2) / f
    d = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    d = d / d.norm(dim=-1, keepdim=True)                               # (HW, 3), camera frame
    rays_d = d[None] @ c2w[:, :3, :3].transpose(-1, -2)                # (F, HW, 3), frame-0 frame
    rays_o = c2w[:, None, :3, 3].expand_as(rays_d)
    plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
    return plucker.reshape(traj.frames, height, width, 6).permute(3, 0, 1, 2)[None].contiguous()
