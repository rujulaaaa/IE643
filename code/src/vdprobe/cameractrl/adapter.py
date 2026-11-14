"""CameraCtrl's injection into the motion modules, as hooks on a diffusers UNetMotionModel.

In the authors' model the first temporal self-attention of every motion module runs a
`PoseAdaptorAttnProcessor`: its input h (layer-normed tokens plus temporal positions, laid out
(batch*h*w, F, C)) is replaced by

    h + scale * qkv_merge(h + pose_feature)

and that tensor serves as query, key and value. Here the same replacement is done by a forward
pre-hook on `<block>.motion_modules.<j>.transformer_blocks.0.attn1`, whose first argument is
exactly that h. Attention processors are left alone, so `edit.InjectingProcessor` still works.

With no camera set (or scale 0) a hook returns nothing and the U-Net is bit-identical to one
without the adapter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .rays import Trajectory, plucker_embedding, trajectory

SITE = re.compile(r"^(down_blocks|up_blocks)\.(\d+)\.motion_modules\.\d+\.transformer_blocks\.0\.attn1$|"
                  r"^(mid_block)\.motion_modules\.\d+\.transformer_blocks\.0\.attn1$")


@dataclass(frozen=True)
class CameraSpec:
    """What the camera input is for one U-Net call.

    `block_scales` and `block_trajectories` are keyed by module-name prefixes ("up_blocks.1",
    "down_blocks", "" for everything); the longest matching prefix wins. A block scale multiplies
    `scale`.
    """
    trajectory: str | Trajectory
    scale: float = 1.0
    block_scales: dict = field(default_factory=dict)
    block_trajectories: dict = field(default_factory=dict)


def _longest_prefix(table: dict, name: str, default):
    hits = [k for k in table if name.startswith(k)]
    return table[max(hits, key=len)] if hits else default


def merge_key(site: str) -> str:
    return site.replace(".", "__")


class CameraAdapter(nn.Module):
    def __init__(self, unet, encoder):
        super().__init__()
        self.encoder = encoder
        self._unet = [unet]                       # in a list: not a submodule, so not in state_dict
        n_levels = len(unet.down_blocks)
        self.sites: list[str] = []
        self.level: dict[str, int] = {}
        merges = {}
        for name, module in unet.named_modules():
            m = SITE.match(name)
            if not m:
                continue
            if m.group(3):
                lvl = n_levels - 1
            else:
                lvl = int(m.group(2)) if m.group(1) == "down_blocks" else n_levels - 1 - int(m.group(2))
            width = module.to_q.in_features
            if width != encoder.channels[lvl]:
                raise ValueError(f"{name}: width {width} but pose feature {lvl} has {encoder.channels[lvl]} channels")
            lin = nn.Linear(width, width)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            merges[merge_key(name)] = lin
            self.sites.append(name)
            self.level[name] = lvl
        if not self.sites:
            raise ValueError("no motion-module temporal attention found in this U-Net")
        self.merges = nn.ModuleDict(merges)
        self._handles: list = []
        self._active: dict[str, tuple[torch.Tensor, float]] = {}
        self._cache: dict = {}
        self.spec: CameraSpec | None = None
        self.calls = 0

    # -- hooks ------------------------------------------------------------------------------

    def attach(self) -> "CameraAdapter":
        if self._handles:
            return self
        modules = dict(self._unet[0].named_modules())
        for site in self.sites:
            self._handles.append(modules[site].register_forward_pre_hook(self._make_hook(site), with_kwargs=True))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, site: str):
        merge = self.merges[merge_key(site)]

        def hook(module, args, kwargs):
            active = self._active.get(site)
            if active is None:
                return None
            pose, scale = active
            h = args[0] if args else kwargs["hidden_states"]
            if h.shape[1:] != pose.shape[1:] or h.shape[0] % pose.shape[0]:
                raise ValueError(f"{site}: tokens {tuple(h.shape)} do not fit the pose feature {tuple(pose.shape)}; "
                                 f"was the camera set for this clip's height, width and frames?")
            pose = pose.to(h.dtype).repeat(h.shape[0] // pose.shape[0], 1, 1)
            new = h + scale * merge(h + pose)
            self.calls += 1
            if args:
                return (new, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": new}

        return hook

    # -- state ------------------------------------------------------------------------------

    @torch.no_grad()
    def features(self, traj: str | Trajectory, height: int, width: int, frames: int) -> list[torch.Tensor]:
        """Pose features of a trajectory, one (h*w, F, C) tensor per level, cached."""
        if isinstance(traj, str):
            traj = trajectory(traj, frames)
        if traj.frames != frames:
            raise ValueError(f"trajectory {traj.name!r} has {traj.frames} frames, the clip has {frames}")
        key = (traj.name, traj.focal, traj.c2w.numpy().tobytes(), height, width)
        if key not in self._cache:
            p = next(self.encoder.parameters())
            emb = plucker_embedding(traj, height, width).to(p.device, p.dtype)
            self._cache[key] = [f[0].permute(2, 3, 1, 0).flatten(0, 1) for f in self.encoder(emb)]   # (h*w, F, C)
        return self._cache[key]

    def set(self, spec: CameraSpec | None, height: int = 0, width: int = 0, frames: int = 0) -> None:
        """Camera for the next U-Net calls; `None` switches it off."""
        self.spec = spec
        self._active = {}
        if spec is None:
            return
        for site in self.sites:
            scale = spec.scale * _longest_prefix(spec.block_scales, site, 1.0)
            if scale == 0:
                continue
            traj = _longest_prefix(spec.block_trajectories, site, spec.trajectory)
            self._active[site] = (self.features(traj, height, width, frames)[self.level[site]], float(scale))

    def clear_cache(self) -> None:
        self._cache = {}
