"""Camera MotionLoRAs (pan, tilt, zoom, roll) as a per-track switch in the lockstep loop.

The eight official AnimateDiff camera LoRAs are attached once under their camera names. Each
track then names its camera in `Track.meta`:

    meta["camera"]              a camera name, None / "none" (no LoRA), or a function step -> name | None
    meta["camera_scale"]        LoRA weight, default 1.0
    meta["camera_block_scales"] {module-name prefix: factor}, longest prefix wins; e.g.
                                {"": 0.0, "up_blocks.1": 1.0} keeps the LoRA in one block only

    run_lockstep(loaded, coords, tracks, noise, before_track=camera_switcher(loaded.unet))

A track without a camera runs with every adapter disabled, which is bit-identical to a U-Net
that has no adapter attached.
"""

from __future__ import annotations

OFF = (None, "none", "off")


def load_camera_loras(pipe, cameras, repo_template: str = "guoyww/animatediff-motion-lora-{camera}") -> list[str]:
    """Attach one adapter per camera (named after it); leaves all of them disabled. Returns the names."""
    names = [c for c in cameras if c not in OFF]
    for c in names:
        pipe.load_lora_weights(repo_template.format(camera=c), adapter_name=c)
    pipe.unet.disable_adapters()
    return names


def _lora_layers(unet, adapter: str):
    for name, module in unet.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter in scaling:
            yield name, module


def set_block_scales(unet, adapter: str, scales: dict[str, float]) -> int:
    """Multiply the adapter's current scaling per layer by the factor of the longest matching
    name prefix (no match: unchanged). `set_adapters` resets the scaling, so call this after it;
    `use_camera` does both. Returns the number of layers whose factor is not 1."""
    n = 0
    for name, module in _lora_layers(unet, adapter):
        hits = [k for k in scales if name.startswith(k)]
        factor = scales[max(hits, key=len)] if hits else 1.0
        if factor != 1.0:
            module.scaling[adapter] = module.scaling[adapter] * factor
            n += 1
    return n


def use_camera(unet, camera: str | None, scale: float = 1.0, block_scales: dict[str, float] | None = None) -> None:
    """Make `camera` the only active adapter at `scale`; None / "none" disables every adapter."""
    if camera in OFF:
        unet.disable_adapters()
        return
    if not any(True for _ in _lora_layers(unet, camera)):
        raise KeyError(f"no adapter named {camera!r} is attached")
    unet.enable_adapters()
    unet.set_adapters([camera], weights=[scale])        # also resets every layer's scaling
    if block_scales:
        set_block_scales(unet, camera, block_scales)


def camera_switcher(unet):
    """`before_track(step, track)` for `run_lockstep`; see the module docstring."""

    def before_track(step: int, track) -> None:
        camera = track.meta.get("camera")
        if callable(camera):
            camera = camera(step)
        use_camera(unet, camera, track.meta.get("camera_scale", 1.0), track.meta.get("camera_block_scales"))

    return before_track
