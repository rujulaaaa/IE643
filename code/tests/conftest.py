import pytest
import torch


@pytest.fixture(scope="session")
def tiny_unet():
    """A randomly initialised UNetMotionModel with AnimateDiff's structure, small enough for CPU."""
    from diffusers import UNetMotionModel

    torch.manual_seed(0)
    unet = UNetMotionModel(
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=("CrossAttnDownBlockMotion", "DownBlockMotion"),
        up_block_types=("UpBlockMotion", "CrossAttnUpBlockMotion"),
        cross_attention_dim=16,
        num_attention_heads=2,
        motion_num_attention_heads=2,
        motion_max_seq_length=8,
        sample_size=8,
    )
    return unet.eval()


class TinyRun:
    """Stands in for a pipeline: B=2 CFG branches, F=4 frames, 8x8 latent, 6 text tokens."""

    B, F, H, W, T = 2, 4, 8, 8, 6

    def __init__(self, unet, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.unet = unet
        self.sample = torch.randn(self.B, 4, self.F, self.H, self.W, generator=g)
        # AnimateDiffPipeline repeats prompt embeddings per frame before the UNet sees them
        self.text = torch.randn(self.B, self.T, 16, generator=g).repeat_interleave(self.F, dim=0)

    @torch.no_grad()
    def __call__(self, steps=3):
        x = self.sample
        for i in range(steps):  # crude stand-in for a sampler: feed the output back in
            x = self.unet(x, torch.tensor(999 - 40 * i), encoder_hidden_states=self.text).sample
        return x


@pytest.fixture
def tiny_run(tiny_unet):
    return lambda seed=0: TinyRun(tiny_unet, seed)
