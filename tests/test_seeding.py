import numpy as np
import torch

from facter.utils.seeding import SeedConfig, seed_all


def test_seed_all_cpu_repro():
    seed_all(SeedConfig(seed=123, deterministic=True, warn_only=True))
    a1 = np.random.rand(5)
    t1 = torch.rand(5)

    seed_all(SeedConfig(seed=123, deterministic=True, warn_only=True))
    a2 = np.random.rand(5)
    t2 = torch.rand(5)

    assert np.allclose(a1, a2)
    assert torch.allclose(t1, t2)


def test_seed_all_cuda_repro_if_available():
    if not torch.cuda.is_available():
        return

    seed_all(SeedConfig(seed=999, deterministic=True, warn_only=True))
    c1 = torch.rand(5, device="cuda")

    seed_all(SeedConfig(seed=999, deterministic=True, warn_only=True))
    c2 = torch.rand(5, device="cuda")

    assert torch.allclose(c1, c2)
