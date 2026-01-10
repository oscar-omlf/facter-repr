import numpy as np
import pandas as pd

from facter.fairness.neighbors import CrossGroupNeighborIndex, NeighborConfig


def test_cross_group_neighbors_and_tau_rho():
    # 4 points in 2D, already normalized
    emb = np.array([
        [1.0, 0.0],   # i0
        [0.9, 0.1],   # i1 close to i0
        [0.0, 1.0],   # i2 orthogonal
        [0.0, 1.0],   # i3 same as i2
    ], dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    df = pd.DataFrame({
        "gender": ["M", "M", "F", "F"],
        "age": [25, 25, 25, 25],
        "occupation": [1, 1, 1, 1],
    })

    cfg = NeighborConfig(protected_cols=("gender",), tau_rho=0.8, top_k=None)
    idx = CrossGroupNeighborIndex(cfg)
    idx.fit(df, emb)

    # For i0 (M), cross-group are i2,i3 (F); sims are 0 -> not eligible for tau_rho=0.8
    elig0 = idx.eligible_neighbors_for_delta(0)
    assert elig0.size == 0

    # For i2 (F), cross-group are i0,i1; sim(i2,i0)=0, sim(i2,i1)=~0.11 -> none eligible
    elig2 = idx.eligible_neighbors_for_delta(2)
    assert elig2.size == 0
