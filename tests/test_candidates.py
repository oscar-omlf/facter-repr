import numpy as np
import pandas as pd

from facter.data.protocol import ProtocolConfig, build_candidate_sets


def test_candidate_set_includes_target_and_excludes_history():
    df = pd.DataFrame(
        {
            "history_mids": [[1, 2, 3, 4, 5]],
            "target_mid": [6],
        }
    )
    item_pool = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cfg = ProtocolConfig(seed=123, n_candidates=6)

    out = build_candidate_sets(df, item_pool=item_pool, cfg=cfg)
    cands = out.iloc[0]["candidate_mids"]
    assert len(cands) == 6
    assert 6 in cands
    assert not any(x in {1, 2, 3, 4, 5} for x in cands)
