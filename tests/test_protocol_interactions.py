import pandas as pd

from facter.data.protocol import ProtocolConfig, build_interactions_ml


def test_build_interactions_min_history():
    ratings = pd.DataFrame(
        {
            "uid": [1, 1, 1, 1, 1, 1],
            "mid": [10, 11, 12, 13, 14, 15],
            "rating": [5, 5, 5, 5, 5, 5],
            "timestamp": [1, 2, 3, 4, 5, 6],
        }
    )
    users = pd.DataFrame(
        {"uid": [1], "gender": ["M"], "age": [25], "occupation": [3], "zip": ["12345"]}
    )
    item_db = {m: {"title": f"Movie {m}", "genres": "X"} for m in range(10, 16)}

    cfg = ProtocolConfig(min_history=5, sample_interactions=10)
    inter = build_interactions_ml(ratings, users, item_db, cfg)

    # With 6 items and min_history=5, only 1 interaction (history of 5, target the 6th)
    assert len(inter) == 1
    row = inter.iloc[0]
    assert row["history_mids"] == [10, 11, 12, 13, 14]
    assert row["target_mid"] == 15
