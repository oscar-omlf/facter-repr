import pandas as pd

from facter.data.protocol import ProtocolConfig, sample_and_split


def test_sample_and_split_deterministic():
    interactions = pd.DataFrame(
        {
            "uid": list(range(100)),
            "gender": ["M"] * 50 + ["F"] * 50,
            "age": [25] * 100,
            "occupation": [1] * 100,
            "history_mids": [[1, 2, 3, 4, 5]] * 100,
            "history_titles": [["A", "B", "C", "D", "E"]] * 100,
            "target_mid": [6] * 100,
            "target_title": ["F"] * 100,
        }
    )
    cfg = ProtocolConfig(sample_interactions=50, test_size=0.3, seed=42, stratify=True)

    cal1, test1 = sample_and_split(interactions, cfg)
    cal2, test2 = sample_and_split(interactions, cfg)

    assert len(cal1) == len(cal2)
    assert len(test1) == len(test2)
    assert set(test1["uid"]) == set(test2["uid"])
    assert set(cal1["uid"]).isdisjoint(set(test1["uid"]))
