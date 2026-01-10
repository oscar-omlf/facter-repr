from pathlib import Path

from facter.data.movielens import load_ml1m


def test_load_ml1m_from_tiny_files(tmp_path: Path):
    # Create tiny ratings/users/movies in MovieLens :: format
    (tmp_path / "ratings.dat").write_text(
        "1::10::5::100\n1::11::4::200\n2::10::5::150\n",
        encoding="utf-8",
    )
    (tmp_path / "users.dat").write_text(
        "1::M::25::3::12345\n2::F::35::7::54321\n",
        encoding="utf-8",
    )
    (tmp_path / "movies.dat").write_text(
        "10::Movie A (2000)::Action|Comedy\n11::Movie B (2001)::Drama\n",
        encoding="latin-1",
    )

    frames = load_ml1m(tmp_path)
    assert set(frames.ratings.columns) == {"uid", "mid", "rating", "timestamp"}
    assert set(frames.users.columns) >= {"uid", "gender", "age", "occupation", "zip"}
    assert set(frames.movies.columns) == {"mid", "title", "genres"}
    assert len(frames.ratings) == 3
