from facter.data.prompts import PromptConfig, build_generation_prompt, build_ranking_prompt


def test_generation_prompt_contains_demographics_and_history():
    row = {
        "gender": "M",
        "age": 25,
        "occupation": 3,
        "history_titles": ["A", "B", "C", "D", "E"],
    }
    cfg = PromptConfig(k_recs=10, include_demographics=True, domain="movie")
    p = build_generation_prompt(row, cfg)
    assert "User demographics" in p
    assert "History" in p
    assert "Recommend the next 10 movies" in p


def test_ranking_prompt_contains_candidates():
    row = {
        "gender": "F",
        "age": 35,
        "occupation": 7,
        "history_titles": ["A", "B", "C", "D", "E"],
    }
    candidates = ["X", "Y", "Z"]
    cfg = PromptConfig(k_recs=10, include_demographics=True, domain="movie")
    p = build_ranking_prompt(row, candidates, cfg)
    assert "Candidates" in p
    assert "Rank the candidates" in p
