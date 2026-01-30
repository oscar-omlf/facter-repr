from facter.prompting.repair import PromptRepairConfig, PromptRepairEngine


def test_prompt_repair_injects_rule_after_three_genre_repeats():
    item_db = {
        1: {"title": "A", "genres": "Romance|Drama"},
        2: {"title": "B", "genres": "Romance"},
        3: {"title": "C", "genres": "Romance|Comedy"},
    }
    cfg = PromptRepairConfig(
        buffer_size=50,
        protected_cols=("gender",),
        min_feature_count=3,
        keying="per_attr",
    )
    engine = PromptRepairEngine(cfg, item_db=item_db)

    # Three violations for gender=F, Romance appears each time
    engine.add_violation(attrs={"gender": "F"}, pred_mid=1)
    engine.add_violation(attrs={"gender": "F"}, pred_mid=2)
    engine.add_violation(attrs={"gender": "F"}, pred_mid=3)

    rules = engine.mine_avoid_rules(current_attrs={"gender": "F"})
    assert any("(gender=F)" in r and "Romance" in r for r in rules)

    sys = engine.build_system_prompt(attrs={"gender": "F"}, q_alpha=0.123, iteration=1, max_iterations=5)
    assert "Avoid: (gender=F)" in sys
