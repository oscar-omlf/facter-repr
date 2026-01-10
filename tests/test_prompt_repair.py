from facter.prompting.repair import PromptRepairConfig, PromptRepairEngine


def test_prompt_repair_injects_rule_after_three_genre_repeats():
    item_db = {
        1: {"title": "A", "genres": "Romance|Drama"},
        2: {"title": "B", "genres": "Romance"},
        3: {"title": "C", "genres": "Romance|Comedy"},
    }
    cfg = PromptRepairConfig(buffer_size=50, protected_key="gender", min_feature_count=3)
    engine = PromptRepairEngine(cfg, item_db=item_db)

    # Three violations for gender=F, all Romance appears
    engine.add_violation("F", 1)
    engine.add_violation("F", 2)
    engine.add_violation("F", 3)

    rules = engine.mine_avoid_rules("F")
    assert any("gender=F" in r and "Romance" in r for r in rules)

    sys = engine.build_system_prompt(a_value="F", q_alpha=0.123, iteration=1, max_iterations=5)
    assert "Avoid: (gender=F)" in sys
