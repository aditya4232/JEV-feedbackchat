"""Check annotation coverage and boundary decisions without scoring a model."""

from __future__ import annotations

from evaluation.run_eval import DEFAULT_CASES_PATH, Confusion, load_cases


def test_evaluation_set_covers_requested_contextual_boundaries() -> None:
    cases = {case.id: case for case in load_cases(DEFAULT_CASES_PATH)}

    # Same ambiguous wording gets different deception labels only when the
    # surrounding text explicitly establishes a false claim; approval alone does not.
    assert cases["dec-pretend"].labels["deception"] == 0
    assert cases["dec-meeting"].labels["deception"] == 1
    assert cases["neg-pretend-ambiguous"].labels["deception"] == 0

    # These are provisional independently reviewed annotation cases, not output-quality claims about the
    # offline fixture mock or the keyword baseline.
    assert cases["neg-quoted-lie"].labels["deception"] == 0
    assert cases["negated-concealment"].labels["deception"] == 0
    assert cases["neg-discuss-trait"].labels["cunning"] == 0
    assert cases["neg-sarcasm"].labels["happy"] == 0
    assert cases["neg-sarcasm"].labels["sad"] == 0


def test_taxonomy_boundaries_do_not_collapse_strategy_into_cunning() -> None:
    cases = {case.id: case for case in load_cases(DEFAULT_CASES_PATH)}

    assert cases["sad-frustrated"].labels["sad"] == 0
    assert cases["cun-butter"].labels["cunning"] == 0
    assert cases["cun-loophole"].labels["cunning"] == 0
    assert cases["cun-guilt"].labels["cunning"] == 1


def test_f1_is_zero_for_missed_positives_and_undefined_without_support() -> None:
    assert Confusion(tp=0, fn=3).f1() == 0.0
    assert Confusion(tp=0, fp=1, fn=3).f1() == 0.0
    assert Confusion(tn=10).f1() is None
