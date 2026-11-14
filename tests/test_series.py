import pytest

from rlpredictor.model.series import (
    game_prob_from_series,
    length_buckets,
    score_distribution,
    series_win_prob,
)


@pytest.mark.parametrize("bo", [1, 3, 5, 7])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.63, 0.9])
def test_distribution_sums_to_one(p, bo):
    d = score_distribution(p, bo)
    assert sum(d.values()) == pytest.approx(1.0)
    assert sum(length_buckets(d, bo).values()) == pytest.approx(1.0)


def test_symmetry_and_known_values():
    d = score_distribution(0.5, 7)
    assert d[(4, 0)] == pytest.approx(d[(0, 4)]) == pytest.approx(1 / 16)
    assert series_win_prob(0.5, 7) == pytest.approx(0.5)
    assert series_win_prob(0.7, 1) == pytest.approx(0.7)
    # Bo3 at p: p^2 + 2 p^2 q
    assert series_win_prob(0.6, 3) == pytest.approx(0.36 + 2 * 0.36 * 0.4)
    # Longer series favour the stronger team
    assert series_win_prob(0.6, 7) > series_win_prob(0.6, 5) > series_win_prob(0.6, 3)


def test_inverse():
    for bo in (3, 5, 7):
        assert game_prob_from_series(series_win_prob(0.58, bo), bo) == pytest.approx(0.58, abs=1e-6)


def test_rejects_even():
    with pytest.raises(ValueError):
        score_distribution(0.5, 4)
