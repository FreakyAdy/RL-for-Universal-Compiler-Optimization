"""Reward computation tests."""

from rl_uco.env.reward import compute_reward
from rl_uco.config import FAILURE_REWARD


def test_faster_is_better():
    r = compute_reward(50, 10, 100, 20, correct=True)
    assert r > compute_reward(100, 20, 100, 20, correct=True)


def test_incorrect_penalty():
    r = compute_reward(50, 10, 100, 20, correct=False)
    assert r == FAILURE_REWARD
