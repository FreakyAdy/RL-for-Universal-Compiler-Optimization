"""Reward computation from hardware metrics."""

from __future__ import annotations

from rl_uco.config import FAILURE_REWARD, REWARD_ENERGY_WEIGHT, REWARD_TIME_WEIGHT


def compute_reward(
    wall_time_ns: float,
    energy_j: float,
    baseline_wall_time_ns: float,
    baseline_energy_j: float,
    correct: bool = True,
) -> float:
    if not correct:
        return FAILURE_REWARD
    if baseline_wall_time_ns <= 0:
        return FAILURE_REWARD
    time_ratio = wall_time_ns / baseline_wall_time_ns
    if baseline_energy_j > 0 and energy_j > 0:
        energy_ratio = energy_j / baseline_energy_j
    else:
        energy_ratio = time_ratio
    cost = REWARD_TIME_WEIGHT * time_ratio + REWARD_ENERGY_WEIGHT * energy_ratio
    return -cost
