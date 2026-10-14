"""Discrete action wrapper for joint setpoint, ventilation, and shade control."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


SETPOINT_LEVELS_C = np.arange(22.0, 29.0 + 0.5, 0.5, dtype=np.float32)
VENTILATION_LEVELS = np.arange(0.0, 1.0 + 0.1, 0.1, dtype=np.float32)
ACTION_LEVELS = np.asarray(
    [len(SETPOINT_LEVELS_C)] * 3
    + [len(VENTILATION_LEVELS)] * 3
    + [2] * 3,
    dtype=np.int64,
)


class MultiDiscreteJointShadeWrapper(gym.ActionWrapper):
    """Map nine categorical actions to the environment's normalized Box action.

    Action ordering is ground, middle, and top for each actuator group:

    * indices 0--2: cooling setpoint, 15 levels from 22 to 29 degC;
    * indices 3--5: ventilation fraction, 11 levels from 0 to 1;
    * indices 6--8: binary shade state, 0 retracted and 1 deployed.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete(ACTION_LEVELS.copy())

        base_env = env.unwrapped
        self._minimum_setpoint = float(base_env.min_setpoint)
        self._maximum_setpoint = float(base_env.max_setpoint)
        if not np.isclose(self._minimum_setpoint, SETPOINT_LEVELS_C[0]):
            raise ValueError(
                "The wrapped environment minimum setpoint must be 22 degC."
            )
        if not np.isclose(self._maximum_setpoint, SETPOINT_LEVELS_C[-1]):
            raise ValueError(
                "The wrapped environment maximum setpoint must be 29 degC."
            )

    def action(self, action: np.ndarray) -> np.ndarray:
        discrete_action = np.asarray(action, dtype=np.int64)
        if discrete_action.shape != (9,):
            raise ValueError(
                f"Expected a nine-element action, received {discrete_action.shape}."
            )
        if not self.action_space.contains(discrete_action):
            raise ValueError(f"Invalid MultiDiscrete action: {discrete_action}")

        setpoints_c = SETPOINT_LEVELS_C[discrete_action[:3]]
        setpoint_actions = (
            2.0
            * (setpoints_c - self._minimum_setpoint)
            / (self._maximum_setpoint - self._minimum_setpoint)
            - 1.0
        )

        ventilation = VENTILATION_LEVELS[discrete_action[3:6]]
        ventilation_actions = 2.0 * ventilation - 1.0

        shade_actions = 2.0 * discrete_action[6:9].astype(np.float32) - 1.0

        return np.concatenate(
            [setpoint_actions, ventilation_actions, shade_actions]
        ).astype(np.float32)

