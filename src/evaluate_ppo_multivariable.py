"""Deterministically evaluate a trained PPO policy for one EnergyPlus year."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from highrise_env_multivariable import HighRiseEnergyPlusEnv


class FixedVentilationWrapper(gym.ActionWrapper):
    """Expose three setpoint actions and append fixed ventilation actions."""

    def __init__(self, env: gym.Env, ventilation_fraction: float = 0.35):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.ventilation_action = np.full(
            3,
            2.0 * ventilation_fraction - 1.0,
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        setpoint_actions = np.asarray(action, dtype=np.float32)
        return np.concatenate(
            [setpoint_actions, self.ventilation_action]
        ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--idf", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument(
        "--energyplus-dir",
        default="/usr/local/EnergyPlus-22-1-0",
    )
    parser.add_argument("--output", default="ppo_annual_evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--setpoint-only", action="store_true")
    parser.add_argument("--ventilation", type=float, default=0.35)
    args = parser.parse_args()

    if not 0.0 <= args.ventilation <= 1.0:
        raise ValueError("--ventilation must be between 0 and 1")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    base_env = HighRiseEnergyPlusEnv(
        idf_path=args.idf,
        weather_path=args.weather,
        energyplus_dir=args.energyplus_dir,
        output_root=output / "energyplus",
        safe_control=True,
        action_penalty_dimension=3 if args.setpoint_only else 6,
    )
    env = (
        FixedVentilationWrapper(base_env, args.ventilation)
        if args.setpoint_only
        else base_env
    )

    # Load the observation statistics learned during training. The raw
    # EnergyPlus environment is stepped directly to avoid automatic episode
    # resetting by DummyVecEnv at the end of the annual simulation.
    vector_environment = DummyVecEnv([lambda: env])
    normalizer = VecNormalize.load(args.vecnormalize, vector_environment)
    normalizer.training = False
    normalizer.norm_reward = False

    model = PPO.load(args.model)

    total_reward = 0.0
    total_electricity_kwh = 0.0
    total_discomfort_degree_hours = 0.0
    total_co2_exceedance_kppm_hours = 0.0
    total_interventions = 0
    total_action_movement = 0.0
    steps = 0
    setpoint_sum = np.zeros(3, dtype=np.float64)
    ventilation_sum = np.zeros(3, dtype=np.float64)
    rows: list[dict] = []

    try:
        observation, _ = env.reset(seed=args.seed)

        while True:
            normalized_observation = normalizer.normalize_obs(
                observation.reshape(1, -1)
            )[0]
            action, _ = model.predict(
                normalized_observation,
                deterministic=True,
            )
            observation, reward, terminated, truncated, info = env.step(action)

            # EnergyPlus may emit a final terminal packet that repeats the last
            # measured timestep. Do not count that packet twice.
            if info.get("simulation_complete", False):
                break

            steps += 1
            total_reward += float(reward)
            total_electricity_kwh += float(info["cooling_electric_kwh"])
            total_discomfort_degree_hours += float(info["discomfort_degree_hours"])
            total_co2_exceedance_kppm_hours += float(
                info["co2_exceedance_kppm_hours"]
            )
            total_interventions += int(info["safety_interventions"])
            total_action_movement += float(info["action_movement"])

            setpoints = np.asarray(info["setpoints_c"], dtype=np.float64)
            ventilation = np.asarray(
                info["ventilation_fraction"], dtype=np.float64
            )
            setpoint_sum += setpoints
            ventilation_sum += ventilation

            rows.append(
                {
                    "step": steps,
                    "month": info["month"],
                    "day": info["day"],
                    "hour": info["hour"],
                    "minute": info["minute"],
                    "reward": float(reward),
                    "temperature_g_c": info["temperatures_c"][0],
                    "temperature_m_c": info["temperatures_c"][1],
                    "temperature_t_c": info["temperatures_c"][2],
                    "co2_g_ppm": info["co2_ppm"][0],
                    "co2_m_ppm": info["co2_ppm"][1],
                    "co2_t_ppm": info["co2_ppm"][2],
                    "setpoint_g_c": setpoints[0],
                    "setpoint_m_c": setpoints[1],
                    "setpoint_t_c": setpoints[2],
                    "ventilation_g": ventilation[0],
                    "ventilation_m": ventilation[1],
                    "ventilation_t": ventilation[2],
                    "cooling_electric_kwh": info["cooling_electric_kwh"],
                    "discomfort_degree_hours": info["discomfort_degree_hours"],
                    "co2_exceedance_kppm_hours": info[
                        "co2_exceedance_kppm_hours"
                    ],
                    "safety_interventions": info["safety_interventions"],
                    "action_movement": info["action_movement"],
                }
            )

            if terminated or truncated:
                break
    finally:
        normalizer.close()

    if steps == 0:
        raise RuntimeError("Evaluation produced no measured timesteps.")

    summary = {
        "model": str(Path(args.model).resolve()),
        "vecnormalize": str(Path(args.vecnormalize).resolve()),
        "seed": args.seed,
        "controller": (
            "setpoint_only_ppo" if args.setpoint_only else "joint_ppo"
        ),
        "fixed_ventilation_fraction": (
            args.ventilation if args.setpoint_only else None
        ),
        "steps": steps,
        "total_reward": total_reward,
        "cooling_electricity_kwh_year": total_electricity_kwh,
        "cooling_electricity_mwh_year": total_electricity_kwh / 1000.0,
        "discomfort_degree_hours_year": total_discomfort_degree_hours,
        "co2_exceedance_kppm_hours_year": total_co2_exceedance_kppm_hours,
        "safety_interventions": total_interventions,
        "intervention_rate_percent_of_control_actions": (
            100.0
            * total_interventions
            / (steps * (3 if args.setpoint_only else 6))
        ),
        "total_action_movement": total_action_movement,
        "mean_setpoints_c_g_m_t": (setpoint_sum / steps).tolist(),
        "mean_ventilation_fraction_g_m_t": (
            ventilation_sum / steps
        ).tolist(),
    }

    trajectory_path = output / "annual_trajectory.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output / "annual_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved trajectory: {trajectory_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
