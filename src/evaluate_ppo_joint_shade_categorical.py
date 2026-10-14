"""Evaluate factorized categorical PPO for one EnergyPlus year."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from highrise_env_multivariable_blinds import HighRiseEnergyPlusEnv
from multidiscrete_joint_shade_wrapper import MultiDiscreteJointShadeWrapper


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
    parser.add_argument(
        "--output",
        default="ppo_joint_shade_categorical_evaluation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shade-mode",
        choices=("policy", "retracted", "deployed"),
        default="policy",
        help=(
            "Use PPO shade actions, force requested shades retracted, or "
            "force requested shades deployed. The safety shield remains active."
        ),
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    base_environment = HighRiseEnergyPlusEnv(
        idf_path=args.idf,
        weather_path=args.weather,
        energyplus_dir=args.energyplus_dir,
        output_root=output / "energyplus",
        safe_control=True,
        action_penalty_dimension=9,
    )
    environment = MultiDiscreteJointShadeWrapper(base_environment)

    # VecNormalize is used only to reproduce the observation transformation
    # learned during training. The underlying environment is stepped directly
    # so DummyVecEnv cannot automatically reset at annual termination.
    vector_environment = DummyVecEnv([lambda: environment])
    normalizer = VecNormalize.load(args.vecnormalize, vector_environment)
    normalizer.training = False
    normalizer.norm_reward = False

    model = PPO.load(args.model, device="auto")

    total_reward = 0.0
    total_electricity_kwh = 0.0
    total_discomfort_degree_hours = 0.0
    total_co2_exceedance_kppm_hours = 0.0
    total_interventions = 0
    total_action_movement = 0.0
    steps = 0

    setpoint_sum = np.zeros(3, dtype=np.float64)
    ventilation_sum = np.zeros(3, dtype=np.float64)
    requested_shade_sum = np.zeros(3, dtype=np.float64)
    applied_shade_sum = np.zeros(3, dtype=np.float64)
    reported_shade_sum = np.zeros(3, dtype=np.float64)
    shade_status_absolute_error_sum = np.zeros(3, dtype=np.float64)
    rows: list[dict] = []

    try:
        observation, _ = environment.reset(seed=args.seed)

        while True:
            normalized_observation = normalizer.normalize_obs(
                observation.reshape(1, -1)
            )[0]
            action, _ = model.predict(
                normalized_observation,
                deterministic=True,
            )
            action = np.asarray(action, dtype=np.int64).reshape(9).copy()
            if args.shade_mode == "retracted":
                action[6:9] = 0
            elif args.shade_mode == "deployed":
                action[6:9] = 1
            observation, reward, terminated, truncated, info = (
                environment.step(action)
            )

            # Ignore the terminal packet if it repeats the final measured step.
            if info.get("simulation_complete", False):
                break

            steps += 1
            total_reward += float(reward)
            total_electricity_kwh += float(info["cooling_electric_kwh"])
            total_discomfort_degree_hours += float(
                info["discomfort_degree_hours"]
            )
            total_co2_exceedance_kppm_hours += float(
                info["co2_exceedance_kppm_hours"]
            )
            total_interventions += int(info["safety_interventions"])
            total_action_movement += float(info["action_movement"])

            setpoints = np.asarray(info["setpoints_c"], dtype=np.float64)
            requested_setpoints = np.asarray(
                info["requested_setpoints_c"], dtype=np.float64
            )
            ventilation = np.asarray(
                info["ventilation_fraction"], dtype=np.float64
            )
            requested_ventilation = np.asarray(
                info["requested_ventilation_fraction"], dtype=np.float64
            )
            requested_shade = np.asarray(
                info["requested_blind_closure_fraction"], dtype=np.float64
            )
            applied_shade = np.asarray(
                info["blind_closure_fraction"], dtype=np.float64
            )
            reported_shade = np.asarray(
                info["blind_status_fraction"], dtype=np.float64
            )

            setpoint_sum += setpoints
            ventilation_sum += ventilation
            requested_shade_sum += requested_shade
            applied_shade_sum += applied_shade
            reported_shade_sum += reported_shade
            shade_status_absolute_error_sum += np.abs(
                reported_shade - applied_shade
            )

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
                    "occupancy_g": info["occupancies"][0],
                    "occupancy_m": info["occupancies"][1],
                    "occupancy_t": info["occupancies"][2],
                    "co2_g_ppm": info["co2_ppm"][0],
                    "co2_m_ppm": info["co2_ppm"][1],
                    "co2_t_ppm": info["co2_ppm"][2],
                    "solar_w_m2": info["solar_w_m2"],
                    "requested_setpoint_g_c": requested_setpoints[0],
                    "requested_setpoint_m_c": requested_setpoints[1],
                    "requested_setpoint_t_c": requested_setpoints[2],
                    "setpoint_g_c": setpoints[0],
                    "setpoint_m_c": setpoints[1],
                    "setpoint_t_c": setpoints[2],
                    "requested_ventilation_g": requested_ventilation[0],
                    "requested_ventilation_m": requested_ventilation[1],
                    "requested_ventilation_t": requested_ventilation[2],
                    "ventilation_g": ventilation[0],
                    "ventilation_m": ventilation[1],
                    "ventilation_t": ventilation[2],
                    "requested_shade_g": requested_shade[0],
                    "requested_shade_m": requested_shade[1],
                    "requested_shade_t": requested_shade[2],
                    "shade_command_g": applied_shade[0],
                    "shade_command_m": applied_shade[1],
                    "shade_command_t": applied_shade[2],
                    "shade_status_g": reported_shade[0],
                    "shade_status_m": reported_shade[1],
                    "shade_status_t": reported_shade[2],
                    "cooling_electric_kwh": info["cooling_electric_kwh"],
                    "discomfort_degree_hours": info[
                        "discomfort_degree_hours"
                    ],
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
        "controller": "categorical_joint_setpoint_ventilation_binary_shade_ppo",
        "shade_mode": args.shade_mode,
        "safety_shield_active": True,
        "action_dimension": 9,
        "action_space_nvec": [15, 15, 15, 11, 11, 11, 2, 2, 2],
        "steps": steps,
        "total_reward": total_reward,
        "cooling_electricity_kwh_year": total_electricity_kwh,
        "cooling_electricity_mwh_year": total_electricity_kwh / 1000.0,
        "discomfort_degree_hours_year": total_discomfort_degree_hours,
        "co2_exceedance_kppm_hours_year": (
            total_co2_exceedance_kppm_hours
        ),
        "safety_interventions": total_interventions,
        "intervention_rate_percent_of_control_actions": (
            100.0 * total_interventions / (steps * 9)
        ),
        "total_action_movement": total_action_movement,
        "mean_setpoints_c_g_m_t": (setpoint_sum / steps).tolist(),
        "mean_ventilation_fraction_g_m_t": (
            ventilation_sum / steps
        ).tolist(),
        "mean_requested_shade_fraction_g_m_t": (
            requested_shade_sum / steps
        ).tolist(),
        "shade_deployment_rate_g_m_t": (
            applied_shade_sum / steps
        ).tolist(),
        "mean_reported_shade_status_g_m_t": (
            reported_shade_sum / steps
        ).tolist(),
        "mean_absolute_shade_command_status_error_g_m_t": (
            shade_status_absolute_error_sum / steps
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
