"""Evaluate fixed and occupancy-rule baselines in the six-action environment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from highrise_env_multivariable import HighRiseEnergyPlusEnv


MIN_SETPOINT = 22.0
MAX_SETPOINT = 29.0


def setpoint_to_action(setpoints_c: np.ndarray) -> np.ndarray:
    return 2.0 * (setpoints_c - MIN_SETPOINT) / (
        MAX_SETPOINT - MIN_SETPOINT
    ) - 1.0


def fraction_to_action(fractions: np.ndarray) -> np.ndarray:
    return 2.0 * fractions - 1.0


def controller_action(name: str, observation: np.ndarray) -> np.ndarray:
    if name == "fixed_24_vent35":
        setpoints = np.full(3, 24.0, dtype=np.float32)
        ventilation = np.full(3, 0.35, dtype=np.float32)
    elif name == "occupancy_rule":
        occupied = observation[6:9] > 0.0
        setpoints = np.where(occupied, 24.0, 28.0).astype(np.float32)
        ventilation = np.where(occupied, 0.35, 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown controller: {name}")

    return np.concatenate(
        [
            setpoint_to_action(setpoints),
            fraction_to_action(ventilation),
        ]
    ).astype(np.float32)


def evaluate_controller(
    name: str,
    args: argparse.Namespace,
    output: Path,
) -> dict:
    controller_output = output / name
    controller_output.mkdir(parents=True, exist_ok=True)

    env = HighRiseEnergyPlusEnv(
        idf_path=args.idf,
        weather_path=args.weather,
        energyplus_dir=args.energyplus_dir,
        output_root=controller_output / "energyplus",
        safe_control=False,
    )

    totals = {
        "reward": 0.0,
        "electricity_kwh": 0.0,
        "discomfort_degree_hours": 0.0,
        "co2_exceedance_kppm_hours": 0.0,
        "action_movement": 0.0,
    }
    setpoint_sum = np.zeros(3, dtype=np.float64)
    ventilation_sum = np.zeros(3, dtype=np.float64)
    rows: list[dict] = []
    steps = 0

    try:
        observation, _ = env.reset(seed=args.seed)

        while True:
            action = controller_action(name, observation)
            observation, reward, terminated, truncated, info = env.step(action)

            # Do not double-count the final fallback packet.
            if info.get("simulation_complete", False):
                break

            steps += 1
            totals["reward"] += float(reward)
            totals["electricity_kwh"] += float(info["cooling_electric_kwh"])
            totals["discomfort_degree_hours"] += float(
                info["discomfort_degree_hours"]
            )
            totals["co2_exceedance_kppm_hours"] += float(
                info["co2_exceedance_kppm_hours"]
            )
            totals["action_movement"] += float(info["action_movement"])

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
                    "occupancy_g": info["occupancies"][0],
                    "occupancy_m": info["occupancies"][1],
                    "occupancy_t": info["occupancies"][2],
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
                    "action_movement": info["action_movement"],
                }
            )

            if terminated or truncated:
                break
    finally:
        env.close()

    if steps == 0:
        raise RuntimeError(f"{name} produced no measured timesteps.")

    summary = {
        "controller": name,
        "steps": steps,
        "total_reward": totals["reward"],
        "cooling_electricity_kwh_year": totals["electricity_kwh"],
        "cooling_electricity_mwh_year": totals["electricity_kwh"] / 1000.0,
        "discomfort_degree_hours_year": totals["discomfort_degree_hours"],
        "co2_exceedance_kppm_hours_year": totals[
            "co2_exceedance_kppm_hours"
        ],
        "safety_interventions": 0,
        "intervention_rate_percent_of_control_actions": 0.0,
        "total_action_movement": totals["action_movement"],
        "mean_setpoints_c_g_m_t": (setpoint_sum / steps).tolist(),
        "mean_ventilation_fraction_g_m_t": (
            ventilation_sum / steps
        ).tolist(),
    }

    with (controller_output / "annual_trajectory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (controller_output / "annual_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idf", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument(
        "--energyplus-dir",
        default="/usr/local/EnergyPlus-22-1-0",
    )
    parser.add_argument("--output", default="multivariable_baselines")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    controllers = ("fixed_24_vent35", "occupancy_rule")
    summaries = [
        evaluate_controller(name, args, output) for name in controllers
    ]

    comparison_fields = [
        "controller",
        "steps",
        "total_reward",
        "cooling_electricity_mwh_year",
        "discomfort_degree_hours_year",
        "co2_exceedance_kppm_hours_year",
        "total_action_movement",
    ]
    with (output / "baseline_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=comparison_fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {field: summary[field] for field in comparison_fields}
            )

    print(f"Saved comparison: {output / 'baseline_comparison.csv'}")


if __name__ == "__main__":
    main()
