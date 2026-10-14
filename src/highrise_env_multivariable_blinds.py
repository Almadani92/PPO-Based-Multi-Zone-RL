from __future__ import annotations

"""Nine-action EnergyPlus environment for the journal extension.

Actions, in order, are three cooling setpoints, three ventilation fractions,
and three binary interior-shade deployment commands for the representative
ground, middle, and top apartments.
"""

import os
import queue
import shutil
import sys
import threading
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ZONES = ("G SW APARTMENT", "M SW APARTMENT", "T SW APARTMENT")
IDEAL_SYSTEMS = tuple(f"{zone} IDEAL LOADS" for zone in ZONES)
VENT_SCHEDULES = (
    "RL VENTILATION FRACTION G",
    "RL VENTILATION FRACTION M",
    "RL VENTILATION FRACTION T",
)
BLIND_DEPLOY_SCHEDULES = (
    "RL BLIND DEPLOY G",
    "RL BLIND DEPLOY M",
    "RL BLIND DEPLOY T",
)
CONTROLLED_WINDOWS = ("GWINDOW3", "MWINDOW3", "TWINDOW3")

ENV_VERSION = "2026.09.23-journal-nine-action-blind-audit-v1"


def load_energyplus_api(energyplus_dir: str | Path):
    energyplus_dir = Path(energyplus_dir).expanduser().resolve()
    candidates = [
        energyplus_dir / "pyenergyplus" / "api.py",
        energyplus_dir / "Python" / "pyenergyplus" / "api.py",
    ]
    api_file = next((path for path in candidates if path.is_file()), None)
    if api_file is None:
        discovered = list(energyplus_dir.rglob("pyenergyplus/api.py"))
        api_file = discovered[0] if discovered else None
    if api_file is None:
        raise FileNotFoundError(
            "EnergyPlus Python API was not found below " f"{energyplus_dir}"
        )
    sys.path.insert(0, str(api_file.parent.parent))
    from pyenergyplus.api import EnergyPlusAPI

    return EnergyPlusAPI()


class HighRiseEnergyPlusEnv(gym.Env):
    """Three-zone, nine-action EnergyPlus environment.

    The blind component is binary at actuation: a normalized command mapping
    to a fraction below 0.5 retracts the device; a value at or above 0.5
    deploys it. The continuous request is retained in ``info``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        idf_path: str | Path,
        weather_path: str | Path,
        energyplus_dir: str | Path | None = None,
        output_root: str | Path = "rl_outputs_blinds",
        min_setpoint: float = 22.0,
        max_setpoint: float = 29.0,
        comfort_low: float = 22.0,
        comfort_high: float = 26.0,
        energy_weight: float = 1.0,
        comfort_weight: float = 100.0,
        cop: float = 3.5,
        safe_control: bool = False,
        occupied_max_setpoint: float = 25.5,
        emergency_temperature: float = 26.0,
        emergency_setpoint: float = 22.0,
        intervention_weight: float = 0.2,
        iaq_limit_ppm: float = 1000.0,
        iaq_weight: float = 1000.0,
        movement_weight: float = 0.02,
        occupied_min_ventilation: float = 0.35,
        action_penalty_dimension: int = 9,
    ):
        super().__init__()
        self.idf_path = Path(idf_path).expanduser().resolve()
        self.weather_path = Path(weather_path).expanduser().resolve()
        self.energyplus_dir = Path(
            energyplus_dir
            or os.environ.get("ENERGYPLUS_DIR", "/usr/local/EnergyPlus-22-1-0")
        ).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not self.idf_path.exists():
            raise FileNotFoundError(self.idf_path)
        if not self.weather_path.exists():
            raise FileNotFoundError(self.weather_path)

        self.min_setpoint = float(min_setpoint)
        self.max_setpoint = float(max_setpoint)
        self.comfort_low = float(comfort_low)
        self.comfort_high = float(comfort_high)
        self.energy_weight = float(energy_weight)
        self.comfort_weight = float(comfort_weight)
        self.cop = float(cop)
        self.safe_control = bool(safe_control)
        self.occupied_max_setpoint = float(occupied_max_setpoint)
        self.emergency_temperature = float(emergency_temperature)
        self.emergency_setpoint = float(emergency_setpoint)
        self.intervention_weight = float(intervention_weight)
        self.iaq_limit_ppm = float(iaq_limit_ppm)
        self.iaq_weight = float(iaq_weight)
        self.movement_weight = float(movement_weight)
        self.occupied_min_ventilation = float(occupied_min_ventilation)
        self.action_penalty_dimension = int(action_penalty_dimension)
        if self.action_penalty_dimension <= 0:
            raise ValueError("action_penalty_dimension must be positive")

        self.action_space = spaces.Box(-1.0, 1.0, shape=(9,), dtype=np.float32)
        # T(3), RH(3), occupancy(3), CO2(3), outdoor T/RH/solar(3), time(2),
        # and nine applied actions = 26 observations.
        low = np.array(
            [-10] * 3
            + [0] * 3
            + [0] * 3
            + [0] * 3
            + [-20, 0, 0, -1, -1]
            + [self.min_setpoint] * 3
            + [0] * 3
            + [0] * 3,
            dtype=np.float32,
        )
        high = np.array(
            [60] * 3
            + [100] * 3
            + [100] * 3
            + [5000] * 3
            + [60, 100, 1500, 1, 1]
            + [self.max_setpoint] * 3
            + [1] * 3
            + [1] * 3,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.api = load_energyplus_api(self.energyplus_dir)
        self.state = None
        self.thread = None
        self.action_queue: queue.Queue = queue.Queue(maxsize=1)
        self.transition_queue: queue.Queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.handles_initialized = False
        self.variable_handles: dict[str, int | list[int]] = {}
        self.actuator_handles: dict[str, list[int]] = {}
        self.current_setpoints = np.full(3, 24.0, dtype=np.float32)
        self.current_ventilation = np.zeros(3, dtype=np.float32)
        self.current_blind_closure = np.zeros(3, dtype=np.float32)
        self.requested_setpoints = self.current_setpoints.copy()
        self.requested_ventilation = self.current_ventilation.copy()
        self.requested_blind_closure = self.current_blind_closure.copy()
        self.current_safety_interventions = 0
        self.previous_applied_action = np.concatenate(
            [
                self.current_setpoints,
                self.current_ventilation,
                self.current_blind_closure,
            ]
        )
        self.last_observation = np.zeros(26, dtype=np.float32)
        self.last_info: dict = {}
        self.terminal_emitted = False
        self.episode_number = 0
        self.output_directory: Path | None = None

    def _setpoint_from_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return self.min_setpoint + (action + 1.0) * 0.5 * (
            self.max_setpoint - self.min_setpoint
        )

    @staticmethod
    def _fraction_from_action(action: np.ndarray) -> np.ndarray:
        return np.clip(
            (np.asarray(action, dtype=np.float32) + 1.0) * 0.5, 0.0, 1.0
        )

    def _request_variables(self):
        ex = self.api.exchange
        for zone in ZONES:
            ex.request_variable(self.state, "Zone Mean Air Temperature", zone)
            ex.request_variable(self.state, "Zone Air Relative Humidity", zone)
            ex.request_variable(self.state, "Zone People Occupant Count", zone)
            ex.request_variable(self.state, "Zone Air CO2 Concentration", zone)
        for system in IDEAL_SYSTEMS:
            ex.request_variable(
                self.state, "Zone Ideal Loads Zone Total Cooling Energy", system
            )
        for window in CONTROLLED_WINDOWS:
            ex.request_variable(
                self.state,
                "Surface Shading Device Is On Time Fraction",
                window,
            )
        ex.request_variable(
            self.state, "Site Outdoor Air Drybulb Temperature", "Environment"
        )
        ex.request_variable(
            self.state, "Site Outdoor Air Relative Humidity", "Environment"
        )
        ex.request_variable(
            self.state,
            "Site Direct Solar Radiation Rate per Area",
            "Environment",
        )

    def _initialize_handles(self):
        ex = self.api.exchange
        if not ex.api_data_fully_ready(self.state):
            return False
        self.variable_handles = {
            "temperature": [
                ex.get_variable_handle(
                    self.state, "Zone Mean Air Temperature", zone
                )
                for zone in ZONES
            ],
            "humidity": [
                ex.get_variable_handle(
                    self.state, "Zone Air Relative Humidity", zone
                )
                for zone in ZONES
            ],
            "occupancy": [
                ex.get_variable_handle(
                    self.state, "Zone People Occupant Count", zone
                )
                for zone in ZONES
            ],
            "co2": [
                ex.get_variable_handle(
                    self.state, "Zone Air CO2 Concentration", zone
                )
                for zone in ZONES
            ],
            "cooling": [
                ex.get_variable_handle(
                    self.state,
                    "Zone Ideal Loads Zone Total Cooling Energy",
                    system,
                )
                for system in IDEAL_SYSTEMS
            ],
            "blind_status": [
                ex.get_variable_handle(
                    self.state,
                    "Surface Shading Device Is On Time Fraction",
                    window,
                )
                for window in CONTROLLED_WINDOWS
            ],
            "outdoor_temperature": ex.get_variable_handle(
                self.state,
                "Site Outdoor Air Drybulb Temperature",
                "Environment",
            ),
            "outdoor_humidity": ex.get_variable_handle(
                self.state,
                "Site Outdoor Air Relative Humidity",
                "Environment",
            ),
            "solar": ex.get_variable_handle(
                self.state,
                "Site Direct Solar Radiation Rate per Area",
                "Environment",
            ),
        }
        self.actuator_handles = {
            "setpoint": [
                ex.get_actuator_handle(
                    self.state,
                    "Zone Temperature Control",
                    "Cooling Setpoint",
                    zone,
                )
                for zone in ZONES
            ],
            "ventilation": [
                ex.get_actuator_handle(
                    self.state,
                    "Schedule:Constant",
                    "Schedule Value",
                    schedule,
                )
                for schedule in VENT_SCHEDULES
            ],
            "blind_deploy": [
                ex.get_actuator_handle(
                    self.state,
                    "Schedule:Constant",
                    "Schedule Value",
                    schedule,
                )
                for schedule in BLIND_DEPLOY_SCHEDULES
            ],
        }
        handles: list[int] = []
        for value in self.variable_handles.values():
            handles.extend(value if isinstance(value, list) else [value])
        for value in self.actuator_handles.values():
            handles.extend(value)
        if any(handle < 0 for handle in handles):
            missing = [handle for handle in handles if handle < 0]
            raise RuntimeError(
                f"{len(missing)} EnergyPlus API handle(s) were not found. "
                "Confirm the three blind schedules, WindowShadingControl objects, "
                "controlled surface names, and requested output variable."
            )
        self.handles_initialized = True
        return True

    def _begin_timestep(self, state):
        ex = self.api.exchange
        if self.stop_event.is_set() or ex.warmup_flag(state):
            return
        if not self.handles_initialized and not self._initialize_handles():
            return
        try:
            action = self.action_queue.get(timeout=120)
        except queue.Empty:
            self.stop_event.set()
            return
        if action is None:
            self.stop_event.set()
            return
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (9,):
            raise ValueError(f"Expected 9 actions, received shape {action.shape}")

        self.requested_setpoints = self._setpoint_from_action(action[:3])
        self.requested_ventilation = self._fraction_from_action(action[3:6])
        self.requested_blind_closure = self._fraction_from_action(action[6:9])
        self.current_setpoints = self.requested_setpoints.copy()
        self.current_ventilation = self.requested_ventilation.copy()
        self.current_blind_closure = self.requested_blind_closure.copy()
        self.current_safety_interventions = 0

        if self.safe_control:
            occupancies = np.array(
                [
                    ex.get_variable_value(state, handle)
                    for handle in self.variable_handles["occupancy"]
                ]
            )
            temperatures = np.array(
                [
                    ex.get_variable_value(state, handle)
                    for handle in self.variable_handles["temperature"]
                ]
            )
            co2_ppm = np.array(
                [
                    ex.get_variable_value(state, handle)
                    for handle in self.variable_handles["co2"]
                ]
            )
            occupied = occupancies > 0.0
            before = np.concatenate(
                [
                    self.current_setpoints,
                    self.current_ventilation,
                    self.current_blind_closure,
                ]
            )
            self.current_setpoints[occupied] = np.minimum(
                self.current_setpoints[occupied], self.occupied_max_setpoint
            )
            emergency = occupied & (
                temperatures >= self.emergency_temperature
            )
            self.current_setpoints[emergency] = self.emergency_setpoint
            high_co2 = occupied & (co2_ppm >= self.iaq_limit_ppm)
            self.current_ventilation[high_co2] = np.maximum(
                self.current_ventilation[high_co2],
                self.occupied_min_ventilation,
            )
            hot_occupied = occupied & (temperatures >= self.comfort_high)
            self.current_blind_closure[hot_occupied] = np.maximum(
                self.current_blind_closure[hot_occupied], 0.70
            )
            after = np.concatenate(
                [
                    self.current_setpoints,
                    self.current_ventilation,
                    self.current_blind_closure,
                ]
            )
            self.current_safety_interventions = int(
                np.count_nonzero(np.abs(before - after) > 1e-6)
            )

        # Binary device command: 0 = retracted, 1 = deployed.
        self.current_blind_closure = (
            self.current_blind_closure >= 0.5
        ).astype(np.float32)
        for handle, setpoint in zip(
            self.actuator_handles["setpoint"], self.current_setpoints
        ):
            ex.set_actuator_value(state, handle, float(setpoint))
        for handle, fraction in zip(
            self.actuator_handles["ventilation"], self.current_ventilation
        ):
            ex.set_actuator_value(state, handle, float(fraction))
        for handle, deployed in zip(
            self.actuator_handles["blind_deploy"],
            self.current_blind_closure,
        ):
            ex.set_actuator_value(state, handle, float(deployed))

    def _end_timestep(self, state):
        ex = self.api.exchange
        if (
            self.stop_event.is_set()
            or ex.warmup_flag(state)
            or not self.handles_initialized
        ):
            return
        temperatures = np.array(
            [
                ex.get_variable_value(state, handle)
                for handle in self.variable_handles["temperature"]
            ]
        )
        humidities = np.array(
            [
                ex.get_variable_value(state, handle)
                for handle in self.variable_handles["humidity"]
            ]
        )
        occupancies = np.array(
            [
                ex.get_variable_value(state, handle)
                for handle in self.variable_handles["occupancy"]
            ]
        )
        co2_ppm = np.array(
            [
                ex.get_variable_value(state, handle)
                for handle in self.variable_handles["co2"]
            ]
        )
        cooling_j = np.array(
            [
                ex.get_variable_value(state, handle)
                for handle in self.variable_handles["cooling"]
            ]
        )
        blind_status = np.array(
            [
                ex.get_variable_value(state, handle)
                for handle in self.variable_handles["blind_status"]
            ]
        )
        outdoor_temperature = ex.get_variable_value(
            state, self.variable_handles["outdoor_temperature"]
        )
        outdoor_humidity = ex.get_variable_value(
            state, self.variable_handles["outdoor_humidity"]
        )
        solar_w_m2 = ex.get_variable_value(
            state, self.variable_handles["solar"]
        )
        hour_decimal = float(ex.hour(state)) + float(ex.minutes(state)) / 60.0
        angle = 2.0 * np.pi * hour_decimal / 24.0
        observation = np.concatenate(
            [
                temperatures,
                humidities,
                occupancies,
                co2_ppm,
                [
                    outdoor_temperature,
                    outdoor_humidity,
                    solar_w_m2,
                    np.sin(angle),
                    np.cos(angle),
                ],
                self.current_setpoints,
                self.current_ventilation,
                self.current_blind_closure,
            ]
        ).astype(np.float32)

        occupied = occupancies > 0.0
        discomfort = np.where(
            occupied,
            np.maximum(self.comfort_low - temperatures, 0.0)
            + np.maximum(temperatures - self.comfort_high, 0.0),
            0.0,
        )
        cooling_thermal_kwh = float(
            np.maximum(cooling_j, 0.0).sum() / 3_600_000.0
        )
        cooling_electric_kwh = cooling_thermal_kwh / self.cop
        discomfort_degree_hours = float(discomfort.sum() / 6.0)
        co2_excess = np.where(
            occupied, np.maximum(co2_ppm - self.iaq_limit_ppm, 0.0), 0.0
        )
        co2_exceedance_kppm_hours = float(
            co2_excess.sum() / (6.0 * 1000.0)
        )
        applied_action = np.concatenate(
            [
                self.current_setpoints,
                self.current_ventilation,
                self.current_blind_closure,
            ]
        )
        action_scale = np.array(
            [7.0] * 3 + [1.0] * 6, dtype=np.float32
        )
        action_movement = float(
            np.abs(
                (applied_action - self.previous_applied_action) / action_scale
            ).sum()
            / self.action_penalty_dimension
        )
        self.previous_applied_action = applied_action.copy()
        intervention_penalty = (
            self.intervention_weight
            * self.current_safety_interventions
            / self.action_penalty_dimension
        )
        reward = -(
            self.energy_weight * cooling_electric_kwh
            + self.comfort_weight * discomfort_degree_hours
            + self.iaq_weight * co2_exceedance_kppm_hours
            + intervention_penalty
            + self.movement_weight * action_movement
        )
        try:
            final_substep = (
                ex.zone_time_step_number(state)
                >= ex.num_time_steps_in_hour(state)
            )
        except Exception:
            final_substep = ex.zone_time_step_number(state) >= 6
        terminated = (
            ex.month(state) == 12
            and ex.day_of_month(state) == 31
            and ex.hour(state) >= 24
            and final_substep
        )
        info = {
            "month": int(ex.month(state)),
            "day": int(ex.day_of_month(state)),
            "hour": int(ex.hour(state)),
            "minute": int(ex.minutes(state)),
            "temperatures_c": temperatures.tolist(),
            "occupancies": occupancies.tolist(),
            "co2_ppm": co2_ppm.tolist(),
            "setpoints_c": self.current_setpoints.tolist(),
            "requested_setpoints_c": self.requested_setpoints.tolist(),
            "ventilation_fraction": self.current_ventilation.tolist(),
            "requested_ventilation_fraction": (
                self.requested_ventilation.tolist()
            ),
            "blind_closure_fraction": self.current_blind_closure.tolist(),
            "requested_blind_closure_fraction": (
                self.requested_blind_closure.tolist()
            ),
            "blind_status_fraction": blind_status.tolist(),
            "safety_interventions": self.current_safety_interventions,
            "intervention_penalty": intervention_penalty,
            "action_movement": action_movement,
            "action_penalty_dimension": self.action_penalty_dimension,
            "iaq_weight": self.iaq_weight,
            "co2_exceedance_kppm_hours": co2_exceedance_kppm_hours,
            "solar_w_m2": float(solar_w_m2),
            "cooling_thermal_kwh": cooling_thermal_kwh,
            "cooling_electric_kwh": cooling_electric_kwh,
            "discomfort_degree_hours": discomfort_degree_hours,
        }
        self.last_observation = observation
        self.last_info = info
        self.terminal_emitted = terminated
        self.transition_queue.put(
            (observation, float(reward), terminated, False, info)
        )

    def _run_energyplus(self):
        self._request_variables()
        runtime = self.api.runtime
        runtime.set_console_output_status(self.state, False)
        runtime.callback_begin_zone_timestep_before_set_current_weather(
            self.state, self._begin_timestep
        )
        runtime.callback_end_zone_timestep_after_zone_reporting(
            self.state, self._end_timestep
        )
        args = [
            "-w",
            str(self.weather_path),
            "-d",
            str(self.output_directory),
            str(self.idf_path),
        ]
        exit_code = runtime.run_energyplus(self.state, args)
        if exit_code != 0 and not self.stop_event.is_set():
            self.transition_queue.put(
                RuntimeError(
                    f"EnergyPlus exited with code {exit_code}. See "
                    f"{self.output_directory}/eplusout.err"
                )
            )
        elif (
            exit_code == 0
            and not self.stop_event.is_set()
            and not self.terminal_emitted
        ):
            info = dict(self.last_info)
            info["simulation_complete"] = True
            self.transition_queue.put(
                (self.last_observation.copy(), 0.0, True, False, info)
            )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.action_queue = queue.Queue(maxsize=1)
        self.transition_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.handles_initialized = False
        self.variable_handles = {}
        self.actuator_handles = {}
        self.current_setpoints = np.full(3, 24.0, dtype=np.float32)
        self.current_ventilation = np.zeros(3, dtype=np.float32)
        self.current_blind_closure = np.zeros(3, dtype=np.float32)
        self.requested_setpoints = self.current_setpoints.copy()
        self.requested_ventilation = self.current_ventilation.copy()
        self.requested_blind_closure = self.current_blind_closure.copy()
        self.previous_applied_action = np.concatenate(
            [
                self.current_setpoints,
                self.current_ventilation,
                self.current_blind_closure,
            ]
        )
        self.current_safety_interventions = 0
        self.last_observation = np.zeros(26, dtype=np.float32)
        self.last_info = {}
        self.terminal_emitted = False
        self.state = self.api.state_manager.new_state()
        self.episode_number += 1
        self.output_directory = (
            self.output_root / f"episode_{self.episode_number:04d}"
        )
        if self.output_directory.exists():
            shutil.rmtree(self.output_directory)
        self.output_directory.mkdir(parents=True)
        neutral_action = np.concatenate(
            [
                np.full(
                    3,
                    (24.0 - 25.5) / 3.5,
                    dtype=np.float32,
                ),
                np.full(3, -1.0, dtype=np.float32),
                np.full(3, -1.0, dtype=np.float32),
            ]
        )
        self.action_queue.put(neutral_action)
        self.thread = threading.Thread(
            target=self._run_energyplus, daemon=True
        )
        self.thread.start()
        packet = self.transition_queue.get(timeout=180)
        if isinstance(packet, Exception):
            raise packet
        observation, _, terminated, _, info = packet
        if terminated:
            raise RuntimeError("EnergyPlus terminated before the first RL step.")
        return observation, info

    def step(self, action):
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(
                "The EnergyPlus simulation is not running. Call reset()."
            )
        self.action_queue.put(np.asarray(action, dtype=np.float32), timeout=120)
        packet = None
        for _ in range(180):
            try:
                packet = self.transition_queue.get(timeout=1)
                break
            except queue.Empty:
                if self.thread is not None and not self.thread.is_alive():
                    info = dict(self.last_info)
                    info["simulation_complete"] = True
                    return (
                        self.last_observation.copy(),
                        0.0,
                        True,
                        False,
                        info,
                    )
        if packet is None:
            raise TimeoutError(
                "No EnergyPlus transition was received within 180 seconds. "
                f"See {self.output_directory}/eplusout.err"
            )
        if isinstance(packet, Exception):
            raise packet
        return packet

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            try:
                self.action_queue.put_nowait(None)
            except queue.Full:
                pass
            self.thread.join()
        if self.state is not None:
            try:
                self.api.runtime.clear_callbacks()
                self.api.state_manager.delete_state(self.state)
            except Exception:
                pass
        self.thread = None
        self.state = None
