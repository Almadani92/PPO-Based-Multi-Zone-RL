from pathlib import Path
from stable_baselines3 import PPO

root = Path(__file__).resolve().parents[1]

expected = {
    "categorical": "MultiDiscrete",
    "six_action": "Box",
    "setpoint_only": "Box",
}

count = 0

for family, expected_space in expected.items():
    for seed in (42, 123, 2026):
        directory = root / "models" / family / f"seed{seed}"
        model_path = directory / "model.zip"
        vecnormalize_path = directory / "vecnormalize.pkl"

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        if not vecnormalize_path.exists():
            raise FileNotFoundError(vecnormalize_path)

        model = PPO.load(model_path, device="cpu")
        action_space = type(model.action_space).__name__

        if action_space != expected_space:
            raise RuntimeError(
                f"{family} seed {seed}: expected {expected_space}, "
                f"found {action_space}"
            )

        count += 1
        print(
            family,
            seed,
            f"timesteps={model.num_timesteps}",
            f"action_space={model.action_space}",
        )

print(f"Validation passed: {count}/9 models loaded successfully.")
