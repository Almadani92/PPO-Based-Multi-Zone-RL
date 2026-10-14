# PPO-Based Multi-Zone RL

Testing code and pretrained PPO policies for coordinated control of
cooling setpoints, outdoor-air ventilation, and binary window shades
in a high-rise residential EnergyPlus model.

## Included controllers

- Setpoint-only PPO: three continuous actions
- Six-action PPO: six continuous actions
- Categorical PPO: nine factored discrete actions
- Categorical PPO with a minimum shade dwell time
- Conventional and DOE/ASHRAE prototype baselines

Models are provided for seeds 42, 123, and 2026.

## Requirements

- Python 3.10
- EnergyPlus 22.1
- Stable-Baselines3
- Gymnasium
- PyTorch
- NumPy
- pandas

Install the dependencies by running:

    python -m pip install -r requirements.txt

EnergyPlus is expected at:

    /usr/local/EnergyPlus-22-1-0

## Weather data

Weather files are not included. Place a compatible EPW file inside the
weather directory.

The models were trained using Dubai International weather. They can be
evaluated without retraining under other compatible weather files.

## Validate the models

Run:

    python scripts/validate_models.py

## Test categorical PPO

Run:

    bash scripts/test_categorical.sh weather/your_weather.epw 42

The seed must be 42, 123, or 2026.

## Test the 120-minute dwell constraint

Run:

    bash scripts/test_dwell120.sh weather/your_weather.epw 42

At the 10-minute simulation resolution, 12 steps correspond to a
120-minute minimum shade dwell time.

## Outputs and storage

Evaluation results are written to the outputs directory. Temporary
EnergyPlus files can consume hundreds of megabytes and are excluded
from Git.

## Building model

The supplied IDF retains the DOE/ASHRAE high-rise apartment prototype.
Three vertically distributed apartments are controlled.

## Citation

Citation information is provided in CITATION.cff.

## Access

This repository is currently private and intended for controlled
research testing.
