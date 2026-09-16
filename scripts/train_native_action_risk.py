#!/usr/bin/env python3
"""Train the risk-only expert with whole-route holdout, never an actor."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.local_policy_training import LocalPolicyTrainingConfig
from dronedream_agent_core.training.action_risk_artifacts import load_action_risk_dataset
from dronedream_agent_core.training.action_risk_training import train_action_risk_expert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dataset", type=Path, action="append", required=True)
    parser.add_argument("--validation-dataset", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = LocalPolicyTrainingConfig.model_validate_json(args.config.read_bytes())
    training = [load_action_risk_dataset(path) for path in args.training_dataset]
    validation = [load_action_risk_dataset(path) for path in args.validation_dataset]
    receipt = train_action_risk_expert(training, validation, config, args.output)
    print(json.dumps(receipt, allow_nan=False), flush=True)
    return 0 if receipt["offline_validation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
