"""Command line entry point for running training from a YAML config."""

import argparse

from yaml_parser import load_config
from environments import create_env


def main():
    parser = argparse.ArgumentParser(description="PPO training with FFM")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg.get("environment", {})
    env = create_env(env_cfg)

    # Placeholder: in a full implementation this is where PPO training would run.
    obs = env.reset()
    print("Environment initialized with observation shape", getattr(getattr(obs, 'shape', None), '__str__', lambda: obs)())

    env.close()


if __name__ == "__main__":
    main()
