from .minigrid_env import MiniGridEnv
from .vizdoom_env import VizDoomEnv


def create_env(config: dict):
    """Instantiate an environment wrapper from a configuration dictionary.

    Parameters
    ----------
    config: dict
        Must contain a ``type`` field specifying ``"minigrid"`` or ``"vizdoom"``.
        The remaining keys are forwarded to the respective environment wrapper.
    """
    cfg = dict(config)
    env_type = cfg.pop("type", None)
    env_type = env_type.lower() if isinstance(env_type, str) else env_type

    env_id = cfg.pop("name", cfg.pop("env_id", None))
    if env_id is not None:
        cfg["env_id"] = env_id

    cfg.setdefault("render_mode", "rgb_array")

    if env_type == "minigrid":
        return MiniGridEnv(**cfg)
    if env_type == "vizdoom":
        return VizDoomEnv(**cfg)
    raise ValueError(f"Unknown environment type: {env_type}")
