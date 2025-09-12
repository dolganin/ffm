import gym
try:
    import vizdoom  # noqa: F401
except Exception:
    vizdoom = None


class VizDoomEnv:
    """Wrapper for VizDoom environments created via ``gym.make``."""

    def __init__(self, env_id: str, **kwargs):
        if vizdoom is None:
            raise ImportError("vizdoom is not installed")
        self.env = gym.make(env_id, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_episode_steps = getattr(self.env, "max_episode_steps", None)

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        # convert to CHW float32 if needed
        if hasattr(obs, "transpose"):
            obs = obs.transpose(2, 0, 1).astype("float32") / 255.0
        return obs

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        if hasattr(obs, "transpose"):
            obs = obs.transpose(2, 0, 1).astype("float32") / 255.0
        return obs, reward, done, info

    def close(self):
        self.env.close()

    def render(self):
        """Render the underlying environment and return an RGB array."""
        return self.env.render()
