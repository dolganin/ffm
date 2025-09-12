import gym
try:
    import minigrid  # noqa: F401
except Exception as e:  # pragma: no cover - only executed when module missing
    minigrid = None


class MiniGridEnv:
    """Simple wrapper around gym-minigrid environments.

    Parameters
    ----------
    env_id: str
        Name of the MiniGrid environment to create via ``gym.make``.
    kwargs: dict
        Additional keyword arguments forwarded to ``gym.make``.
    """

    def __init__(self, env_id: str, **kwargs):
        if minigrid is None:
            raise ImportError("gym-minigrid is not installed")
        self.env = gym.make(env_id, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_episode_steps = getattr(self.env, "max_episode_steps", None)

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):  # gym>=0.26 returns (obs, info)
            obs = obs[0]
        return obs

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:  # gymnasium style
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        return obs, reward, done, info

    def close(self):
        self.env.close()
