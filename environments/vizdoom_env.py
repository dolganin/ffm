"""Minimal VizDoom environment wrapper.

This wrapper supports two modes of operation:

* If an ``env_id`` is provided it falls back to ``gym.make`` so that
  pre‑registered gym environments (e.g. from ``vizdoomgym``) can still be
  used.
* Otherwise the environment is created directly via the ``vizdoom`` Python
  API using the provided ``config`` and ``scenario`` files.  Observations are
  returned as CHW ``float32`` arrays in the ``[0, 1]`` range and actions are
  discrete one‑hot vectors over the available buttons.
"""

from __future__ import annotations

from typing import Optional

import gym
import numpy as np

try:  # pragma: no cover - exercised only when vizdoom is available
    from vizdoom import DoomGame, ScreenFormat, ScreenResolution
except Exception:  # pragma: no cover - when vizdoom is missing
    DoomGame = None  # type: ignore


class VizDoomEnv:
    def __init__(
        self,
        env_id: Optional[str] = None,
        *,
        config: Optional[str] = None,
        scenario: Optional[str] = None,
        frame_skip: int = 4,
        resolution: str = "RES_160X120",
        **kwargs,
    ):
        if DoomGame is None:
            raise ImportError("vizdoom is not installed")

        # ------------------------------------------------------------------ gym
        if env_id is not None:
            self.env = gym.make(env_id, **kwargs)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self.max_episode_steps = getattr(self.env, "max_episode_steps", None)
            self._gym_env = True
            return

        # -------------------------------------------------------------- vizdoom API
        game = DoomGame()
        if config:
            game.load_config(config)
        if scenario:
            game.set_doom_scenario_path(scenario)
        if resolution:
            res = getattr(ScreenResolution, resolution)
            game.set_screen_resolution(res)
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_window_visible(False)
        game.init()

        self.game = game
        self.frame_skip = frame_skip
        self._gym_env = False

        n_buttons = game.get_available_buttons_size()
        self.actions = np.eye(n_buttons, dtype=np.uint8).tolist()
        self.action_space = gym.spaces.Discrete(len(self.actions))

        height, width = game.get_screen_height(), game.get_screen_width()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, height, width), dtype=np.float32
        )
        self.max_episode_steps = None

    # ------------------------------------------------------------------- helpers
    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3 and obs.shape[0] in (1, 3, 4):  # CHW
            obs = obs[:3]
        elif obs.ndim == 3:  # HWC -> CHW
            obs = np.transpose(obs[:, :, :3], (2, 0, 1))
        return obs.astype(np.float32) / 255.0

    def _get_obs(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return self._process_obs(state.screen_buffer)

    # ---------------------------------------------------------------------- env API
    def reset(self):
        if self._gym_env:
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            return self._process_obs(obs)

        self.game.new_episode()
        return self._get_obs()

    def step(self, action: int):
        if self._gym_env:
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            return self._process_obs(obs), reward, done, info

        reward = self.game.make_action(self.actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        obs = self._get_obs() if not done else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, reward, done, {}

    def close(self):
        if self._gym_env:
            self.env.close()
        else:
            self.game.close()

    def render(self):  # pragma: no cover - visualization helper
        if self._gym_env:
            return self.env.render()
        state = self.game.get_state()
        if state is None:
            h, w = self.game.get_screen_height(), self.game.get_screen_width()
            return np.zeros((h, w, 3), dtype=np.uint8)
        buf = state.screen_buffer
        if buf.ndim == 3 and buf.shape[0] in (1, 3, 4):
            buf = np.transpose(buf[:3], (1, 2, 0))
        return buf

