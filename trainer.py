"""Simplified PPO trainer with tensorboard logging and GIF evaluation."""

from __future__ import annotations

import os
from collections import deque
from types import SimpleNamespace
from typing import List, Sequence

import imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from environments import create_env


def polynomial_decay(initial: float, final: float, max_decay_steps: int, power: float, step: int) -> float:
    """Polynomial decay schedule used for various hyper parameters."""
    if max_decay_steps <= 0:
        return final
    step = min(step, max_decay_steps)
    frac = (1 - step / max_decay_steps) ** power
    return final + (initial - final) * frac


class DummyModel(torch.nn.Module):
    """Minimal model used for placeholder training.

    The implementation is intentionally lightweight – the goal of this
    repository is to showcase the training interface rather than provide a
    performant model implementation.  The forward pass returns a categorical
    distribution over two actions, a dummy value estimate and the recurrent
    state as-is.
    """

    def __init__(self):
        super().__init__()

    def init_recurrent_cell_states(self, batch: int, device: torch.device):
        h = torch.zeros(1, batch, 1, device=device)
        c = torch.zeros(1, batch, 1, device=device)
        return h, c

    def forward(self, obs: torch.Tensor, rc: Sequence[torch.Tensor], device: torch.device):
        pol = [torch.distributions.Categorical(logits=torch.zeros(obs.shape[0], 2, device=device))]
        value = torch.zeros(obs.shape[0], device=device)
        return pol, value, rc


class PPOTrainer:
    """Orchestrates the training loop and logging."""

    def __init__(self, config: dict, run_id: str = "run", device: torch.device | None = None):
        self.config = config
        self.run_id = run_id
        self.device = device or torch.device("cpu")

        self.writer = SummaryWriter(os.path.join("runs", run_id))
        self.video_dir = os.path.join("videos", run_id)

        # Schedules
        self.lr_schedule = config.get(
            "learning_rate_schedule",
            {"initial": 3e-4, "final": 3e-4, "max_decay_steps": 1, "power": 1.0},
        )
        self.beta_schedule = config.get(
            "beta_schedule",
            {"initial": 1.0, "final": 1.0, "max_decay_steps": 1, "power": 1.0},
        )
        self.cr_schedule = config.get(
            "clip_range_schedule",
            {"initial": 0.2, "final": 0.2, "max_decay_steps": 1, "power": 1.0},
        )

        # Dummy model/buffer placeholders.  In a full implementation these
        # would be replaced by actual neural networks and rollout buffers.
        self.model = DummyModel().to(self.device)
        self.buffer = SimpleNamespace(
            values=torch.zeros(1, device=self.device),
            advantages=torch.zeros(1, device=self.device),
            prepare_batch_dict=lambda: None,
            samples_flat=[],
        )

        self.recurrence = config.get("recurrence", {"layer_type": "gru"})

    # ------------------------------------------------------------------ utils
    def _sample_training_data(self) -> List[dict]:
        """Collect training data.

        This placeholder implementation merely fabricates random data to
        exercise the training loop.
        """

        self.buffer.values = torch.randn(4, device=self.device)
        self.buffer.advantages = torch.randn(4, device=self.device)
        return [
            {
                "reward": float(np.random.randn()),
                "length": int(np.random.randint(1, 10)),
                "success": bool(np.random.rand() > 0.5),
            }
        ]

    def _train_epochs(self, lr: float, clip_range: float, beta: float) -> np.ndarray:
        """Run training epochs and return statistics.

        Returns an array shaped (1, 4) with [pi_loss, v_loss, loss, entropy].
        """

        return np.zeros((1, 4), dtype=np.float32)

    def _process_episode_info(self, infos: Sequence[dict]) -> dict:
        """Aggregate episode information into statistics."""

        if not infos:
            return {}
        rewards = [i.get("reward", 0.0) for i in infos]
        lengths = [i.get("length", 0.0) for i in infos]
        result = {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "length_mean": float(np.mean(lengths)),
            "length_std": float(np.std(lengths)),
        }
        successes = [i.get("success") for i in infos if "success" in i]
        if successes:
            result["success_percent"] = float(np.mean(successes) * 100.0)
        return result

    # ----------------------------------------------------------------- logging
    def _write_training_summary(self, step: int, stats: np.ndarray, episode_result: dict) -> None:
        self.writer.add_scalar("loss/pi", stats[0], step)
        self.writer.add_scalar("loss/v", stats[1], step)
        self.writer.add_scalar("loss/total", stats[2], step)
        self.writer.add_scalar("loss/entropy", stats[3], step)

        if "reward_mean" in episode_result:
            self.writer.add_scalar("charts/reward_mean", episode_result["reward_mean"], step)
        if "length_mean" in episode_result:
            self.writer.add_scalar("charts/length_mean", episode_result["length_mean"], step)
        if "success_percent" in episode_result:
            self.writer.add_scalar("charts/success_rate", episode_result["success_percent"], step)

    # --------------------------------------------------------------- evaluation
    def _to_hwc_uint8(self, frame) -> np.ndarray:
        """Normalize a frame to HWC uint8 RGB format."""
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        arr = np.asarray(frame)

        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            a_min, a_max = float(arr.min()), float(arr.max())
            if 0.0 <= a_min and a_max <= 1.0:
                arr = (arr * 255.0).round()
            else:
                rng = max(1e-6, a_max - a_min)
                arr = ((arr - a_min) / rng * 255.0).round()
            arr = arr.clip(0, 255).astype(np.uint8)

        return arr

    def _load_initial_rc(self, batch: int, device: torch.device):
        """Load initial recurrent states from config paths if provided."""
        layer_type = self.recurrence.get("layer_type", "gru")
        hxs, cxs = self.model.init_recurrent_cell_states(batch, device)
        init_conf = self.recurrence.get("init_state", {})

        hx_path = init_conf.get("hx_path")
        if hx_path:
            hxs = torch.load(hx_path, map_location=device)

        if layer_type != "gru":
            cx_path = init_conf.get("cx_path")
            if cx_path:
                cxs = torch.load(cx_path, map_location=device)
            return hxs, cxs

        return hxs

    def _record_eval_episode(self, step: int) -> None:
        print(f"[eval] Recording evaluation gif for step {step}...")

        old_n_workers = self.config.get("n_workers", 1)
        self.config["n_workers"] = 1
        env = create_env(self.config["environment"])
        self.config["n_workers"] = old_n_workers

        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) and len(reset_out) >= 1 else reset_out

        frames: List[np.ndarray] = []
        rc = self._load_initial_rc(1, self.device)

        raw = env.render()
        first = self._to_hwc_uint8(raw)
        if self.config.get("downscale_eval_frames", True):
            target_h, target_w = first.shape[0] // 2, first.shape[1] // 2
        else:
            target_h, target_w = first.shape[0], first.shape[1]
        frame = first[:: first.shape[0] // target_h or 1, :: first.shape[1] // target_w or 1]
        frames.append(frame)

        done, steps = False, 0
        max_steps = int(self.config.get("eval_max_steps", 100))

        while not done and steps < max_steps:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(np.asarray(obs)).unsqueeze(0).to(self.device)
                pol, _, rc = self.model(obs_tensor, rc, self.device)
                act = np.array([p.probs.argmax(-1).item() for p in pol])

            step_out = env.step(act)
            if isinstance(step_out, (tuple, list)) and len(step_out) >= 4:
                obs = step_out[0]
                terminated = bool(step_out[2])
                truncated = bool(step_out[3])
                done = terminated or truncated
            else:
                obs, _, done, _ = step_out

            raw = env.render()
            f = self._to_hwc_uint8(raw)
            if f.shape[0] != target_h or f.shape[1] != target_w:
                f = f[:: max(1, f.shape[0] // target_h), :: max(1, f.shape[1] // target_w)]
                f = f[:target_h, :target_w]

            frames.append(f)
            steps += 1

        env.close()
        os.makedirs(self.video_dir, exist_ok=True)
        path = os.path.join(self.video_dir, f"step_{step:05d}.gif")
        imageio.mimsave(path, frames, duration=1 / self.config.get("fps", 15))
        print(f"[eval] Saved gif to {path}")

    # -------------------------------------------------------------- save/close
    def _save_model(self) -> None:
        os.makedirs("models", exist_ok=True)
        path = os.path.join("models", f"{self.run_id}.pt")
        torch.save(self.model.state_dict(), path)
        print(f"[save] Model saved to {path}")

    def close(self) -> None:
        self.writer.close()

    # --------------------------------------------------------------- main loop
    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model."""
        print("Step 6: Starting training")
        episode_infos = deque(maxlen=100)

        updates = int(self.config.get("updates", 1))
        for update in range(updates):
            learning_rate = polynomial_decay(
                self.lr_schedule["initial"],
                self.lr_schedule["final"],
                self.lr_schedule["max_decay_steps"],
                self.lr_schedule["power"],
                update,
            )
            beta = polynomial_decay(
                self.beta_schedule["initial"],
                self.beta_schedule["final"],
                self.beta_schedule["max_decay_steps"],
                self.beta_schedule["power"],
                update,
            )
            clip_range = polynomial_decay(
                self.cr_schedule["initial"],
                self.cr_schedule["final"],
                self.cr_schedule["max_decay_steps"],
                self.cr_schedule["power"],
                update,
            )

            sampled_episode_info = self._sample_training_data()
            self.buffer.prepare_batch_dict()

            training_stats = self._train_epochs(learning_rate, clip_range, beta)
            training_stats = np.mean(training_stats, axis=0)

            episode_infos.extend(sampled_episode_info)
            episode_result = self._process_episode_info(episode_infos)

            rm = episode_result.get("reward_mean", 0.0)
            rs = episode_result.get("reward_std", 0.0)
            lm = episode_result.get("length_mean", 0.0)
            ls = episode_result.get("length_std", 0.0)
            sp = episode_result.get("success_percent", None)

            if sp is not None:
                result = (
                    "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success={:.2f} "
                    "pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}"
                ).format(
                    update,
                    rm,
                    rs,
                    lm,
                    ls,
                    sp,
                    training_stats[0],
                    training_stats[1],
                    training_stats[3],
                    training_stats[2],
                    torch.mean(self.buffer.values),
                    torch.mean(self.buffer.advantages),
                )
            else:
                result = (
                    "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} "
                    "pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}"
                ).format(
                    update,
                    rm,
                    rs,
                    lm,
                    ls,
                    training_stats[0],
                    training_stats[1],
                    training_stats[3],
                    training_stats[2],
                    torch.mean(self.buffer.values),
                    torch.mean(self.buffer.advantages),
                )

            print(result)

            self._write_training_summary(update, training_stats, episode_result)
            video_every = self.config.get("video_every")
            if video_every and update % video_every == 0:
                self._record_eval_episode(update)

            del self.buffer.samples_flat
            self.buffer.samples_flat = []
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._save_model()
