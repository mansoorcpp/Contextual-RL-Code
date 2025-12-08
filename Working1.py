from __future__ import annotations
import os
import json
import random
import csv
from collections import deque
from typing import Dict, Callable, List

import numpy as np
import cv2
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
max_contexts = 10
num_stack = 4
frame_shape = (84, 84)  # H, W
features_dim = 512
skip = 4
total_timesteps = 2_000_000
tensorboard_log = "./ppo_carl_mario_tb/"
N_LEVEL_VARIANTS = 15  # how many different maps per inertia/env
Eval = False


def make_carl_contexts_for_inertia(inertia: float) -> Dict[int, Dict]:
    base = get_context_features()
    contexts: Dict[int, Dict] = {}

    # choose even level indices: 0, 2, 4, ...
    level_indices = list(range(0, N_LEVEL_VARIANTS, 3))  # [0,2,4,6,8,10,12] for 14

    for ctx_id, level_idx in enumerate(level_indices):   # ctx_id: 0..len(level_indices)-1
        contexts[ctx_id] = {
            "level_width": int(base["level_width"]),
            "level_index": int(level_idx),
            "noise_seed": int(ctx_id * 10000),
            "mario_state": int(base["mario_state"]),
            "mario_inertia": float(inertia),
        }

    return contexts

# =========================
# CONTEXT LOGGER
# =========================
class ContextLogger:
    """
    Logs rewards separately for each (context_id, inertia) pair.

    - JSON logs
    - PDF plots for full history
    - PNG plots for last 100 episodes

    Also prints per-episode reward & running average to console.
    """

    def __init__(self, save_dir: str = "ctx_logs"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # key: (ctx_id, inertia) -> list of rewards
        self.ctx_rewards: Dict[tuple, List[float]] = {}
        # episodes per context
        self.episode_counts: Dict[tuple, int] = {}

    def log_reward(self, ctx_id: int, inertia: float, reward: float):
        key = (ctx_id, float(inertia))
        if key not in self.ctx_rewards:
            self.ctx_rewards[key] = []
            self.episode_counts[key] = 0

        self.ctx_rewards[key].append(float(reward))
        self.episode_counts[key] += 1
        ep_count = self.episode_counts[key]

        avg_reward = float(np.mean(self.ctx_rewards[key]))
        print(
            f"[Context {ctx_id} | inertia={inertia:.3f}] "
            f"Episode {ep_count} | Reward: {reward:.3f} | Avg: {avg_reward:.3f}"
        )

        # Plot/save periodically
        if ep_count % 100 == 0:
            self._plot_last_100(key)
        if ep_count % 250 == 0:
            self._plot_full_history(key)
            self._save_json(key)

    # ---------- plotting ----------
    def _plot_full_history(self, key: tuple):
        ctx_id, inertia = key
        rewards = np.array(self.ctx_rewards[key], dtype=np.float32)
        avg_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(1, len(avg_rewards) + 1), avg_rewards, linewidth=1.5)
        plt.title(f"Context {ctx_id} | inertia={inertia:.3f}\nRunning Mean Reward (Full)")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.grid(alpha=0.2)

        out = os.path.join(self.save_dir, f"ctx_{ctx_id}_inertia_{inertia:.3f}_full.pdf")
        plt.savefig(out, dpi=300)
        plt.close()

    def _plot_last_100(self, key: tuple):
        ctx_id, inertia = key
        rewards = np.array(self.ctx_rewards[key][-100:], dtype=np.float32)
        if len(rewards) == 0:
            return
        window = min(10, len(rewards))
        running_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        x_axis = np.arange(window, len(rewards) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(x_axis, running_avg, linewidth=1.5)
        plt.title(f"Context {ctx_id} | inertia={inertia:.3f}\nRunning Avg (Last 100)")
        plt.xlabel("Episode (last 100)")
        plt.ylabel("Average Reward")
        plt.grid(alpha=0.2)

        out = os.path.join(
            self.save_dir, f"ctx_{ctx_id}_inertia_{inertia:.3f}_last100.png"
        )
        plt.savefig(out, dpi=300)
        plt.close()

    # ---------- JSON ----------
    def _save_json(self, key: tuple):
        ctx_id, inertia = key
        rewards = self.ctx_rewards[key]
        obj = {
            "context_id": ctx_id,
            "inertia": inertia,
            "rewards": rewards,
            "episodes": list(range(1, len(rewards) + 1)),
        }
        out = os.path.join(self.save_dir, f"ctx_{ctx_id}_inertia_{inertia:.3f}.json")
        with open(out, "w") as f:
            json.dump(obj, f, indent=4)

# =========================
# CONTEXT DEFINITIONS
# =========================
def get_context_features() -> Dict:
    """
    Base context values used by our own sampler.
    These should be consistent with CARLMarioEnv.get_context_features keys.
    """
    return {
        "level_width": 400,   # <-- int, not 400.0
        "level_index": 5,
        "noise_seed": 50_000,
        "mario_state": 0,
        "mario_inertia": 0.89,  # base, will override per context
    }


def context_sampler(idx: int, max_ctx: int = max_contexts) -> Dict:
    """
    Our discrete context sampler.

    Returns a dictionary with:
      - context_id (for our logging)
      - level_width, level_index, noise_seed, mario_state, mario_inertia
    """
    base = get_context_features()
    aggregation = 1.0 / max_ctx
    inertia = 0.5 + (idx + 0.5) * aggregation  # spans [0.5, 1.5) as idx increases

    return {
        "context_id": int(idx),
        "level_width": int(base["level_width"]),   # <-- force int
        "level_index": int(base["level_index"]),
        "noise_seed": int(base["noise_seed"]),
        "mario_state": int(base["mario_state"]),
        "mario_inertia": float(inertia),
    }


def one_hot(index: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    v[int(index) % size] = 1.0
    return v


def context_to_vec_raw(ctx: Dict) -> np.ndarray:
    """
    Raw context -> 19D vector: [level_width, 14x level_index OH, noise_seed,
                                 3x mario_state OH, inertia]
    """
    level_width = np.array([float(ctx["level_width"])], dtype=np.float32)
    level_idx_oh = one_hot(ctx["level_index"], 14)
    mario_state_oh = one_hot(ctx["mario_state"], 3)
    inertia = np.array([float(ctx["mario_inertia"])], dtype=np.float32)
    return np.concatenate(
        [level_width, level_idx_oh, mario_state_oh, inertia], axis=0
    )


# Normalization settings for 20D context vector
_CONTEXT_NORM = {
    "level_width_min": 0.0,
    "level_width_max": 500.0,
    "inertia_min": 0.5,
    "inertia_max": 1.5,
}


def normalize_context_vec(raw_vec: np.ndarray) -> np.ndarray:
    assert raw_vec.shape[0] == 19, f"expected context len 19, got {raw_vec.shape[0]}"
    out = np.empty_like(raw_vec, dtype=np.float32)

    # level_width
    out[0] = np.clip(
        (raw_vec[0] - _CONTEXT_NORM["level_width_min"])
        / (_CONTEXT_NORM["level_width_max"] - _CONTEXT_NORM["level_width_min"]),
        0.0,
        1.0,
    )
    # one-hot level_index
    out[1:15] = raw_vec[1:15]
    # mario_state one-hot
    out[15:18] = raw_vec[15:18]
    # inertia mapped
    out[18] = np.clip(
        (raw_vec[18] - _CONTEXT_NORM["inertia_min"])
        / (_CONTEXT_NORM["inertia_max"] - _CONTEXT_NORM["inertia_min"]),
        0.0,
        1.0,
    )
    return out


# =========================
# CARL IMPORT
# =========================
try:
    from carl.envs.mario.carl_mario import CARLMarioEnv
except Exception as e:
    raise RuntimeError("CARLMarioEnv import failed: " + str(e))


# =========================
# WRAPPER
# =========================
class SkipAndGrayResizeWrapper(gym.Wrapper):
    """
    - Grayscale + resize
    - Stack `num_stack` frames channels-first
    - Attach normalized context vector (fixed per active CARL context)
    - Reward = delta(info["completed"]) per step
    - Change CARLMarioEnv.context_id every `level_change_interval` episodes
      and call `_update_context()` to switch to a new pre-generated map.
    """

    def __init__(
        self,
        env: gym.Env,
        context: Dict,
        shape=(84, 84),
        skip: int = 4,
        num_stack: int = 4,
        logger: ContextLogger | None = None,
        level_change_interval: int | None = 10,
    ):
        super().__init__(env)
        self.shape = (int(shape[0]), int(shape[1]))
        self.num_stack = int(num_stack)
        self.context = dict(context)  # includes context_id for logging only
        self.frames = deque(maxlen=self.num_stack)
        self.skip = int(skip)
        self.logger = logger

        self.level_change_interval = level_change_interval
        self.episode_counter = 0

        # Track CARL context usage
        self._initialized = False
        self._current_carl_context_id: int | None = None
        self._num_carl_contexts: int = (
            len(getattr(env, "contexts", {})) if isinstance(env, CARLMarioEnv) else 0
        )

        # Precompute context vector dimension
        sample_raw = context_to_vec_raw(self.context)
        sample_norm = normalize_context_vec(sample_raw)
        context_dim = sample_norm.shape[0]

        self.observation_space = spaces.Dict(
            {
                "frames": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_stack, self.shape[0], self.shape[1]),
                    dtype=np.float32,
                ),
                "context": spaces.Box(
                    low=0.0, high=1.0, shape=(context_dim,), dtype=np.float32
                ),
            }
        )

        self.action_space = env.action_space

        self._prev_completed = 0.0
        self._ep_reward_accum = 0.0

    # ---------- image processing ----------
    def _process(self, obs) -> np.ndarray:
        # CARL can add context to obs; handle dict{"obs", ...}
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs["obs"]

        img = np.array(obs, copy=False)

        # if (C,H,W) with C=3, convert to HWC
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        img = img.astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(
            gray,
            (self.shape[1], self.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        return resized.astype(np.float32) / 255.0

    def _get_obs_frames(self) -> np.ndarray:
        if len(self.frames) < self.num_stack:
            pad = (
                self.frames[0]
                if len(self.frames) > 0
                else np.zeros(self.shape, dtype=np.float32)
            )
            arr = [pad] * (self.num_stack - len(self.frames)) + list(self.frames)
        else:
            arr = list(self.frames)
        return np.stack(arr, axis=0).astype(np.float32)

    # ---------- reset / step ----------
    def reset(self, **kwargs):
        """
        Reset the environment.

        First-ever reset:
          - delegate to CARLMarioEnv.reset() so it can build levels and contexts.

        Subsequent resets (when env is CARLMarioEnv):
          - every `level_change_interval` episodes, switch CARLMarioEnv.context_id
            to another pre-generated context and call _update_context()
          - reset only the underlying MarioEnv (carl_env.env.reset)
          - keep wrapper context synchronized with carl_env.context
        """
        global Eval
        self.episode_counter += 1
        self._prev_completed = 0.0
        self._ep_reward_accum = 0.0

        base_env = self.env  # potentially CARLMarioEnv

        # -------- first-ever reset: let CARL handle everything --------
        if not self._initialized:
            ret = base_env.reset(**kwargs)
            obs, info = ret

            self._num_carl_contexts = len(getattr(base_env, "contexts", {}))
            self._current_carl_context_id = getattr(base_env, "context_id", 0)

            # sync wrapper context with CARL context (keep logging-only context_id)
            carl_ctx = dict(getattr(base_env, "context", {}))
            for k, v in carl_ctx.items():
                if k != "context_id":
                    self.context[k] = v
            self._initialized = True

        else:
            # -------- subsequent resets: manage context_id + _update_context --------
            carl_env: CARLMarioEnv = base_env
            if self._current_carl_context_id is None:
                self._current_carl_context_id = 0

            if ((self.episode_counter - 1) % self.level_change_interval == 0):
                # ensure we know how many contexts exist
                if self._num_carl_contexts == 0:
                    self._num_carl_contexts = len(getattr(carl_env, "contexts", {}))

                self._current_carl_context_id = (
                    self._current_carl_context_id + 1
                ) % self._num_carl_contexts

                # set CARLMarioEnv's active context and apply it
                carl_env.context_id = self._current_carl_context_id
                carl_env.context = carl_env.contexts[self._current_carl_context_id]
                # IMPORTANT: _update_context() is now called with valid context_id
                carl_env._update_context()

                # sync wrapper context with CARL context (keep logging-only context_id)
                carl_ctx = dict(carl_env.context)

                for k, v in carl_ctx.items():
                    if k != "context_id":
                        self.context[k] = v

            # Now reset only the underlying MarioEnv
            mario_env = carl_env.env
            ret2 = mario_env.reset(**kwargs)

            obs, info = ret2
            for i in range(5):
                print("Changed context and reset MarioEnv")

        # ---------- build debug info ----------
        if ((self.episode_counter - 1) % self.level_change_interval == 0):
            debug_info = {}
            debug_info["wrapper_context"] = dict(self.context)

            carl_env = self.env
            current_id = getattr(carl_env, "context_id", None)
            carl_current_ctx = dict(getattr(carl_env, "context", {}))

            debug_info["carl_context_id"] = current_id
            debug_info["carl_context"] = carl_current_ctx

            mario_env = carl_env.env
            debug_info["mario_level_index"] = getattr(
                mario_env, "current_level_idx", None
            )
            debug_info["mario_state"] = getattr(mario_env, "mario_state", None)
            debug_info["mario_inertia"] = getattr(mario_env, "mario_inertia", None)

            # Build debug message
            debug_msg = f"\n[RESET] Episode {self.episode_counter}\n"
            debug_msg += f"[RESET] Wrapper context: {debug_info['wrapper_context']}\n"
            debug_msg += f"[RESET] CARLMarioEnv context_id={current_id}, context={carl_current_ctx}\n"
            debug_msg += (
                f"[RESET] MarioEnv state: "
                f"level_index={debug_info['mario_level_index']}, "
                f"mario_state={debug_info['mario_state']}, "
                f"mario_inertia={debug_info['mario_inertia']}\n"
            )

            print(debug_msg, end="")

            # Write to txt file in ctx_logs
            if not Eval:
                os.makedirs("ctx_logs", exist_ok=True)
                debug_file = os.path.join("ctx_logs", "reset_debug.txt")
                with open(debug_file, "a") as f:
                    f.write(debug_msg)
            else:
                os.makedirs("eval_results", exist_ok=True)
                debug_file = os.path.join("eval_results", "reset_debug.txt")
                with open(debug_file, "a") as f:
                    f.write(debug_msg)

        # ---------- build stacked-frame observation ----------
        frame = self._process(obs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(frame)

        obs_dict = {
            "frames": self._get_obs_frames(),
            "context": normalize_context_vec(context_to_vec_raw(self.context)),
        }

        return obs_dict, info

    def step(self, action):
        total_reward = 0.0
        terminated, truncated = False, False
        info = {}

        for _ in range(self.skip):
            ret = self.env.step(action)

            if isinstance(ret, tuple) and len(ret) == 5:
                obs, _, term, trunc, info = ret
                terminated |= bool(term)
                truncated |= bool(trunc)
            elif isinstance(ret, tuple) and len(ret) == 4:
                obs, _, done, info = ret
                terminated |= bool(done)
                truncated |= False
            else:
                raise ValueError(f"Unexpected env.step return: {ret}")

            completed = float(info.get("completed", 0.0))
            reward_delta = completed - self._prev_completed
            self._prev_completed = completed

            total_reward += reward_delta
            self._ep_reward_accum += reward_delta

            frame = self._process(obs)
            self.frames.append(frame)

            if terminated or truncated:
                break

        obs_dict = {
            "frames": self._get_obs_frames(),
            "context": normalize_context_vec(context_to_vec_raw(self.context)),
        }

        # Log at episode end
        if (terminated or truncated) and self.logger is not None:
            ctx_id = int(self.context.get("context_id", -1))
            inertia = float(self.context.get("mario_inertia", 0.0))
            self.logger.log_reward(ctx_id, inertia, self._ep_reward_accum)
            self._ep_reward_accum = 0.0
            self._prev_completed = 0.0

        return obs_dict, float(total_reward), terminated, truncated, info


# =========================
# FEATURE EXTRACTOR
# =========================
class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        in_channels = observation_space["frames"].shape[0]
        H = observation_space["frames"].shape[1]
        W = observation_space["frames"].shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            n_flat = self.cnn(dummy).shape[1]

        context_size = observation_space["context"].shape[0]
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_size),
            nn.Linear(context_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_flat),
            nn.ReLU(),
        )

        self.final_layer = nn.Sequential(
            nn.Linear(n_flat + n_flat, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    def forward(self, observations):
        frames = observations["frames"].float()
        context = observations["context"].float()

        frame_feat = self.cnn(frames)
        ctx_feat = self.context_proj(context)

        fused = torch.cat([frame_feat, ctx_feat], dim=1)
        return self.final_layer(fused)


# ... all your previous imports, logger, context code, wrapper, etc. stay unchanged ...

# =========================
# ENV FACTORIES
# =========================

def make_train_env_fn(
    idx: int, logger: ContextLogger | None = None
) -> Callable[[], gym.Env]:
    """
    Each training env corresponds to one inertia, but internally has multiple
    CARL contexts (different maps). The wrapper will switch context_id every
    `level_change_interval` episodes.
    """
    def _init():
        # inertia from your original sampler
        base_ctx = context_sampler(idx, max_ctx=max_contexts)
        inertia = float(base_ctx["mario_inertia"])

        # Build multiple CARL contexts for this inertia
        carl_contexts = make_carl_contexts_for_inertia(inertia)  # keys: 0..N_LEVEL_VARIANTS-1

        base_env = CARLMarioEnv(
            contexts=carl_contexts,
            obs_context_features=[],
            obs_context_as_dict=False,
        )

        # Initial wrapper context: use context 0 plus logging-only context_id
        initial_ctx = dict(carl_contexts[0])
        initial_ctx["context_id"] = idx  # for logging only

        wrapped = SkipAndGrayResizeWrapper(
            base_env,
            context=initial_ctx,
            shape=frame_shape,
            skip=skip,
            num_stack=num_stack,
            logger=logger,
            level_change_interval=750,  # map changes every 750 episodes
        )
        return wrapped

    return _init

def make_eval_env_fn(
    inertia: float,
    level_change_interval: int = 1,
) -> Callable[[], gym.Env]:
    """
    For evaluation: single env with fixed inertia but multiple CARL contexts
    (different maps). The wrapper switches map (context_id) every
    `level_change_interval` episodes.
    """
    def _init():
        global Eval
        Eval = True
        carl_contexts = make_carl_contexts_for_inertia(inertia)

        base_env = CARLMarioEnv(
            contexts=carl_contexts,
            obs_context_features=[],
            obs_context_as_dict=False,
        )

        initial_ctx = dict(carl_contexts[0])
        initial_ctx["context_id"] = 0  # logging only

        wrapped = SkipAndGrayResizeWrapper(
            base_env,
            context=initial_ctx,
            shape=frame_shape,
            skip=skip,
            num_stack=num_stack,
            logger=None,
            level_change_interval=level_change_interval,  # 1 => new map each episode
        )
        return wrapped

    return _init

# =========================
# CUSTOM EVALUATOR
# =========================
def evaluate_model(
    model: PPO,
    inertias: List[float],
    episodes_per_context: int = 10,
    outdir: str = "eval_results",
    deterministic: bool = True,
):
    """
    Evaluate the trained model on a list of inertia values.

    For each inertia:
      - run `episodes_per_context` episodes
      - each episode uses a NEW randomly generated map/level
        (via level_change_interval=1 in the wrapper)
      - log per-episode:
          * context_id
          * level_index
          * noise_seed
          * mario_state
          * mario_inertia
          * total_reward
          * completed
          * length
      - save JSON + CSV with full episode results
    """
    os.makedirs(outdir, exist_ok=True)

    for idx, inertia in enumerate(inertias):
        # Create env where inertia is fixed, but map changes every episode
        env_fn = make_eval_env_fn(inertia, level_change_interval=1)
        env = env_fn()  # single non-vectorized env

        episode_data: List[Dict] = []

        for ep in range(episodes_per_context):
            # Reset env (wrapper) with a random seed
            ret = env.reset(seed=np.random.randint(0, 2**31 - 1))
            if isinstance(ret, tuple) and len(ret) == 2:
                obs, info = ret
            else:
                obs, info = ret, {}

            done = False
            trunc = False
            ep_reward = 0.0
            last_completed = 0.0
            ep_len = 0

            # ---- read context info at episode start ----
            # env is SkipAndGrayResizeWrapper
            wrapper = env
            carl_env = wrapper.env          # CARLMarioEnv
            current_cid = getattr(carl_env, "context_id", None)
            carl_ctx = dict(getattr(carl_env, "context", {}))

            level_index = carl_ctx.get("level_index", None)
            noise_seed = carl_ctx.get("noise_seed", None)
            mario_state = carl_ctx.get("mario_state", None)
            mario_inertia = carl_ctx.get("mario_inertia", None)

            # ---- run episode ----
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=deterministic)
                step_ret = env.step(action)
                if len(step_ret) == 5:
                    obs, reward, terminated, truncated, info = step_ret
                    done = bool(terminated or truncated)
                    trunc = bool(truncated)
                else:
                    raise ValueError("Unexpected env.step return format in eval")

                ep_reward += float(reward)
                last_completed = float(info.get("completed", last_completed))
                ep_len += 1

            # store per-episode data
            episode_data.append(
                {
                    "episode": ep + 1,
                    "context_id": int(current_cid) if current_cid is not None else None,
                    "level_index": int(level_index) if level_index is not None else None,
                    "noise_seed": int(noise_seed) if noise_seed is not None else None,
                    "mario_state": int(mario_state) if mario_state is not None else None,
                    "mario_inertia": float(mario_inertia) if mario_inertia is not None else None,
                    "total_reward": float(ep_reward),
                    "completed": float(last_completed),
                    "length": int(ep_len),
                }
            )

        # ---- aggregate stats for this inertia ----
        total_rewards = [d["total_reward"] for d in episode_data]
        completions = [d["completed"] for d in episode_data]

        avg_reward = float(np.mean(total_rewards)) if len(total_rewards) > 0 else 0.0
        avg_completion = float(np.mean(completions)) if len(completions) > 0 else 0.0

        print(
            f"[EVAL] Context {idx} | inertia={inertia:.3f} | "
            f"Avg Reward={avg_reward:.3f} | Avg Completion={avg_completion:.3f}"
        )

        # ---- save JSON ----
        stat_dict = {
            "inertia": float(inertia),
            "episodes_per_context": int(episodes_per_context),
            "avg_reward": avg_reward,
            "avg_completion": avg_completion,
            "episodes": episode_data,  # full per-episode info
        }
        json_path = os.path.join(outdir, f"context_{idx}_results.json")
        with open(json_path, "w") as f:
            json.dump(stat_dict, f, indent=4)

        # ---- save CSV ----
        csv_path = os.path.join(outdir, f"context_{idx}_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Episode",
                    "ContextID",
                    "LevelIndex",
                    "NoiseSeed",
                    "MarioState",
                    "MarioInertia",
                    "TotalReward",
                    "Completed",
                    "Length",
                ]
            )
            for d in episode_data:
                writer.writerow(
                    [
                        d["episode"],
                        d["context_id"],
                        d["level_index"],
                        d["noise_seed"],
                        d["mario_state"],
                        d["mario_inertia"],
                        d["total_reward"],
                        d["completed"],
                        d["length"],
                    ]
                )



# =========================
# MAIN (TRAIN + EVAL)
# =========================
if __name__ == "__main__":
    # ---- reproducibility ----
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- vectorized training env ----
    logger = ContextLogger(save_dir="ctx_logs")
    env_fns = [make_train_env_fn(i, logger=logger) for i in range(max_contexts)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        clip_range=0.3,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef= 0.03
    )

    # ---- TRAIN ----
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_carl_mario_fixed")

    # ---- EVAL (custom) ----
    # Create 10 test inertia values in [0.5, 1.5)
    test_inertias: List[float] = []
    step = 1.0 / max_contexts
    for i in range(max_contexts):
        low = 0.5 + step * i
        high = low + step
        test_inertias.append(random.uniform(low, high))


    # ---- EVAL (SB3 evaluate_policy for a vec env) ----
    eval_env_fns = [make_eval_env_fn(inertia) for inertia in test_inertias]
    eval_vec_env = DummyVecEnv(eval_env_fns)
    eval_vec_env = VecMonitor(eval_vec_env)

    with torch.no_grad():
        evaluate_model(
            model,
            inertias=test_inertias,
            episodes_per_context=50,
            outdir="eval_results",
            deterministic=True,
        )
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            eval_vec_env,
            n_eval_episodes=100,
            deterministic=True,
            render=False,
            return_episode_rewards=True,
        )
    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "eval_results_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=== Evaluation results ===\n")
        for idx, (r, l) in enumerate(zip(episode_rewards, episode_lengths), start=1):
            line = f"Episode {idx:02d}: reward={r:.3f}, length={l}\n"
            print(line, end="")
            f.write(line)
        summary = (
            f"Eval finished — mean reward: {episode_rewards.mean():.3f}, "
            f"std: {episode_rewards.std():.3f}\n"
        )
        print(summary, end="")
        f.write(summary)
