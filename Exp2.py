from __future__ import annotations
import os
import json
import random
import numpy as np
from collections import deque
from typing import Dict, Callable
import csv

import cv2
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces

# stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
max_contexts = 10
num_stack = 4
frame_shape = (84, 84)  # H, W
features_dim = 512
skip = 4
total_timesteps = 1_250_000
tensorboard_log = "./ppo_carl_mario_tb/"
episodes = 0

# ---------------- Modular Logger ----------------
class ContextLogger:
    """
    Logs rewards separately for each context_id & inertia.
    Generates:
    - JSON logs
    - PDF plots (full history)
    - PNG plots (last 100 episodes)
    Prints avg reward and episode reward to console.
    """
    def __init__(self, save_dir="ctx_logs"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.ctx_rewards = {}   # (ctx_id, inertia) -> list of rewards
        self.episode_counts = {}  # episodes per context

    def log_reward(self, ctx_id: int, inertia: float, reward: float):
        key = (ctx_id, inertia)
        if key not in self.ctx_rewards:
            self.ctx_rewards[key] = []
            self.episode_counts[key] = 0

        self.ctx_rewards[key].append(reward)
        self.episode_counts[key] += 1
        ep_count = self.episode_counts[key]

        # Print to console
        avg_reward = np.mean(self.ctx_rewards[key])
        print(f"[Context {ctx_id} | inertia={inertia:.3f}] Episode {ep_count} | "
              f"Reward: {reward:.2f} | Avg reward: {avg_reward:.2f}")

        # Plot/save periodically
        if ep_count % 100 == 0:
            self._plot_last_100(key)
        if ep_count % 250 == 0:
            self._plot_full_history(key)
            self._save_json(key)

    # ---------------- plotting ----------------
    def _plot_full_history(self, key):
        ctx_id, inertia = key
        rewards = np.array(self.ctx_rewards[key])
        avg_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)

        plt.figure(figsize=(10,5))
        plt.plot(np.arange(1, len(avg_rewards)+1), avg_rewards, linewidth=1.5)
        plt.title(f"Context {ctx_id} | inertia={inertia:.3f}\nRunning Mean Reward (Full)")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.grid(alpha=0.2)
        out = os.path.join(self.save_dir, f"ctx_{ctx_id}_inertia_{inertia:.3f}_full.pdf")
        plt.savefig(out, dpi=300)
        plt.close()

    def _plot_last_100(self, key):
        ctx_id, inertia = key
        rewards = np.array(self.ctx_rewards[key][-100:])
        window = min(10, len(rewards))
        running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        x_axis = np.arange(window, len(rewards)+1)

        plt.figure(figsize=(10,5))
        plt.plot(x_axis, running_avg, linewidth=1.5)
        plt.title(f"Context {ctx_id} | inertia={inertia:.3f}\nRunning Avg (Last 100)")
        plt.xlabel("Episode (last 100)")
        plt.ylabel("Average Reward")
        plt.grid(alpha=0.2)
        out = os.path.join(self.save_dir, f"ctx_{ctx_id}_inertia_{inertia:.3f}_last100.png")
        plt.savefig(out, dpi=300)
        plt.close()

    # ---------------- JSON ----------------
    def _save_json(self, key):
        ctx_id, inertia = key
        rewards = self.ctx_rewards[key]
        obj = {
            "context_id": ctx_id,
            "inertia": inertia,
            "rewards": rewards,
            "episodes": list(range(1, len(rewards)+1))
        }
        out = os.path.join(self.save_dir, f"ctx_{ctx_id}_inertia_{inertia:.3f}.json")
        with open(out, "w") as f:
            json.dump(obj, f, indent=4)


# ---------------- safety helpers ----------------
def _ensure_parent_dir_for_file(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def save_json_log(log_data, file_path):
    _ensure_parent_dir_for_file(file_path)
    with open(file_path, "w") as f:
        json.dump(log_data, f, indent=4)

# ---------------- context definitions ----------------
def get_context_features():
    return {
        "level_width": 400.0,
        "level_index": 5,
        "noise_seed": 50_000,
        "mario_state": 2,
        "mario_inertia": -1.0,
    }

def context_sampler(idx: int, max_ctx: int = max_contexts):
    base = get_context_features()
    aggregation = 1.0 / max_ctx
    inertia = 0.5 + idx * aggregation
    return {
        "context_id": int(idx),
        "level_width": float(base["level_width"]),
        "level_index": int(base["level_index"]),
        "noise_seed": int(base["noise_seed"]),
        "mario_state": int(base["mario_state"]),
        "mario_inertia": float(inertia),
    }

def one_hot(index: int, size: int):
    v = np.zeros(size, dtype=np.float32)
    v[int(index) % size] = 1.0
    return v

def context_to_vec_raw(ctx: Dict) -> np.ndarray:
    level_width = np.array([float(ctx["level_width"])], dtype=np.float32)
    level_idx_oh = one_hot(ctx["level_index"], 14)
    noise_seed = np.array([float(ctx["noise_seed"])], dtype=np.float32)
    mario_state_oh = one_hot(ctx["mario_state"], 3)
    inertia = np.array([float(ctx["mario_inertia"])], dtype=np.float32)
    return np.concatenate([level_width, level_idx_oh, noise_seed, mario_state_oh, inertia], axis=0)

# normalization settings
_CONTEXT_NORM = {
    "level_width_min": 0.0,
    "level_width_max": 500.0,
    "noise_seed_max": float(2**31 - 1),
    "inertia_min": 0.5,
    "inertia_max": 1.5,
}

def normalize_context_vec(raw_vec: np.ndarray) -> np.ndarray:
    assert raw_vec.shape[0] == 20, f"expected context len 20, got {raw_vec.shape[0]}"
    out = np.empty_like(raw_vec, dtype=np.float32)
    # level_width
    out[0] = np.clip((raw_vec[0] - _CONTEXT_NORM["level_width_min"]) / (_CONTEXT_NORM["level_width_max"] - _CONTEXT_NORM["level_width_min"]), 0.0, 1.0)
    # one-hot level_index
    out[1:15] = raw_vec[1:15]
    # noise seed scaled
    out[15] = np.clip(raw_vec[15] / _CONTEXT_NORM["noise_seed_max"], 0.0, 1.0)
    # mario_state one-hot
    out[16:19] = raw_vec[16:19]
    # inertia mapped
    out[19] = np.clip((raw_vec[19] - _CONTEXT_NORM["inertia_min"]) / (_CONTEXT_NORM["inertia_max"] - _CONTEXT_NORM["inertia_min"]), 0.0, 1.0)
    return out

# ---------------- CARL import ----------------
try:
    from carl.envs.mario.carl_mario import CARLMarioEnv
except Exception as e:
    raise RuntimeError("CARLMarioEnv import failed: " + str(e))

# ---------------- Wrapper ----------------
class SkipAndGrayResizeWrapper(gym.Wrapper):
    """
    - Grayscale + resize
    - Stack `num_stack` frames (channels-first)
    - Attach normalized context vector
    - Injects context via Gymnasium `options` argument on reset (options={"context": ...})
    - Robust reward fallback (reads info if reward is missing)
    """
    def __init__(self, env: gym.Env, context: Dict, shape=(84,84), skip = 4, num_stack: int = 4, logger: ContextLogger = None):
        super().__init__(env)
        self.shape = (int(shape[0]), int(shape[1]))
        self.num_stack = int(num_stack)
        self.context = dict(context)
        self.frames = deque(maxlen=self.num_stack)
        self.skip = skip
        self.logger = logger  # <-- new: optional logger

        # sample context to get dim
        sample_raw = context_to_vec_raw(self.context)
        sample_norm = normalize_context_vec(sample_raw)
        context_dim = sample_norm.shape[0]

        self.observation_space = spaces.Dict({
            "frames": spaces.Box(low=0.0, high=1.0, shape=(self.num_stack, self.shape[0], self.shape[1]), dtype=np.float32),
            "context": spaces.Box(low=0.0, high=1.0, shape=(context_dim,), dtype=np.float32),
        })

        self.action_space = env.action_space
        self._prev_completed = 0.0
        self._ep_reward_accum = 0.0

    # ---- image processing ----
    def _process(self, obs) -> np.ndarray:
        if isinstance(obs, dict) and "obs" in obs:
            obs = obs["obs"]
        img = obs

        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1,2,0))
        else:
            print("You are screwed!!!!!!")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0)

    def _get_obs_frames(self):
        if len(self.frames) < self.num_stack:
            pad = self.frames[0] if self.frames else np.zeros((self.shape[0], self.shape[1]), dtype=np.float32)
            arr = [pad]*(self.num_stack - len(self.frames)) + list(self.frames)
        else:
            arr = list(self.frames)
        return np.stack(arr, axis=0).astype(np.float32)

    # ---- reset/step ----
    def reset(self, **kwargs):
        global episodes
        episodes += 1
        if episodes % 10 == 0:
            self.context["level_index"] = random.randint(0, 13)
        self.context["noise_seed"] = random.randint(0, 2**31 - 1)
        options = kwargs.pop("options", {}) or {}
        context_to_pass = {k: v for k, v in self.context.items() if k != "context_id"}
        options["context"] = context_to_pass
        kwargs["options"] = options
        ret = self.env.reset(**kwargs)
        self._prev_completed = 0.0
        self._ep_reward_accum = 0.0

        if isinstance(ret, tuple) and len(ret) == 2:
            obs, info = ret
        elif isinstance(ret, tuple):
            obs = ret[0]; info = {}
        else:
            obs = ret; info = {}

        frame = self._process(obs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(frame)

        obs_dict = {"frames": self._get_obs_frames(), "context": normalize_context_vec(context_to_vec_raw(self.context))}
        return obs_dict, info

    def step(self, action):
        total_reward = 0.0
        terminated, truncated = False, False
        info = {}

        for _ in range(self.skip):
            ret = self.env.step(action)
            if isinstance(ret, tuple):
                if len(ret) == 5:
                    obs, _, term, trunc, info = ret
                    terminated |= bool(term)
                    truncated |= bool(trunc)
                elif len(ret) == 4:
                    obs, _, done, info = ret
                    terminated |= bool(done)
                    truncated |= False
                else:
                    raise ValueError(f"Unexpected env.step return length {len(ret)}")
            else:
                raise ValueError("Unexpected env.step return type; expected tuple")

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
            "context": normalize_context_vec(context_to_vec_raw(self.context))
        }

        total_reward = total_reward

        # --- log at episode end ---
        if terminated or truncated:
            if self.logger is not None:
                ctx_id = self.context.get("context_id", -1)
                inertia = self.context.get("mario_inertia", 0.0)
                self.logger.log_reward(ctx_id, inertia, self._ep_reward_accum)
            self._ep_reward_accum = 0.0
            self._prev_completed = 0.0

        return obs_dict, float(total_reward), terminated, truncated, info

# ---------------- Feature extractor ----------------
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
            nn.ReLU()
        )

    def forward(self, observations):
        frames = observations["frames"].float()
        context = observations["context"].float()
        frame_feat = self.cnn(frames)
        ctx_feat = self.context_proj(context)
        fused = torch.cat([frame_feat, ctx_feat], dim=1)
        return self.final_layer(fused)

# ---------------- env factories ----------------
def make_env_fn(idx: int, logger: ContextLogger = None) -> Callable[[], gym.Env]:
    def _init():
        base_env = CARLMarioEnv()
        ctx = context_sampler(idx, max_ctx=max_contexts)
        wrapped = SkipAndGrayResizeWrapper(base_env, context=ctx, shape=frame_shape, num_stack=num_stack, logger=logger)
        return wrapped
    return _init

def make_env_fn1(inertia: float) -> Callable[[], gym.Env]:
    def _init():
        base_env = CARLMarioEnv()
        ctx = context_sampler(0, max_ctx=max_contexts)
        ctx["mario_inertia"] = float(inertia)
        ctx["noise_seed"] = random.randint(0, 2**31 - 1)
        wrapped = SkipAndGrayResizeWrapper(base_env, context=ctx, shape=frame_shape, num_stack=num_stack)
        return wrapped
    return _init

# ------------------- Custom Evaluator ---------------------

def evaluate_model(model, max_contexts: int = max_contexts, episodes_per_context: int = 10, outdir: str = "eval_results"):
    for i in range(10):
        print("Inside evaluate_model...")  # small sanity print
    global eval_Debug_flag
    eval_Debug_flag = True  # enable debug prints during eval

    # --- build test data ---
    test_data = []
    aggregation = 1/max_contexts
    for i in range(max_contexts):
        low = 0.5 + aggregation * i
        high = low + aggregation
        val = random.uniform(low, high)
        test_data.append(val)

    envList = []

    # --- create envs ---
    for val in test_data:
        ctxIdx = int((val - 0.5) // aggregation)
        ctx = context_sampler(ctxIdx, max_ctx=max_contexts)
        ctx["mario_inertia"] = val
        ctx["noise_seed"] = np.random.randint(0, 2**31 - 1)
        ctx["level_index"] = random.randint(0, 13)

        base_env = CARLMarioEnv()
        wrapped = SkipAndGrayResizeWrapper(
            base_env,
            context=ctx,
            skip=skip,
            shape=frame_shape,
            num_stack=num_stack
        )

        envList.append(wrapped)


    # --- evaluate each env ---
    for idx, base_env in enumerate(envList):
        Noise = base_env.context["noise_seed"]
        Inertia = base_env.context["mario_inertia"]

        total_rewards = []
        completions = []

        for ep in range(episodes_per_context):
            # Monitor accepts a gym.Env; base_env is already wrapped but wrapping again is ok
            env = base_env

            # reset may return (obs, info) for gymnasium; our wrapper returns (obs, info)

            ret = env.reset(seed=np.random.randint(0, 2**31 - 1))


            if isinstance(ret, tuple) and len(ret) == 2:
                obs, info = ret
            else:
                obs = ret
                info = {}

            done = False
            trunc = False
            total = 0.0

            while not (done or trunc):
                # SB3 expects numpy arrays / dicts — MultiInputPolicy accepts dict(obs)
                action, _ = model.predict(obs, deterministic=False)
                ret_step = env.step(action)
                if isinstance(ret_step, tuple):
                    if len(ret_step) == 5:
                        obs, reward, terminated, truncated, info = ret_step
                        done = bool(terminated or truncated)
                        trunc = bool(truncated)
                    elif len(ret_step) == 4:
                        obs, reward, terminated, info = ret_step
                        done = bool(terminated)
                        trunc = False
                    else:
                        raise ValueError("Unexpected env.step return format during eval")
                    total += float(reward)
                else:
                    raise ValueError("Unexpected step return type during eval")

            
            total_rewards.append(total)
            completions.append(info.get("completed", 0))

        # --- compute stats ---
        avg_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
        completion_rate = float(np.mean(completions)) if completions else 0.0

        print("Eval Results Below:")
        print(
            f"Eval Context {idx} | "
            f"Inertia: {Inertia:.2f} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Completion Rate: {completion_rate:.2f}"
        )

        statDict = {
            "Inertia": Inertia,
            "AvgReward": avg_reward,
            "AvgCompletionRate": completion_rate,
            "total_rewards": total_rewards,
            "completions": completions,
            "Noise": Noise
        }

        # write results
        os.makedirs(outdir, exist_ok=True)
        json_path = os.path.join(outdir, f"context_{idx}_results.json")
        with open(json_path, "w", newline="") as f:
            json.dump(statDict, f, indent=4)

        csv_path = os.path.join(outdir, f"context_{idx}_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "TotalReward", "Completed"])
            for ep_index in range(len(total_rewards)):
                writer.writerow([ep_index + 1, total_rewards[ep_index], completions[ep_index]])

# ---------------- MAIN (training + eval) ----------------
if __name__ == "__main__":
    # reproducibility seeds
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # create vectorized train env
    logger = ContextLogger(save_dir="ctx_logs")
    env_fns = [make_env_fn(i, logger=logger) for i in range(max_contexts + 1)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)


    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
    )

    # PPO hyperparams: keep sensible defaults; tune later
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
    )

    # train
    print("Starting training...")


    model.learn(total_timesteps=total_timesteps)

    model.save("ppo_carl_mario_fixed")

    # ---------------- EVAL ----------------
    test_data = []
    for i in range(10):
        low = 0.5 + 0.1 * i
        high = low + 0.1
        test_data.append(random.uniform(low, high))

    eval_env_fns = [make_env_fn1(inertia) for inertia in test_data]
    eval_vec_env = DummyVecEnv(eval_env_fns)
    eval_vec_env = VecMonitor(eval_vec_env)
    # evaluate and get per-episode rewards
    with torch.no_grad():
        evaluate_model(model,max_contexts)
        episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_vec_env,
        n_eval_episodes=100,
        deterministic=False,
        render=False,
        return_episode_rewards=True,
        )
    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "eval_results.txt")

    with open(txt_path, "w") as f:
        f.write("=== Evaluation results ===\n")
        for idx, (r, l) in enumerate(zip(episode_rewards, episode_lengths), start=1):
            line = f"Episode {idx:02d}: reward={r:.3f}, length={l}\n"
            print(line, end="")   # keep printing to console
            f.write(line)
        summary = f"Eval finished — mean reward: {episode_rewards.mean():.3f}, std: {episode_rewards.std():.3f}\n"
        print(summary, end="")
        f.write(summary)

    print("=== Evaluation results ===")
    for idx, (r, l) in enumerate(zip(episode_rewards, episode_lengths), start=1):
        print(f"Episode {idx:02d}: reward={r:.3f}, length={l}")
    print(f"Eval finished — mean reward: {episode_rewards.mean():.3f}, std: {episode_rewards.std():.3f}")
