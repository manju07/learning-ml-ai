# Advanced Reinforcement Learning: Complete Guide

## Table of Contents
1. [Introduction to Advanced RL](#introduction-to-advanced-rl)
2. [Soft Actor-Critic (SAC)](#soft-actor-critic-sac)
3. [Twin Delayed DDPG (TD3)](#twin-delayed-ddpg-td3)
4. [Model-Based RL](#model-based-rl)
5. [Offline RL](#offline-rl)
6. [MuZero and Learned World Models](#muzero-and-learned-world-models)
7. [Intrinsic Motivation and Curiosity](#intrinsic-motivation-and-curiosity)
8. [Multi-Agent RL](#multi-agent-rl)
9. [Practical Examples](#practical-examples)
10. [Best Practices](#best-practices)

---

## Introduction to Advanced RL

This guide covers algorithms and concepts beyond DQN and PPO: **off-policy** methods (SAC, TD3), **model-based** RL, **offline** RL, and **planning** with learned models (MuZero).

### Algorithm Comparison

| Algorithm | Policy | Action Space | Sample Efficiency | Stability |
|-----------|--------|--------------|-------------------|-----------|
| **PPO** | On-policy | Continuous | Low | High |
| **DDPG** | Off-policy | Continuous | Medium | Low |
| **TD3** | Off-policy | Continuous | Medium | Medium |
| **SAC** | Off-policy | Continuous | High | High |
| **CQL** | Off-policy | Both | Offline | Medium |

---

## Soft Actor-Critic (SAC)

**SAC** (Haarnoja et al., 2018) is an off-policy actor-critic that maximizes **entropy** alongside reward, enabling exploration and robustness.

### Key Ideas

- **Maximum entropy RL**: J = E[Σ r + α·H(π)]
- **Stochastic policy**: Sample actions for exploration
- **Automatic temperature**: Learns α (entropy coefficient)
- **Off-policy**: Reuse experience from replay buffer

### SAC Components

1. **Actor** (π): Outputs mean and log_std for Gaussian policy
2. **Critic** (Q): Two Q-networks (reduce overestimation)
3. **Target Q**: Soft update for stability

### Policy Update (Reparameterization)

a = tanh(μ + σ·ε), ε ~ N(0,1)

log π(a|s) = log π(μ,σ) - Σ log(1 - tanh²(u_i))  (correction for tanh)

### Losses

- **Critic**: TD loss with min of two Qs
- **Actor**: E[α·log π(a|s) - Q(s,a)]
- **Alpha**: -E[log π(a|s)] - target_entropy

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SquashedGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)
    
    def forward(self, obs, deterministic=False):
        h = self.net(obs)
        mean = self.mean(h)
        if deterministic:
            return torch.tanh(mean)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = (normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

def sac_update(actor, critic, target_critic, alpha, replay_buffer, gamma=0.99, tau=0.005):
    obs, act, rew, next_obs, done = replay_buffer.sample(256)
    
    # Critic loss
    with torch.no_grad():
        next_act, next_log_prob = actor(next_obs)
        next_q1, next_q2 = target_critic(next_obs, next_act)
        next_q = torch.min(next_q1, next_q2) - alpha * next_log_prob.unsqueeze(-1)
        target_q = rew.unsqueeze(-1) + gamma * (1 - done.unsqueeze(-1)) * next_q
    
    q1, q2 = critic(obs, act)
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    
    # Actor loss
    new_act, log_prob = actor(obs)
    q1, q2 = critic(obs, new_act)
    q = torch.min(q1, q2)
    actor_loss = (alpha * log_prob - q).mean()
    
    # Alpha (optional: fix alpha = 0.2)
    # alpha_loss = -log_alpha * (log_prob + target_entropy).detach().mean()
    
    return critic_loss, actor_loss
```

### Using Stable-Baselines3

```python
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("HalfCheetah-v4", n_envs=4)
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto"  # Automatic entropy tuning
)
model.learn(total_timesteps=1_000_000)
```

---

## Twin Delayed DDPG (TD3)

**TD3** (Fujimoto et al., 2018) improves DDPG with:
1. **Twin Q-networks**: Use min of two Qs (reduce overestimation)
2. **Delayed policy updates**: Update actor less frequently
3. **Target policy smoothing**: Add noise to target actions

```python
# TD3 pseudocode
# 1. Critic: target_q = r + γ * min_i Q'_i(s', a' + ε), ε ~ clip(N(0,σ), -c, c)
# 2. Update critics with TD loss
# 3. Every d steps: update actor to maximize Q(s, π(s))
```

```python
from stable_baselines3 import TD3

model = TD3(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=1_000_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    policy_delay=2,  # Update actor every 2 critic updates
    target_policy_noise=0.2,
    target_noise_clip=0.5
)
model.learn(total_timesteps=1_000_000)
```

---

## Model-Based RL

**Model-based RL** learns a **world model** (dynamics) and uses it for planning or training.

### Dynamics Model

Predict next state and reward: (s', r) = f(s, a)

### Categories

1. **Model-based planning**: Plan with learned model (e.g., MCTS)
2. **Model-based policy learning**: Generate synthetic experience
3. **World models**: Learn in latent space (e.g., Dreamer)

### Simple World Model

```python
class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.forward_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim + 1)  # next_obs + reward
        )
    
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        out = self.forward_net(x)
        next_obs, reward = out[:, :-1], out[:, -1:]
        return next_obs, reward

# Train on replay buffer
# Use for planning: rollout trajectories, optimize actions
```

### MBPO (Model-Based Policy Optimization)

1. Collect data with random/SAC policy
2. Train dynamics model
3. Generate short rollouts from model
4. Train policy on model-generated data
5. Periodically add real data

---

## Offline RL

**Offline RL** learns from a fixed dataset **without** environment interaction. Critical for healthcare, robotics (safe), recommendations.

### Challenges

- **Distribution shift**: Policy may visit OOD states
- **Extrapolation error**: Q-values overestimated for unseen actions
- **No exploration**: Cannot improve data collection

### Conservative Q-Learning (CQL)

CQL penalizes Q-values for actions **outside** the dataset:

L_CQL = E[log Σ_a exp(Q(s,a)) - E_a~π_β [Q(s,a)]]

Minimize Q for non-data actions → conservative estimates.

```python
# CQL loss (simplified)
def cql_loss(q_values, dataset_actions, alpha=1.0):
    # log-sum-exp over all actions (or sampled)
    log_sum_exp_q = torch.logsumexp(q_values, dim=1)
    dataset_q = q_values.gather(1, dataset_actions).squeeze()
    return alpha * (log_sum_exp_q - dataset_q).mean()
```

### IQL (Implicit Q-Learning)

- No explicit maximization over actions
- Use expectile regression for V and Q
- Extract policy via advantage-weighted regression

### Practical Offline RL

```python
# d4rl datasets: pip install d4rl
import d4rl
import gym

env = gym.make("halfcheetah-medium-v2")
dataset = env.get_dataset()  # obs, actions, rewards, next_obs, dones

# Use CQL, IQL, or BC from offline data
```

---

## MuZero and Learned World Models

**MuZero** (DeepMind) learns:
1. **Representation**: h = f(s)
2. **Dynamics**: h' = g(h, a)
3. **Prediction**: (p, v) = p(h) — policy and value

No explicit model of environment; model is **implicit** in latent space.

### MuZero for Planning

1. Encode state: h = f(s)
2. For each step: h = g(h, a), (p, v) = p(h)
3. MCTS in latent space using p, v, g
4. Select action, execute in env
5. Train on (s, a, r, ..., G) with consistency losses

### Simplified MuZero Style

```python
class MuZeroNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.representation = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + act_dim, hidden_dim),
            nn.ReLU()
        )
        self.prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)  # policy logits
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.reward_head = nn.Linear(hidden_dim, 1)
```

---

## Intrinsic Motivation and Curiosity

### Curiosity-Driven Exploration

**Curiosity**: Reward = r_extrinsic + β * r_intrinsic

**Intrinsic reward**: Prediction error (ICM) or random network distillation (RND).

### RND (Random Network Distillation)

- **Target network**: Random, fixed
- **Predictor**: Predict target’s output
- **Intrinsic reward**: Prediction error (higher in novel states)

```python
class RND(nn.Module):
    def __init__(self, obs_dim, embed_dim=128):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        for p in self.target.parameters():
            p.requires_grad = False
    
    def intrinsic_reward(self, obs):
        with torch.no_grad():
            target_feat = self.target(obs)
        pred_feat = self.predictor(obs)
        return F.mse_loss(pred_feat, target_feat, reduction='none').mean(-1)
```

---

## Multi-Agent RL

### Types

- **Cooperative**: Shared reward
- **Competitive**: Zero-sum
- **Mixed**: General-sum

### MADDPG

Centralized training, decentralized execution. Each agent has access to all observations during training.

### Independent PPO (IPPO)

Each agent runs PPO independently, treating others as part of the environment.

```python
# Simple multi-agent loop
obs_n = env.reset()
while not done:
    act_n = [agent[i].select_action(obs_n[i]) for i in range(n_agents)]
    next_obs_n, rew_n, done_n, _ = env.step(act_n)
    for i in range(n_agents):
        agent[i].store_transition(obs_n[i], act_n[i], rew_n[i], next_obs_n[i], done_n[i])
    obs_n = next_obs_n
```

---

## Practical Examples

### Example 1: SAC on Custom Environment

```python
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

env = gym.make("Pendulum-v1")
eval_env = gym.make("Pendulum-v1")

model = SAC("MlpPolicy", env, verbose=1)
eval_callback = EvalCallback(eval_env, best_model_save_path="./sac_pendulum", eval_freq=5000)
model.learn(total_timesteps=100_000, callback=eval_callback)
```

### Example 2: Offline RL with d4rl

```python
import d4rl
import gym
from stable_baselines3 import CQL

env = gym.make("halfcheetah-medium-v2")
# CQL needs offline dataset - use custom replay buffer from d4rl
model = CQL("MlpPolicy", env, verbose=1)
# Load dataset into buffer, train
```

### Example 3: Curiosity with PPO

```python
# Wrap env with curiosity reward
class CuriosityWrapper(gym.Wrapper):
    def __init__(self, env, rnd, scale=0.01):
        super().__init__(env)
        self.rnd = rnd
        self.scale = scale
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        intr = self.rnd.intrinsic_reward(obs)
        rew = rew + self.scale * intr
        return obs, rew, done, info
```

---

## Best Practices

1. **SAC**: Default choice for continuous control; tune `ent_coef`
2. **TD3**: When SAC is unstable; use `policy_delay`
3. **Offline RL**: Validate on in-distribution; be conservative
4. **Model-based**: Short horizons to reduce model error
5. **Replay buffer**: Large buffer for off-policy
6. **Normalization**: Obs/action normalization helps

---

## Summary

| Topic | Key Point |
|-------|-----------|
| SAC | Max entropy, off-policy, robust |
| TD3 | Twin Q, delayed policy, smoothing |
| Model-based | Learn dynamics, plan or generate data |
| Offline RL | No interaction; CQL, IQL |
| MuZero | Implicit world model, MCTS in latent space |
| Curiosity | RND, ICM for exploration |

**Libraries**: `stable-baselines3`, `d4rl`, `gymnasium`
