# Advanced Reinforcement Learning: Complete Guide

## Table of Contents
1. [Introduction to Advanced RL](#introduction-to-advanced-rl)
2. [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
3. [Soft Actor-Critic (SAC)](#soft-actor-critic-sac)
4. [Offline Reinforcement Learning](#offline-reinforcement-learning)
5. [Practical Examples](#practical-examples)
6. [Common Pitfalls and Troubleshooting](#common-pitfalls-and-troubleshooting)
7. [Production Considerations](#production-considerations)
8. [Best Practices](#best-practices)
9. [References](#references)

---

## Introduction to Advanced RL

**Advanced reinforcement learning** extends foundational RL (Q-learning, policy gradients) with algorithms that achieve stronger sample efficiency, stability, and applicability to real-world settings. Key advances include **on-policy clipping** (PPO), **maximum-entropy** objectives (SAC), and **offline learning from fixed datasets** without environment interaction.

### When to Use Which Algorithm

| Algorithm | Setting | Strengths | Typical Use |
|-----------|---------|-----------|-------------|
| **PPO** | Online, continuous/discrete | Stable, easy to tune | Robotics, games, general RL |
| **SAC** | Online, continuous | Sample efficient, robust | Robotics, continuous control |
| **Offline RL** | Batch data only | No sim required | Historical logs, healthcare |
| **DQN** | Online, discrete | Mature, well-studied | Atari, discrete control |

### Conceptual Foundations

- **On-policy vs off-policy**: On-policy (PPO) uses data from current policy; off-policy (SAC, DQN) reuses past experience, improving sample efficiency.
- **Actor-critic**: Actor (policy) selects actions; critic (value function) estimates returns, reducing variance in policy gradients.
- **Entropy regularization**: Encourages exploration; SAC maximizes entropy explicitly for robustness.

---

## Proximal Policy Optimization (PPO)

**PPO** (Schulman et al., 2017) is an on-policy algorithm that limits policy updates to prevent destructive changes. It clips the probability ratio to keep updates stable.

### Core Idea

The policy gradient can have high variance; large updates can collapse performance. PPO constrains the **policy ratio** \( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \) so updates stay within a trust region.

### PPO-Clip Objective

\[
L^{CLIP} = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
\]

- \(\hat{A}_t\): Advantage estimate (GAE)
- \(\epsilon\): Clip range (typically 0.2)
- If advantage positive: don't increase probability beyond \(1+\epsilon\)
- If advantage negative: don't decrease beyond \(1-\epsilon\)

### Full PPO Loss

\[
L = L^{CLIP} - c_1 L^{VF} + c_2 S[\pi_\theta]
\]

- \(L^{VF}\): Value function loss (MSE)
- \(S[\pi_\theta]\): Entropy bonus (exploration)

### Implementation with Comments

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    """Shared backbone with separate policy and value heads."""
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        shared = self.shared(obs)
        # Gaussian policy for continuous actions
        mean = self.actor_mean(shared)
        std = torch.exp(self.actor_logstd).expand_as(mean)
        return mean, std, self.critic(shared).squeeze(-1)

    def get_action_and_log_prob(self, obs, deterministic=False):
        mean, std, _ = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, obs, actions):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation.
    Balances bias (short horizon) and variance (long horizon).
    """
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * gae * (1 - dones[t])
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, obs, actions, old_log_probs, advantages, returns, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):
    """
    Single PPO update epoch. Typically run multiple epochs per batch.
    """
    log_prob, entropy, value = model.evaluate_actions(obs, actions)
    ratio = torch.exp(log_prob - old_log_probs)

    # Clipped surrogate
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    actor_loss = -torch.min(surr1, surr2).mean()

    critic_loss = F.mse_loss(value, returns)
    entropy_loss = -entropy.mean()

    loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return loss.item()
```

### PPO Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Clip \(\epsilon\) | 0.2 | Smaller = more conservative |
| GAE \(\lambda\) | 0.95 | Higher = more bias, less variance |
| Epochs per batch | 3–10 | More epochs risk overfitting to batch |
| Batch size | 64–4096 | Larger = more stable, slower |

---

## Soft Actor-Critic (SAC)

**SAC** (Haarnoja et al., 2018) is an off-policy, maximum-entropy algorithm for continuous control. It learns a policy that maximizes expected return **and** entropy (exploration).

### Core Idea

Maximum-entropy RL augments the reward: \( r + \alpha \mathcal{H}(\pi(\cdot|s)) \). Higher entropy encourages diverse behavior and robustness. SAC uses:
- **Actor**: Stochastic policy (reparameterization trick)
- **Critic**: Two Q-networks + target networks (reduce overestimation)
- **Auto \(\alpha\)**: Learn temperature to balance return vs entropy

### SAC Objective

- **Critic**: Minimize TD error for \(Q(s,a)\)
- **Actor**: Maximize \(Q(s,\pi(s)) - \alpha \log \pi(a|s)\)
- **Alpha**: Updated to maintain target entropy

### Implementation with Comments

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class SquashedGaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian policy for bounded actions."""
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs, deterministic=False):
        h = self.net(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-20, 2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()  # Reparameterization
        # Tanh squashing
        tanh_action = torch.tanh(action)
        log_prob = (dist.log_prob(action).sum(dim=-1) -
                    torch.log(1 - tanh_action.pow(2) + 1e-6).sum(dim=-1))
        return tanh_action, log_prob


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class SAC:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, auto_alpha=True, target_entropy=None):
        self.actor = SquashedGaussianPolicy(obs_dim, act_dim)
        self.critic1 = QNetwork(obs_dim, act_dim)
        self.critic2 = QNetwork(obs_dim, act_dim)
        self.critic1_tgt = copy.deepcopy(self.critic1)
        self.critic2_tgt = copy.deepcopy(self.critic2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        target_entropy = target_entropy or -act_dim  # Heuristic: -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = target_entropy

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor(obs.unsqueeze(0), deterministic=deterministic)
            return a.squeeze(0)

    def update(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        # Update critics
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_obs)
            q1_next = self.critic1_tgt(next_obs, next_actions)
            q2_next = self.critic2_tgt(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            alpha = self.log_alpha.exp()
            target_q = rewards + self.gamma * (1 - dones) * (min_q_next - alpha * next_log_prob)
        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Update actor
        new_actions, log_prob = self.actor(obs)
        q1 = self.critic1(obs, new_actions)
        q2 = self.critic2(obs, new_actions)
        min_q = torch.min(q1, q2)
        actor_loss = (self.log_alpha.exp() * log_prob - min_q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update alpha (optional)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # Soft update targets
        for p, pt in zip(self.critic1.parameters(), self.critic1_tgt.parameters()):
            pt.data.copy_(self.tau * p + (1 - self.tau) * pt)
        for p, pt in zip(self.critic2.parameters(), self.critic2_tgt.parameters()):
            pt.data.copy_(self.tau * p + (1 - self.tau) * pt)
```

---

## Offline Reinforcement Learning

**Offline RL** learns from a fixed dataset of transitions \((s, a, r, s')\) **without** interacting with the environment. Critical when:
- Real-world interaction is costly or dangerous (robotics, healthcare)
- Data comes from historical logs (recommendations, ad bidding)

### Key Challenge: Distribution Shift

The behavior policy \(\beta\) that collected the data may differ from the learned policy \(\pi\). Evaluating or executing \(\pi\) on out-of-distribution actions can lead to **extrapolation error**: Q-values for unseen \((s,a)\) are overestimated.

### Approaches

| Method | Idea | Typical Use |
|--------|------|-------------|
| **Conservative Q-Learning (CQL)** | Penalize Q-values for OOD actions | General offline RL |
| **Implicit Q-Learning (IQL)** | Avoid explicit max over actions | Stable, simple |
| **Batch-Constrained Q-learning (BCQ)** | Constrain policy close to behavior | Narrow data |
| **TD3+BC** | Add behavior cloning term to TD3 | Simple baseline |

### CQL: Conceptual Overview

CQL minimizes Q-values on **sampled** actions (from replay) and maximizes on **dataset** actions. The learning objective includes:
\[
\min_Q \alpha \cdot \left( \mathbb{E}_{s \sim D}[\log \sum_a \exp(Q(s,a))] - \mathbb{E}_{(s,a) \sim D}[Q(s,a)] \right) + \text{TD loss}
\]

This keeps Q-values **conservative** for actions not well represented in the dataset.

### Simple Offline RL: BC + TD3 (TD3+BC)

```python
def td3_bc_loss(actor, critic, critic_tgt, batch, alpha=2.5):
    """
    TD3+BC: Combine TD3 critic loss with behavior cloning regularization.
    alpha controls how much to trust the dataset policy.
    """
    obs, actions, rewards, next_obs, dones = batch

    # TD target
    with torch.no_grad():
        next_actions = actor(next_obs)  # Deterministic
        target_q = critic_tgt(next_obs, next_actions)
        target_q = rewards + 0.99 * (1 - dones) * target_q

    current_q = critic(obs, actions)
    td_loss = F.mse_loss(current_q, target_q)

    # Behavior cloning: actor should mimic dataset actions
    pred_actions = actor(obs)
    bc_loss = F.mse_loss(pred_actions, actions)

    # Combined: scale BC by 1/mean(|Q|) to balance
    scale = current_q.abs().mean().detach()
    actor_loss = -critic(obs, pred_actions).mean() + alpha * bc_loss / scale

    return td_loss, actor_loss
```

### Practical Offline RL Tips

1. **Data quality matters**: Noisy or biased data yields biased policies.
2. **Stay close to behavior**: Avoid large policy shifts.
3. **Evaluate carefully**: Use logged data for OPE (Off-Policy Evaluation) when possible.
4. **Avoid extrapolation**: Prefer methods that constrain the policy (BCQ, CQL).

---

## Practical Examples

### Example 1: PPO with Gymnasium (Continuous Control)

```python
import gymnasium as gym
import torch

def collect_trajectories(env, model, steps_per_epoch=2048):
    """Collect rollout for PPO update."""
    obs, _ = env.reset()
    obs_buf, act_buf, rew_buf, logp_buf, val_buf, done_buf = [], [], [], [], [], []

    for _ in range(steps_per_epoch):
        with torch.no_grad():
            act, logp = model.get_action_and_log_prob(torch.FloatTensor(obs).unsqueeze(0))
            _, _, val = model.evaluate_actions(
                torch.FloatTensor(obs).unsqueeze(0), act
            )
        act = act.squeeze(0).numpy()
        next_obs, rew, term, trunc, _ = env.step(act)
        done = term or trunc

        obs_buf.append(obs)
        act_buf.append(act)
        rew_buf.append(rew)
        logp_buf.append(logp.item())
        val_buf.append(val.item())
        done_buf.append(done)
        obs = next_obs
        if done:
            obs, _ = env.reset()

    return (
        torch.FloatTensor(np.array(obs_buf)),
        torch.FloatTensor(np.array(act_buf)),
        torch.FloatTensor(rew_buf),
        torch.FloatTensor(logp_buf),
        torch.FloatTensor(val_buf),
        torch.FloatTensor(done_buf),
    )

# Training loop
env = gym.make("Pendulum-v1")
model = ActorCritic(obs_dim=3, act_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
for epoch in range(500):
    obs, actions, rewards, old_log_probs, values, dones = collect_trajectories(env, model)
    next_value = model(torch.FloatTensor(obs[-1]).unsqueeze(0))[2].item()
    advantages, returns = compute_gae(rewards, values, dones, next_value)
    for _ in range(5):
        ppo_update(model, optimizer, obs, actions, old_log_probs, advantages, returns)
```

### Example 2: SAC with Replay Buffer

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=1e6):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)

# SAC training
sac = SAC(obs_dim=3, act_dim=1)
replay = ReplayBuffer(capacity=100000)
obs, _ = env.reset()
for step in range(100000):
    action = sac.select_action(torch.FloatTensor(obs), deterministic=False).numpy()
    next_obs, reward, term, trunc, _ = env.step(action)
    replay.push(obs, action, reward, next_obs, term or trunc)
    obs = next_obs
    if term or trunc:
        obs, _ = env.reset()

    if len(replay) > 1000:
        batch = replay.sample(256)
        sac.update(batch)
```

---

## Common Pitfalls and Troubleshooting

### 1. PPO: Collapsing / NaNs

**Symptom**: Policy collapses (repeated actions) or NaNs in loss.

**Causes**: Learning rate too high; advantage scale explosion; bad initialization.

**Solutions**:
- Normalize advantages: `(A - mean) / (std + 1e-8)`
- Clip gradients: `nn.utils.clip_grad_norm_(model.parameters(), 0.5)`
- Lower LR (e.g., 3e-4)
- Check for division by zero in log_prob

### 2. SAC: Actor Outputs Saturate at ±1

**Symptom**: Tanh outputs always -1 or +1.

**Causes**: Alpha too high; Q-values dominant.

**Solutions**: Tune target entropy; reduce alpha; add gradient clipping to actor.

### 3. Offline RL: Policy Worse Than Behavior

**Symptom**: Learned policy underperforms the data-collection policy.

**Causes**: Extrapolation error; insufficient data coverage.

**Solutions**: Use conservative methods (CQL); increase BC weight; ensure diverse data.

### 4. Unstable Value Estimates

**Symptom**: Value predictions explode or oscillate.

**Solutions**: Use target networks; reduce LR; increase batch size; use value clipping in PPO.

---

## Production Considerations

### Simulation vs Real World

- **Sim-to-real gap**: Policies trained in sim may fail in real environments—use domain randomization, system identification.
- **Safety**: Add constraints (e.g., CBF), human override, redundant sensors.

### Inference

- **Deterministic policies**: Use `deterministic=True` at deployment to avoid sampling variance.
- **Latency**: SAC/PPO inference is cheap (single forward pass); bottleneck is often environment step.

### Monitoring

- **Value and return trends**: Track mean episode return, value predictions.
- **Exploration**: Monitor entropy; low entropy may indicate collapse.
- **Data distribution**: In offline RL, track how far policy deviates from behavior.

---

## Best Practices

1. **Start simple**: CartPole / Pendulum before complex envs.
2. **Tune sparingly**: PPO and SAC have sensible defaults; change one hyperparameter at a time.
3. **Use GAE**: Almost always improves PPO.
4. **Replay buffer size**: 1e6 for SAC; larger can help for long-horizon tasks.
5. **Seed everything**: For reproducibility (env, torch, numpy).

---

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) – Schulman et al., 2017
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) – Haarnoja et al., 2018
- [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779) – Kumar et al., 2020
- [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860) – TD3+BC, Fujimoto & Gu, 2021
- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives](https://arxiv.org/abs/2005.01643) – Levine et al., 2020
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) – PPO, SAC implementations
- [OpenAI Spinning Up](https://spinningup.openai.com/) – Educational RL material
