# Reinforcement Learning: Complete Guide

## Table of Contents
1. [Introduction to RL](#introduction-to-rl)
2. [RL Fundamentals](#rl-fundamentals)
3. [Markov Decision Process](#markov-decision-process)
4. [Value-Based Methods](#value-based-methods)
5. [Policy-Based Methods](#policy-based-methods)
6. [Actor-Critic Methods](#actor-critic-methods)
7. [Deep Q-Networks (DQN)](#deep-q-networks-dqn)
8. [Policy Gradients](#policy-gradients)
9. [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
10. [Practical Examples](#practical-examples)
11. [Advanced Topics](#advanced-topics)

---

## Introduction to RL

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment, receiving rewards or penalties.

### Key Components
- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **State (s)**: Current situation
- **Action (a)**: What the agent does
- **Reward (r)**: Feedback from environment
- **Policy (π)**: Strategy for selecting actions

### RL vs Other ML
- **Supervised Learning**: Learn from labeled examples
- **Unsupervised Learning**: Find patterns in data
- **Reinforcement Learning**: Learn from trial and error

### Applications
- Game playing (Chess, Go, Atari)
- Robotics
- Autonomous vehicles
- Recommendation systems
- Trading algorithms
- Resource management

---

## RL Fundamentals

### Basic Concepts

```python
import numpy as np
import gym

# Create environment
env = gym.make('CartPole-v1')

# Reset environment
state = env.reset()

# Take action
action = env.action_space.sample()  # Random action
next_state, reward, done, info = env.step(action)

print(f"State: {state}")
print(f"Action: {action}")
print(f"Reward: {reward}")
print(f"Done: {done}")
```

### Exploration vs Exploitation

```python
# Epsilon-greedy strategy
def epsilon_greedy_action(q_values, epsilon):
    """Choose action using epsilon-greedy"""
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))  # Explore
    else:
        return np.argmax(q_values)  # Exploit

# Example
q_values = [0.1, 0.3, 0.2, 0.4]
epsilon = 0.1
action = epsilon_greedy_action(q_values, epsilon)
print(f"Selected action: {action}")
```

---

## Markov Decision Process

### MDP Components

```python
class MDP:
    def __init__(self, states, actions, transitions, rewards, gamma=0.9):
        self.states = states
        self.actions = actions
        self.transitions = transitions  # P(s'|s,a)
        self.rewards = rewards  # R(s,a,s')
        self.gamma = gamma  # Discount factor
    
    def get_transition_prob(self, state, action, next_state):
        """Get transition probability"""
        return self.transitions[state][action][next_state]
    
    def get_reward(self, state, action, next_state):
        """Get reward"""
        return self.rewards[state][action][next_state]
```

### Value Function

```python
def value_iteration(mdp, theta=1e-6):
    """Value iteration algorithm"""
    V = np.zeros(len(mdp.states))
    
    while True:
        V_new = np.zeros(len(mdp.states))
        
        for s in mdp.states:
            q_values = []
            for a in mdp.actions:
                q = 0
                for s_next in mdp.states:
                    prob = mdp.get_transition_prob(s, a, s_next)
                    reward = mdp.get_reward(s, a, s_next)
                    q += prob * (reward + mdp.gamma * V[s_next])
                q_values.append(q)
            V_new[s] = max(q_values)
        
        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new
    
    return V
```

---

## Value-Based Methods

### Q-Learning

```python
class QLearning:
    def __init__(self, num_states, num_actions, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((num_states, num_actions))
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-values"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        self.Q[state, action] += self.lr * (target - self.Q[state, action])
    
    def train(self, env, num_episodes=1000):
        """Train agent"""
        rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            rewards.append(total_reward)
            if episode % 100 == 0:
                print(f"Episode {episode}, Avg Reward: {np.mean(rewards[-100:]):.2f}")
        
        return rewards
```

### SARSA

```python
class SARSA:
    def __init__(self, num_states, num_actions, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((num_states, num_actions))
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]
        
        self.Q[state, action] += self.lr * (target - self.Q[state, action])
```

---

## Policy-Based Methods

### Policy Gradient

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PolicyGradient:
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build policy network
        self.policy = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_dim, activation='softmax')
        ])
        
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.states = []
        self.actions = []
        self.rewards = []
    
    def select_action(self, state):
        """Sample action from policy"""
        state = tf.expand_dims(state, 0)
        probs = self.policy(state)
        action = tf.squeeze(tf.random.categorical(tf.math.log(probs), 1))
        return action.numpy()
    
    def store_transition(self, state, action, reward):
        """Store transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train(self, gamma=0.99):
        """Train policy using REINFORCE"""
        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        
        for reward in reversed(self.rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        # Normalize rewards
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # Update policy
        with tf.GradientTape() as tape:
            states = tf.convert_to_tensor(self.states, dtype=tf.float32)
            actions = tf.convert_to_tensor(self.actions, dtype=tf.int32)
            
            probs = self.policy(states)
            action_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.action_dim), axis=1)
            loss = -tf.reduce_mean(tf.math.log(action_probs + 1e-8) * discounted_rewards)
        
        gradients = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
```

---

## Actor-Critic Methods

### Actor-Critic Implementation

```python
class ActorCritic:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor network (policy)
        self.actor = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_dim, activation='softmax')
        ])
        
        # Critic network (value function)
        self.critic = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.gamma = 0.99
    
    def select_action(self, state):
        """Sample action from actor policy"""
        state = tf.expand_dims(state, 0)
        probs = self.actor(state)
        action = tf.squeeze(tf.random.categorical(tf.math.log(probs), 1))
        return action.numpy()
    
    def train(self, states, actions, rewards, next_states, dones):
        """Train actor and critic"""
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            # Critic: estimate values
            values = tf.squeeze(self.critic(states))
            next_values = tf.squeeze(self.critic(next_states))
            
            # Calculate TD target
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            td_errors = td_targets - values
            
            # Critic loss
            critic_loss = tf.reduce_mean(tf.square(td_errors))
            
            # Actor: policy gradient
            probs = self.actor(states)
            action_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.action_dim), axis=1)
            actor_loss = -tf.reduce_mean(tf.math.log(action_probs + 1e-8) * td_errors)
        
        # Update critic
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # Update actor
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        del tape
```

---

## Deep Q-Networks (DQN)

### DQN Implementation

```python
class DQN:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Main network
        self.q_network = self._build_network(learning_rate)
        
        # Target network
        self.target_network = self._build_network(learning_rate)
        self.update_target_network()
    
    def _build_network(self, learning_rate):
        """Build Q-network"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_dim)
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_dim)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = np.array([self.memory[i][0] for i in batch])
        actions = np.array([self.memory[i][1] for i in batch])
        rewards = np.array([self.memory[i][2] for i in batch])
        next_states = np.array([self.memory[i][3] for i in batch])
        dones = np.array([self.memory[i][4] for i in batch])
        
        # Current Q-values
        current_q = self.q_network.predict(states, verbose=0)
        
        # Next Q-values from target network
        next_q = self.target_network.predict(next_states, verbose=0)
        
        # Calculate target Q-values
        target_q = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Train network
        self.q_network.fit(states, target_q, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
```

---

## Proximal Policy Optimization (PPO)

### PPO Implementation

```python
class PPO:
    def __init__(self, state_dim, action_dim, learning_rate=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_ratio = 0.2
        self.gamma = 0.99
        self.lam = 0.95  # GAE lambda
        
        # Policy network
        self.policy = keras.Sequential([
            layers.Dense(128, activation='tanh', input_shape=(state_dim,)),
            layers.Dense(128, activation='tanh'),
            layers.Dense(action_dim, activation='softmax')
        ])
        
        # Value network
        self.value_net = keras.Sequential([
            layers.Dense(128, activation='tanh', input_shape=(state_dim,)),
            layers.Dense(128, activation='tanh'),
            layers.Dense(1)
        ])
        
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    def select_action(self, state):
        """Sample action from policy"""
        state = tf.expand_dims(state, 0)
        probs = self.policy(state)
        dist = tfp.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.numpy()[0], log_prob.numpy()[0]
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                last_gae = delta + self.gamma * self.lam * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def train(self, states, actions, old_log_probs, rewards, next_states, dones):
        """PPO training"""
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        
        # Compute values
        values = tf.squeeze(self.value_net(states))
        next_values = tf.squeeze(self.value_net(next_states))
        
        # Compute advantages
        advantages, returns = self.compute_gae(
            rewards.numpy(), values.numpy(), next_values.numpy(), dones.numpy()
        )
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Multiple epochs
        for _ in range(10):
            with tf.GradientTape(persistent=True) as tape:
                # Policy loss
                probs = self.policy(states)
                dist = tfp.distributions.Categorical(probs=probs)
                new_log_probs = dist.log_prob(actions)
                
                ratio = tf.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                
                # Value loss
                values_pred = tf.squeeze(self.value_net(states))
                value_loss = tf.reduce_mean(tf.square(returns - values_pred))
            
            # Update networks
            policy_grads = tape.gradient(policy_loss, self.policy.trainable_variables)
            value_grads = tape.gradient(value_loss, self.value_net.trainable_variables)
            
            self.optimizer.apply_gradients(zip(policy_grads, self.policy.trainable_variables))
            self.optimizer.apply_gradients(zip(value_grads, self.value_net.trainable_variables))
            
            del tape
```

---

## Practical Examples

### Example 1: CartPole with DQN

```python
import gym
import numpy as np

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

dqn = DQN(state_dim, action_dim)

# Training
for episode in range(500):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = dqn.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        dqn.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done:
            break
        
        dqn.replay()
    
    if episode % 50 == 0:
        dqn.update_target_network()
        print(f"Episode {episode}, Reward: {total_reward}")
```

### Example 2: Lunar Lander with PPO

```python
env = gym.make('LunarLander-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

ppo = PPO(state_dim, action_dim)

# Training loop
for episode in range(1000):
    states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []
    state = env.reset()
    
    # Collect episode data
    while True:
        action, log_prob = ppo.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        
        state = next_state
        if done:
            break
    
    # Train
    ppo.train(
        np.array(states),
        np.array(actions),
        np.array(log_probs),
        np.array(rewards),
        np.array(next_states),
        np.array(dones)
    )
```

---

## Advanced Topics

### Prioritized Experience Replay

```python
import heapq

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
    
    def add(self, experience, td_error):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return [self.buffer[i] for i in indices], indices, weights
```

### Double DQN

```python
class DoubleDQN(DQN):
    def replay(self):
        """Double DQN: use main network for action selection"""
        # ... (same as DQN but use main network for action selection)
        next_q_main = self.q_network.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_main, axis=1)
        next_q_target = self.target_network.predict(next_states, verbose=0)
        
        target_q = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * next_q_target[i][next_actions[i]]
```

---

## Best Practices

1. **Start Simple**: Use Q-learning for discrete spaces
2. **Use DQN**: For high-dimensional state spaces
3. **Try PPO**: For continuous action spaces
4. **Replay Buffer**: Essential for stable learning
5. **Target Network**: Reduces instability
6. **Epsilon Decay**: Gradually reduce exploration
7. **Reward Shaping**: Design rewards carefully
8. **Monitor Training**: Track rewards and losses

---

## Resources

- **OpenAI Gym**: Environment library
- **Stable Baselines3**: RL algorithms
- **Papers**: 
  - Q-Learning (1992)
  - DQN (2015)
  - PPO (2017)
- **Books**: Sutton & Barto RL Book

---

## Conclusion

Reinforcement Learning enables agents to learn optimal behaviors through interaction. Key takeaways:

1. **Understand MDPs**: Foundation of RL
2. **Choose Right Algorithm**: Match problem to method
3. **Handle Exploration**: Balance explore/exploit
4. **Use Deep RL**: For complex environments
5. **Monitor Training**: Track progress carefully

Remember: RL requires patience and careful hyperparameter tuning!

