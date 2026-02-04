import gymnasium as gym
import numpy as np
import torch
import json
import os
from datetime import datetime

from replay_buffer import ReplayBuffer
from linear_q_learning import LinearQAgent
from dqn import DQNAgent

def set_seed(env, seed):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_agent(agent, env_name, episodes=500, seed=42, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    env = gym.make(env_name)
    set_seed(env, seed)
    
    metrics = {
        'episode_rewards': [],
        'losses': [],
        'avg_q_values': [],
        'epsilons': []
    }
    
    epsilon = epsilon_start
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        episode_q = 0
        steps = 0
        
        while True:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            loss = agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_loss += loss
            episode_q += agent.get_avg_q_value(state)
            steps += 1
            
            state = next_state
            if done:
                break
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        metrics['episode_rewards'].append(episode_reward)
        metrics['losses'].append(episode_loss / steps)
        metrics['avg_q_values'].append(episode_q / steps)
        metrics['epsilons'].append(epsilon)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-10:])
            print(f"Seed {seed} | Episode {episode+1} | Avg Reward (last 10): {avg_reward:.2f} | Epsilon: {epsilon:.3f}")
    
    env.close()
    return metrics

def run_experiment():
    ENV_NAME = "CartPole-v1"
    EPISODES = 500
    SEEDS = [42, 43, 44, 45, 46]
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = {
        'linear_q': [],
        'dqn': [],
        'double_dqn': []
    }
    
    # Linear Q-Learning
    print("\n=== Training Linear Q-Learning ===")
    for seed in SEEDS:
        env = gym.make(ENV_NAME)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()
        
        agent = LinearQAgent(state_dim, action_dim, lr=0.01, gamma=0.99)
        metrics = train_agent(agent, ENV_NAME, episodes=EPISODES, seed=seed)
        all_results['linear_q'].append(metrics)
    
    # DQN
    print("\n=== Training DQN ===")
    for seed in SEEDS:
        env = gym.make(ENV_NAME)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()
        
        agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99, 
                        buffer_size=10000, batch_size=64, target_update=10, double_dqn=False)
        metrics = train_agent(agent, ENV_NAME, episodes=EPISODES, seed=seed)
        all_results['dqn'].append(metrics)
    
    # Double DQN
    print("\n=== Training Double DQN ===")
    for seed in SEEDS:
        env = gym.make(ENV_NAME)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()
        
        agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99, 
                        buffer_size=10000, batch_size=64, target_update=10, double_dqn=True)
        metrics = train_agent(agent, ENV_NAME, episodes=EPISODES, seed=seed)
        all_results['double_dqn'].append(metrics)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RESULTS_DIR, f"cartpole_comparison_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump(all_results, f)
    print(f"\nResults saved to {filepath}")
    return filepath

if __name__ == "__main__":
    run_experiment()
