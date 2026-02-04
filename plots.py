import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_comparison(results_file):
    results = load_results(results_file)
    seeds = len(results['linear_q'])
    episodes = len(results['linear_q'][0]['episode_rewards'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Episode Rewards
    for name, label, color in [('linear_q', 'Linear Q', 'blue'), 
                                ('dqn', 'DQN', 'green'), 
                                ('double_dqn', 'Double DQN', 'red')]:
        rewards = np.array([run['episode_rewards'] for run in results[name]])
        mean = rewards.mean(axis=0)
        std = rewards.std(axis=0)
        axes[0, 0].plot(mean, label=label, color=color)
        axes[0, 0].fill_between(range(episodes), mean-std, mean+std, alpha=0.2, color=color)
    axes[0, 0].set_title('Episode Reward vs Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 550)
    
    # Plot 2: Moving Average Reward (window=20) - FIXED INDEXING
    window = 20
    for name, label, color in [('linear_q', 'Linear Q', 'blue'), 
                                ('dqn', 'DQN', 'green'), 
                                ('double_dqn', 'Double DQN', 'red')]:
        rewards = np.array([run['episode_rewards'] for run in results[name]])
        # Compute moving average correctly
        cumsum = np.cumsum(np.insert(rewards, 0, 0, axis=1), axis=1)
        ma = (cumsum[:, window:] - cumsum[:, :-window]) / float(window)
        mean = ma.mean(axis=0)
        std = ma.std(axis=0)
        # X-axis: starts at window-1 (index 19) to match MA length
        x_axis = range(window - 1, episodes)
        axes[0, 1].plot(x_axis, mean, label=label, color=color)
        axes[0, 1].fill_between(x_axis, mean-std, mean+std, alpha=0.2, color=color)
    axes[0, 1].set_title(f'Moving Average Reward (window={window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Avg Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 550)
    
    # Plot 3: Loss Curves (DQN & Double DQN only)
    for name, label, color in [('dqn', 'DQN', 'green'), ('double_dqn', 'Double DQN', 'red')]:
        losses = np.array([run['losses'] for run in results[name]])
        mean = losses.mean(axis=0)
        std = losses.std(axis=0)
        axes[1, 0].plot(mean, label=label, color=color)
        axes[1, 0].fill_between(range(episodes), mean-std, mean+std, alpha=0.2, color=color)
    axes[1, 0].set_title('Training Loss vs Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('MSE Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Avg Q-Value Magnitude
    for name, label, color in [('linear_q', 'Linear Q', 'blue'), 
                                ('dqn', 'DQN', 'green'), 
                                ('double_dqn', 'Double DQN', 'red')]:
        q_vals = np.array([run['avg_q_values'] for run in results[name]])
        mean = q_vals.mean(axis=0)
        std = q_vals.std(axis=0)
        axes[1, 1].plot(mean, label=label, color=color)
        axes[1, 1].fill_between(range(episodes), mean-std, mean+std, alpha=0.2, color=color)
    axes[1, 1].set_title('Avg Q-Value Magnitude vs Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Mean Q(s,a)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = results_file.replace('.json', '.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plots saved to {plot_path}")
    plt.show()
    
    # Print convergence statistics
    print("\n" + "="*60)
    print("📊 CONVERGENCE STATISTICS (CartPole-v1 | Threshold: 475 over 100 episodes)")
    print("="*60)
    for name, label in [('linear_q', 'Linear Q'), ('dqn', 'DQN'), ('double_dqn', 'Double DQN')]:
        convergence_episodes = []
        for run in results[name]:
            rewards = run['episode_rewards']
            solved = False
            for i in range(100, len(rewards)):
                if np.mean(rewards[i-100:i]) >= 475:
                    convergence_episodes.append(i)
                    solved = True
                    break
            if not solved:
                convergence_episodes.append(episodes)  # Mark as not solved
        
        mean_ep = np.mean(convergence_episodes)
        std_ep = np.std(convergence_episodes)
        solved_count = sum(1 for ep in convergence_episodes if ep < episodes)
        
        if solved_count == len(convergence_episodes):
            print(f"{label:12s} | Episodes to solve: {mean_ep:.1f} ± {std_ep:.1f}")
        else:
            print(f"{label:12s} | Solved: {solved_count}/{len(convergence_episodes)} runs | Avg ep: {mean_ep:.1f}")

if __name__ == "__main__":
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory '{results_dir}' not found. Run train.py first!")
        exit(1)
    
    files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not files:
        print("❌ No results files found. Run train.py first.")
        exit(1)
    
    latest = sorted(files)[-1]
    filepath = os.path.join(results_dir, latest)
    print(f"📈 Plotting results from: {filepath}\n")
    plot_comparison(filepath)