import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_cumulative_rewards(results_file):
    results = load_results(results_file)
    episodes = len(results['linear_q'][0]['episode_rewards'])
    
    plt.figure(figsize=(10, 6))
    
    # Plot cumulative rewards for each method
    for name, label, color in [('linear_q', 'Linear Q', 'blue'), 
                                ('dqn', 'DQN', 'green'), 
                                ('double_dqn', 'Double DQN', 'red')]:
        rewards = np.array([run['episode_rewards'] for run in results[name]])
        cumsum = np.cumsum(rewards, axis=1)  # Shape: (seeds, episodes)
        mean = cumsum.mean(axis=0)
        std = cumsum.std(axis=0)
        
        plt.plot(mean, label=label, color=color, linewidth=2)
        plt.fill_between(range(episodes), mean-std, mean+std, alpha=0.2, color=color)
    
    plt.title('Cumulative Sum of Rewards vs Episodes', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Reward (Sum of All Episode Rewards)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = results_file.replace('.json', '_cumulative.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cumulative reward plot saved to: {plot_path}")
    
    # Show final cumulative reward statistics
    print("\n" + "="*65)
    print("💰 FINAL CUMULATIVE REWARD (After 500 Episodes)")
    print("="*65)
    for name, label in [('linear_q', 'Linear Q'), ('dqn', 'DQN'), ('double_dqn', 'Double DQN')]:
        rewards = np.array([run['episode_rewards'] for run in results[name]])
        cumsum = np.cumsum(rewards, axis=1)
        final_cumsum = cumsum[:, -1]  # Last episode cumulative sum
        mean_final = final_cumsum.mean()
        std_final = final_cumsum.std()
        print(f"{label:12s} | Total Reward: {mean_final:8.1f} ± {std_final:5.1f} | Seeds: {len(final_cumsum)}")
    
    plt.show()

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
    print(f"📈 Generating cumulative reward plot from: {filepath}\n")
    plot_cumulative_rewards(filepath)