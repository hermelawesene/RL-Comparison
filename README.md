# Linear vs. Non-Linear Value Function Approximation in Reinforcement Learning

## Empirical Comparison of Algorithms on CartPole-v1
- **Algorithms Compared:** Linear Q-learning, DQN, Double DQN
- **Evaluation Metrics:** Convergence speed, stability, sample efficiency with raw state features

###  Key Findings
| Metric | Linear Q | DQN | Double DQN |
| --- | --- | --- | --- |
| **Cumulative Reward (500 episodes)** | 20,561 ± 12,075 | 96,805 ± 12,941 | 99,445 ± 12,145 |
| **Sample Efficiency** | Baseline | 4.7× higher | 4.8× higher |
| **Stability** |  Catastrophic failures (40% seeds) | Volatile learning |  Smoothest progression |
| **Q-Value Bias** | N/A | Overestimation (+12%) | Reduced bias |
| **Solved Environment** | 0/5 seeds | 0/5 seeds | 1/5 seeds |

>  *Core Insight:* Non-linear approximation achieves **4.7× higher sample efficiency** by automatically learning feature interactions from raw states—eliminating the manual engineering required for linear methods to succeed.

##  Repository Structure
```
RL-Comparison/
├── train.py                 # Main training script (5 seeds × 3 algorithms)
├── plot_results.py          # Generates 4-panel comparison plots
├── plot_cumulative_rewards.py  # Cumulative reward analysis
├── linear_q_agent.py        # Linear Q-learning implementation
├── dqn_agent.py             # DQN + Double DQN implementation
├── replay_buffer.py         # Experience replay buffer
├── results/                 # Output directory (auto-created)
│   ├── cartpole_comparison_<timestamp>
```

##  Setup & Execution
### Prerequisites
```bash
# Create virtual environment (recommended)
python -m venv .venv
# Windows PowerShell
dot-source .venv\Scripts\Activate.ps1 
# Linux/MacOS
dot-source .venv/bin/activate 
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### `requirements.txt` Contents:
- gymnasium==1.0.0
- pygame==2.5.2
- torch==2.10.0
- numpy==1.26.4
-matplotlib==3.8.3 
```
### Run Experiments
defaults:
```powershell
y# Train all agents (500 episodes × 5 seeds)
python train.py
# Generate comparison plots
ypython plot_results.py 
y# Generate cumulative reward analysis
ypython plot_cumulative_rewards.py 
```
 Runtime: approximately 30–40 minutes on CPU (Intel i7, 32GB RAM)
 Output: Results saved to `results/` with timestamped filenames.
