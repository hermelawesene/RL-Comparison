import numpy as np

class LinearQAgent:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.weights = np.zeros((action_dim, state_dim))
    
    def get_q_values(self, state):
        return self.weights @ state
    
    def select_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        return np.argmax(self.get_q_values(state))
    
    def update(self, state, action, reward, next_state, done):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state) if not done else np.zeros(self.action_dim)
        target = reward + self.gamma * np.max(next_q_values) * (1 - done)
        td_error = target - q_values[action]
        self.weights[action] += self.lr * td_error * state
        return td_error ** 2
    
    def get_avg_q_value(self, state):
        return np.mean(self.get_q_values(state))

