# simulator/spectrum_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SpectrumEnv(gym.Env):
    def __init__(self, scenario="normal"):
        super().__init__()
        self.scenario = scenario
        self.channels = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        self.users = [
            {'name': f'User{i}', 'demand': 0, 'priority': 'emergency' if i < 2 else 'video' if i < 6 else 'iot'}
            for i in range(10)
        ]
        self.allocations = np.zeros(5)
        self.user_served = np.zeros(10)
        self.observation_space = spaces.Box(low=0, high=10, shape=(25,), dtype=np.float32)
        self.action_space = spaces.Discrete(50)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.channels = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        self.allocations = np.zeros(5)
        self.user_served = np.zeros(10)
        if self.scenario == "normal":
            demands = np.random.randint(1, 6, size=10)
        elif self.scenario == "congested":
            demands = np.random.randint(3, 8, size=10)
        elif self.scenario == "emergency":
            demands = np.random.randint(1, 6, size=10)
            demands[0:2] = np.random.randint(4, 7, size=2)
        for i, user in enumerate(self.users):
            user['demand'] = demands[i]
        state = np.concatenate([
            self.channels,
            [user['demand'] for user in self.users],
            self.user_served
        ])
        return state, {}

    def step(self, action):
        channel_id = action // 10
        user_id = action % 10
        reward = 0.0
        if self.allocations[channel_id] == 0 and self.user_served[user_id] == 0:
            demand = self.users[user_id]['demand']
            if self.channels[channel_id] >= demand:
                self.allocations[channel_id] = user_id + 1
                self.channels[channel_id] -= demand
                self.user_served[user_id] = 1
                reward += 20.0  # Increased reward for valid allocation
                if self.users[user_id]['priority'] == 'emergency':
                    reward += 10.0  # Higher reward for emergency
                elif self.users[user_id]['priority'] == 'video':
                    reward += 5.0
            else:
                reward -= 2.0  # Reduced penalty for insufficient bandwidth
        else:
            reward -= 2.0  # Reduced penalty for invalid action

        served_demands = [self.users[i]['demand'] * self.user_served[i] for i in range(10)]
        served_sum = np.sum(served_demands)
        served_sum_sq = np.sum([d ** 2 for d in served_demands])
        fairness = (served_sum ** 2) / (10 * served_sum_sq) if served_sum_sq > 0 else 0
        reward += 10.0 * fairness  # Increased fairness weight

        terminated = all(self.allocations != 0) or all(self.user_served == 1)
        truncated = False
        state = np.concatenate([
            self.channels,
            [user['demand'] for user in self.users],
            self.user_served
        ])
        info = {
            'fairness': fairness,
            'scenario': self.scenario,
            'utilization': 1 - np.sum(self.channels) / 50.0,
            'emergency_satisfaction': np.sum(self.user_served[:2]) / 2.0
        }
        return state, reward, terminated, truncated, info

    def render(self):
        print(f"Scenario: {self.scenario}")
        print("Channels (MHz):", self.channels)
        print("Allocations:", self.allocations)
        print("Users Served:", self.user_served)
        print("Users:")
        for user in self.users:
            print(f"  {user['name']}: Demand={user['demand']} MHz, Priority={user['priority']}")

if __name__ == "__main__":
    for scenario in ["normal", "congested", "emergency"]:
        print(f"\nTesting {scenario} scenario with random strategy:")
        env = SpectrumEnv(scenario=scenario)
        state, _ = env.reset()
        print("Initial State:", state)
        env.render()
        total_reward = 0.0
        step_count = 0
        while True:
            available_channels = [i for i, alloc in enumerate(env.allocations) if alloc == 0]
            available_users = [i for i, served in enumerate(env.user_served) if served == 0]
            if not available_channels or not available_users:
                break
            channel_id = np.random.choice(available_channels)
            user_id = np.random.choice(available_users)
            action = channel_id * 10 + user_id
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            print(f"Step {step_count}: Action=[{channel_id}, {user_id}], Reward={reward}, Fairness={info['fairness']:.2f}")
            if terminated or truncated:
                break
        print("Final State:", state)
        print(f"Total Reward: {total_reward}")
        print(f"Utilization: {info['utilization']*100:.1f}%")
        print(f"Emergency Satisfaction: {info['emergency_satisfaction']*100:.1f}%")
        env.render()