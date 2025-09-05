# strategies/train_qlearning.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from simulator.spectrum_env import SpectrumEnv

# Create environment (Normal scenario)
env = SpectrumEnv(scenario="normal")

# Initialize Q-learning (DQN) agent
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0005,  # Slightly lower for stability
    buffer_size=20000,     # Larger buffer for more experience
    learning_starts=500,   # Start learning after more steps
    batch_size=64,         # Larger batch for better updates
    tau=0.1,               # Softer target updates
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.2,  # Explore longer
    exploration_final_eps=0.01,  # Lower final exploration rate
    verbose=1
)

# Train for 50,000 steps
model.learn(total_timesteps=50000, log_interval=100)

# Save the model
model.save("strategies/qlearning_model")

# Evaluate the model
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}")

# Test the trained model
env = SpectrumEnv(scenario="normal")
state, _ = env.reset()
print("Initial State:", state)
env.render()
total_reward = 0.0
step_count = 0
while True:
    action, _ = model.predict(state, deterministic=True)
    state, reward, terminated, truncated, info = env.step(action)
    channel_id = action // 10
    user_id = action % 10
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