# strategies/evaluate_all_scenarios.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from simulator.spectrum_env import SpectrumEnv

try:
    # Load the trained model
    model_path = "strategies/qlearning_model.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Run train_qlearning.py first.")
    model = DQN.load(model_path)
    print(f"Loaded model from {model_path}")

    # Evaluate and test on all scenarios
    for scenario in ["normal", "congested", "emergency"]:
        print(f"\nEvaluating Q-learning agent on {scenario} scenario:")
        env = SpectrumEnv(scenario=scenario)
        
        # Evaluate the model
        try:
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
            print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        except Exception as e:
            print(f"Evaluation error in {scenario} scenario: {e}")

        # Test the model
        try:
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
                print(f"Step {step_count}: Action=[{channel_id}, {user_id}], Reward={reward:.2f}, Fairness={info['fairness']:.2f}")
                if terminated or truncated:
                    break
            print("Final State:", state)
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Utilization: {info['utilization']*100:.1f}%")
            print(f"Emergency Satisfaction: {info['emergency_satisfaction']*100:.1f}%")
            env.render()
        except Exception as e:
            print(f"Test error in {scenario} scenario: {e}")
except Exception as e:
    print(f"Error loading model or running evaluation: {e}")
    sys.exit(1)