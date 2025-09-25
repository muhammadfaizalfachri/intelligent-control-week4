import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent

env = gym.make('CartPole-v1', render_mode="human")  # ✅ render langsung

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen (gunakan model terlatih jika tersedia)
agent = DQNAgent(state_size, action_size)
agent.epsilon = 0.01  # Minimalkan eksplorasi saat testing

for e in range(5):
    state, _ = env.reset()   # ✅ unpack (state, info)
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)  # ✅ 5 return values
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        if done:
            print(f"Test Episode: {e+1}, Score: {time}")
            break

env.close()
