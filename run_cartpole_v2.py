"""
cartpole_watch.py
Run a saved tabular CartPole controller and render a few episodes.
"""

import time
import gymnasium as gym
import numpy as np
from pathlib import Path

# ------- 1. load the Q-table ---------------------------------------------------
Q_PATH = Path("cartpole_qtable.npy")      # same folder as training script
if not Q_PATH.exists():
    raise FileNotFoundError(f"{Q_PATH} not found: run training first.")
Q = np.load(Q_PATH)
print(f"Loaded Q-table with shape {Q.shape}")

# ------- 2. define the same discretiser (bins must match training) -------------
bins = [
    np.linspace(-2.4,   2.4,   3 - 1),   # x
    np.linspace(-3.0,   3.0,   3 - 1),   # ẋ
    np.linspace(-0.418, 0.418, 10 - 1),  # θ
    np.linspace(-4.0,   4.0,  10 - 1)    # θ̇
]

def discretise(obs: np.ndarray) -> int:
    i0, i1, i2, i3 = [np.digitize(o, b) for o, b in zip(obs, bins)]
    return (((i0 * 3) + i1) * 10 + i2) * 10 + i3

# ------- 3. make a rendering env -----------------------------------------------
env = gym.make("CartPole-v1", render_mode="human")
EPISODES = 5

for ep in range(1, EPISODES + 1):
    obs, _ = env.reset(seed=None)
    s = discretise(obs)
    done = truncated = False
    ep_ret = 0

    while not (done or truncated):
        action = int(np.argmax(Q[s]))         # greedy
        obs, reward, done, truncated, _ = env.step(action)
        ep_ret += reward
        s = discretise(obs)

        time.sleep(1 / 60)                    # slow down for better visibility
    print(f"Episode {ep}: return = {ep_ret}")

env.close()
print("Done. Close the window to exit.")
