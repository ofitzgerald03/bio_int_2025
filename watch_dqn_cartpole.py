# watch_dqn_cartpole.py
# ---------------------------------------------------------------
# Visualise a trained DQN agent on CartPole-v1
# ---------------------------------------------------------------
# Usage:  python watch_dqn_cartpole.py
# ---------------------------------------------------------------

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = Path("dqn_cartpole.pt")
EPISODES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1.  DQN network definition (must match the one used in training) ────────
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.ReLU(),
            nn.Linear(32, 32),      nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ── 2.  Load env and network weights ────────────────────────────────────────
env = gym.make("CartPole-v1", render_mode="human")
obs_dim   = env.observation_space.shape[0]
n_actions = env.action_space.n

policy = DQN(obs_dim, n_actions).to(DEVICE)
if not WEIGHTS.exists():
    raise FileNotFoundError(f"{WEIGHTS} not found — train the agent first.")
policy.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
policy.eval()
print(f"Loaded weights from {WEIGHTS}")

# ── 3.  Run greedy episodes ─────────────────────────────────────────────────
for ep in range(1, EPISODES + 1):
    obs, _ = env.reset(seed=None)
    done = truncated = False
    ep_ret = 0

    while not (done or truncated):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = int(torch.argmax(policy(obs_t), dim=1).item())

        obs, reward, done, truncated, _ = env.step(action)
        ep_ret += reward

        time.sleep(1 / 60)        # ~60 FPS so your eyes can follow

    print(f"Episode {ep}: return = {ep_ret}")

env.close()
print("Done. Close the render window to exit.")
