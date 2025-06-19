# lunar_DQN.py
# -------------------------------------------------------------
# Discrete Lunar Lander solved with Double-DQN
# -------------------------------------------------------------
# pip install "gymnasium[box2d]" torch numpy tensorboard
# -------------------------------------------------------------

import random
import time
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from tqdm import trange
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os


# ──────────────────────────────────────────────────────────────
# Hyper-params
# ──────────────────────────────────────────────────────────────
class CFG:
    seed             = 42                         # RNG seed for reproducibility
    env_id           = "LunarLander-v3"           # Gymnasium environment ID
    device           = "cuda" if torch.cuda.is_available() else "cpu"  # run on GPU if available

    buffer_size      = 100_000   # max transitions stored in replay buffer
    batch_size       = 256       # SGD mini-batch size
    gamma            = 0.98      # discount factor (future-reward weight)
    lr               = 5e-4      # Adam learning rate for online network
    target_tau       = 5e-3      # soft-update rate for target network
    warmup_steps     = 10_000     # steps of pure exploration before first SGD update
    total_steps      = 500_000   # hard training budget (env steps)
    eval_every       = 10_000    # evaluation cadence (env steps)

    eps_start        = 1.0       # initial epsilon for epsilon-greedy exploration
    eps_end          = 0.01      # final epsilon after decay
    eps_decay_steps  = 500_000   # steps over which ε decays linearly

    solve_score      = 200       # threshold mean return that counts as "solved". should be 200
    earlystop_count  = 5
    log_dir          = Path("runs/lunar_dqn")  # TensorBoard & checkpoint directory


cfg = CFG()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

# ──────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, size, obs_dim):
        self.obs  = np.empty((size, obs_dim), dtype=np.float32)
        self.next = np.empty((size, obs_dim), dtype=np.float32)
        self.act  = np.empty(size,              dtype=np.int64)
        self.rew  = np.empty(size,              dtype=np.float32)
        self.done = np.empty(size,              dtype=np.float32)
        self.ptr = self.size = 0; self.max = size

    def add(self, s, a, r, d, s2):
        self.obs [self.ptr] = s
        self.next[self.ptr] = s2
        self.act [self.ptr] = a
        self.rew [self.ptr] = r
        self.done[self.ptr] = d
        self.ptr  = (self.ptr + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch):
        idx = np.random.randint(0, self.size, size=batch)
        return ( self.obs[idx], self.act[idx], self.rew[idx],
                 self.done[idx], self.next[idx] )

# ──────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),     
            nn.Linear(64,  64), nn.ReLU(),   
            nn.Linear(64,  64), nn.ReLU(),         
            nn.Linear(64,  n_actions)
        )
        
    def forward(self, x): return self.net(x)

# ──────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────
class DQNAgent_lander:
    def __init__(self, obs_dim, n_actions):
        self.n_actions = n_actions
        self.online = DQN(obs_dim, n_actions).to(cfg.device)
        self.target = DQN(obs_dim, n_actions).to(cfg.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.step_ct = 0

    def act(self, s, eps):
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)
        s_t = torch.tensor(s, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online(s_t)
        return int(torch.argmax(q, 1).item())

    def update(self, buf: ReplayBuffer):
        if buf.size < cfg.warmup_steps: return
        s, a, r, d, s2 = buf.sample(cfg.batch_size)

        s   = torch.tensor(s,  device=cfg.device)
        a   = torch.tensor(a,  device=cfg.device).unsqueeze(1)
        r   = torch.tensor(r,  device=cfg.device)
        d   = torch.tensor(d,  device=cfg.device)
        s2  = torch.tensor(s2, device=cfg.device)

        q  = self.online(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            a2      = torch.argmax(self.online(s2), 1, keepdim=True)
            q_next  = self.target(s2).gather(1, a2).squeeze(1)
            target  = r + cfg.gamma * (1 - d) * q_next

        loss = F.mse_loss(q, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        # soft update
        with torch.no_grad():
            for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
                p_t.mul_(1 - cfg.target_tau).add_(cfg.target_tau * p_o)

# ──────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────
def evaluate_policy_lander(agent, env, episodes=40):
    scores = []
    for _ in range(episodes):
        s, _ = env.reset(seed=None)
        done = truncated = False
        ep = 0
        while not (done or truncated):
            a = agent.act(s, eps=0.0)
            s, r, done, truncated, _ = env.step(a)
            ep += r
        scores.append(ep)
    return np.mean(scores)

# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────
def train():
    env       = gym.make(cfg.env_id)
    eval_env  = gym.make(cfg.env_id)

    agent = DQNAgent_lander(env.observation_space.shape[0], env.action_space.n)
    buf   = ReplayBuffer(cfg.buffer_size, env.observation_space.shape[0])
    writer = SummaryWriter(cfg.log_dir / time.strftime("%Y%m%d-%H%M%S"))

    s,_ = env.reset(seed=cfg.seed)
    eps = cfg.eps_start
    consec_good = 0
    eval_steps = []
    eval_returns = []


    for step in trange(cfg.total_steps, desc="env steps"):
        a   = agent.act(s, eps)
        s2, r, done, truncated, _ = env.step(a)
        buf.add(s, a, r, float(done), s2)
        s = s2

        agent.update(buf)

        eps = max(cfg.eps_end,
                  cfg.eps_start - (cfg.eps_start-cfg.eps_end)*step/cfg.eps_decay_steps)

        if done or truncated:
            s,_ = env.reset()

        if (step+1) % cfg.eval_every == 0:
            m = evaluate_policy_lander(agent, eval_env)
            writer.add_scalar("eval/return", m, step+1)
            print(f"\n[eval] step {step+1} | mean_return = {m:.1f}")

            eval_steps.append(step + 1)        
            eval_returns.append(m)             

            consec_good = consec_good + 1 if m >= cfg.solve_score else 0
            if consec_good >= cfg.earlystop_count:
                print("Solved! Early stopping."); break

    torch.save(agent.online.state_dict(), "lunar_dqn.pt")

        # ─── save + plot learning curve ──────────────────────────────────
    os.makedirs("plots", exist_ok=True)

    np.savez("lunar_training_curve.npz",
             steps=np.array(eval_steps),
             returns=np.array(eval_returns))

    plt.figure(figsize=(6,4))
    plt.plot(eval_steps, eval_returns, marker='o', label="Mean Return")
    plt.axhline(cfg.solve_score, color='red', linestyle='--', linewidth=1.2,
            label=f"Solved threshold ({cfg.solve_score})")
    plt.xlabel("Training step")
    plt.ylabel("Mean return (50 eps)")
    plt.title("Lunar-Lander DQN learning curve")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/lunar_learning_curve.png", dpi=300)
    # plt.show()

    env.close(); eval_env.close(); writer.close()

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
