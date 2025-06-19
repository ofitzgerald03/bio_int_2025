# cartpole_dqn.py
# -------------------------------------------------------------------------
# Discrete CartPole-v1 solved with Deep Q-Network (PyTorch) in a modular way
# -------------------------------------------------------------------------
# pip install "gymnasium[classic-control]" torch numpy matplotlib tqdm tensorboard
# -------------------------------------------------------------------------

import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import matplotlib.pyplot as plt
import numpy as np
import os


# ────────────────────────────────────────────────────────────────────────────
# 1.  Hyper-parameters (tweak here or pass via argparse later)
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class CFG:
    seed:            int   = 42
    device:          str   = "cuda" if torch.cuda.is_available() else "cpu"
    # env
    env_id:          str   = "CartPole-v1"
    # replay / training
    buffer_size:     int   = 50_000
    batch_size:      int   = 64
    gamma:           float = 0.98
    lr:              float = 1e-3
    target_sync_tau: float = 1e-2        # soft update
    warmup_steps:    int   = 1_000
    total_steps:     int   = 150_000
    eval_every:      int   = 5_000
    # exploration
    eps_start:       float = 1.0
    eps_end:         float = 0.02
    eps_decay_steps: int   = 50_000
    # early-stopping
    early_patience:  int   = 3           # consecutive evals
    solve_score:     float = 475.0
    log_dir:         Path  = Path("runs/dqn_cartpole")

cfg = CFG()

# reproducibility
torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

# ────────────────────────────────────────────────────────────────────────────
# 2.  Replay buffer
# ────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_buf      = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.empty((capacity, obs_dim), dtype=np.float32)   # NEW
        self.act_buf  = np.empty((capacity,),       dtype=np.int64)
        self.rew_buf  = np.empty((capacity,),       dtype=np.float32)
        self.done_buf = np.empty((capacity,),       dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, obs, act, rew, done, next_obs):      # accept 5 fields
        self.obs_buf[self.ptr]      = obs
        self.next_obs_buf[self.ptr] = next_obs          # NEW
        self.act_buf[self.ptr]      = act
        self.rew_buf[self.ptr]      = rew
        self.done_buf[self.ptr]     = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return ( self.obs_buf[idx],
                 self.act_buf[idx],
                 self.rew_buf[idx],
                 self.done_buf[idx],
                 self.next_obs_buf[idx] )               # use stored next_obs


# ────────────────────────────────────────────────────────────────────────────
# 3.  DQN network
# ────────────────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.ReLU(),
            nn.Linear(32, 32),      nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ────────────────────────────────────────────────────────────────────────────
# 4.  Agent
# ────────────────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, obs_dim, n_actions):
        self.n_actions = n_actions
        self.online  = DQN(obs_dim, n_actions).to(cfg.device)
        self.target  = DQN(obs_dim, n_actions).to(cfg.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt     = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.step_ct = 0

    def act(self, obs: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.online(obs_t)
        return int(torch.argmax(q_vals, dim=1).item())

    def update(self, buffer: ReplayBuffer):
        if buffer.size < cfg.warmup_steps:
            return
        obs, act, rew, done, next_obs = buffer.sample(cfg.batch_size)

        obs_t      = torch.tensor(obs,       device=cfg.device)
        act_t      = torch.tensor(act,       device=cfg.device)
        rew_t      = torch.tensor(rew,       device=cfg.device)
        done_t     = torch.tensor(done,      device=cfg.device)
        next_obs_t = torch.tensor(next_obs,  device=cfg.device)

        q_pred = self.online(obs_t).gather(1, act_t.view(-1,1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(next_obs_t).max(1)[0]
            target = rew_t + cfg.gamma * (1.0 - done_t) * q_next

        loss = F.mse_loss(q_pred, target)

        self.opt.zero_grad(); loss.backward(); self.opt.step()

        # soft target update
        with torch.no_grad():
            for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
                p_t.data.mul_(1.0 - cfg.target_sync_tau).add_(cfg.target_sync_tau * p_o.data)

# ────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation helper
# ────────────────────────────────────────────────────────────────────────────
def evaluate_policy(agent: DQNAgent, env: gym.Env, episodes=20):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=None)
        done = truncated = False
        ep_ret = 0
        while not (done or truncated):
            action = agent.act(obs, eps=0.0)
            obs, r, done, truncated, _ = env.step(action)
            ep_ret += r
        returns.append(ep_ret)
    return np.mean(returns)

# ────────────────────────────────────────────────────────────────────────────
# 6.  Trainer
# ────────────────────────────────────────────────────────────────────────────
def train():
    env  = gym.make(cfg.env_id)
    eval_env = gym.make(cfg.env_id)

    eval_steps = []
    eval_returns = []     

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    buffer = ReplayBuffer(cfg.buffer_size, obs_dim)
    agent  = DQNAgent(obs_dim, n_actions)

    writer = SummaryWriter(log_dir=cfg.log_dir / time.strftime("%Y%m%d-%H%M%S"))

    obs, _ = env.reset(seed=cfg.seed)
    eps = cfg.eps_start
    consec_good = 0

    for step in trange(cfg.total_steps, desc="train-steps"):
        # select & execute action
        action = agent.act(obs, eps)
        next_obs, reward, done, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, float(done), next_obs)
        obs = next_obs

        # agent update
        agent.update(buffer)

        # ε decay
        eps = max(cfg.eps_end,
                  cfg.eps_start - (cfg.eps_start-cfg.eps_end)*step/cfg.eps_decay_steps)

        # episode end
        if done or truncated:
            obs, _ = env.reset()

        # periodic evaluation
        if (step+1) % cfg.eval_every == 0:
            mean_ret = evaluate_policy(agent, eval_env)
            writer.add_scalar("eval/return", mean_ret, step+1)
            print(f"\n[eval] step {step+1} | mean_return = {mean_ret:.1f}")

            eval_steps.append(step + 1)          
            eval_returns.append(mean_ret)        

            # early-stopping
            if mean_ret >= cfg.solve_score:
                consec_good += 1
            else:
                consec_good = 0
            if consec_good >= cfg.early_patience:
                print("Early stopping — environment solved!")
                break

    writer.close()
    # save final net
    torch.save(agent.online.state_dict(), "dqn_cartpole.pt")


        # ─── save & plot training curve ──────────────────────────────────
    os.makedirs("plots", exist_ok=True)

    # Save raw data for later reuse
    np.savez("cartpole_training_curve.npz",
             steps=np.array(eval_steps),
             returns=np.array(eval_returns))

    # Quick plot
    plt.figure(figsize=(6,4))
    plt.plot(eval_steps, eval_returns, marker='o', label="Mean Return")
    plt.axhline(cfg.solve_score, color='red', linestyle='--', linewidth=1.2,
            label=f"Solved threshold ({cfg.solve_score})")
    plt.xlabel("Training step")
    plt.ylabel("Mean return (50 eps)")
    plt.title("CartPole DQN learning curve")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/cartpole_learning_curve.png", dpi=300)
    # plt.show()  # uncomment if you want to view during training

    env.close(); eval_env.close()

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
