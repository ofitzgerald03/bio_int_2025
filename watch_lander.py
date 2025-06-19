# watch_lander.py
# ---------------------------------------------------------------------------
# Render a trained Lunar Lander *discrete* DQN for a few greedy episodes.
#
# Requirements:
#   • lunar_DQN.py has finished and saved "lunar_dqn.pt"
#   • gymnasium[box2d] installed
# ---------------------------------------------------------------------------

import time
import torch
import gymnasium as gym

# ─────────────────────────────────────────────────────────────────────────────
# Import network architecture from the training file
# ─────────────────────────────────────────────────────────────────────────────
from lunar_DQN import DQN

WEIGHTS_PATH = "lunar_dqn.pt"          # change if you saved with another name
EPISODES     = 5                       # number of demo runs
FPS          = 60                      # playback speed

def main():
    # 1.  create env in human-render mode
    env = gym.make("LunarLander-v3", render_mode="human")

    # 2.  rebuild the net and load weights
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = DQN(obs_dim, n_actions)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    net.eval()

    # helper for greedy action
    def act(state):
        with torch.no_grad():
            q = net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(q, 1).item())
 
    # 3.  run demo episodes
    for ep in range(1, EPISODES + 1):
        s, _ = env.reset(seed=None)
        done = truncated = False
        ep_ret = 0
        while not (done or truncated):
            a = act(s)
            s, r, done, truncated, _ = env.step(a)
            ep_ret += r
            time.sleep(1 / FPS)
        print(f"Episode {ep}: return = {ep_ret:.1f}")

    env.close()

if __name__ == "__main__":
    main()

