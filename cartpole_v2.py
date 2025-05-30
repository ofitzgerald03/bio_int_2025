import gymnasium as gym
import numpy as np
from collections import deque
from tqdm import trange    # nice progress-bar

SEED          = 42
N_EPISODES    = 4_000
GAMMA         = 0.99
# ALPHA         = 1e-2       # learning rate
EPS_START     = 1.0
EPS_END       = 0.01
# EPS_DECAY_STEPS = 25_000   # steps, not episodes

# 1.  Build env and seeding ---------------------------------------------------
env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=SEED)

# 2.  Pre-compute bin edges ---------------------------------------------------
# finer discretisation
bins = [
    np.linspace(-2.4,  2.4,  3 - 1),      # x
    np.linspace(-3.0,  3.0,  3 - 1),      # x_dot
    np.linspace(-0.418, 0.418, 10 - 1),   # theta  (10 bins)
    np.linspace(-4.0,   4.0, 10 - 1)      # omega  (10 bins)
]
ALPHA = 0.05
EPS_DECAY_STEPS = 10_000


def discretise(obs):
    """
    Convert continuous observation to a discrete state index.
    :param obs: np.ndarray of shape (4,) with [x, x_dot, theta, omega]
    :return: int, the discrete state index
    """
    idx = []
    for o, b in zip(obs, bins):
        idx.append(np.digitize(o, b))
    # mixed-radix to flat index
    i0, i1, i2, i3 = idx
    return (((i0 * 3) + i1) * 10 + i2) * 10 + i3

n_states  = 3 * 3 * 10 * 10
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions), dtype=np.float32)

def evaluate_policy(Q_eval, env, n_episodes=100):
    total, perfect = 0, 0
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=None)
        s = discretise(obs)
        done = truncated = False
        ep_ret = 0
        while not (done or truncated):
            a = np.argmax(Q_eval[s])
            obs, r, done, truncated, _ = env.step(a)
            ep_ret += r
            s = discretise(obs)
        total   += ep_ret
        perfect += (ep_ret == 500)
    mean_ret = total / n_episodes
    return mean_ret, perfect           



# 3.  Training loop -----------------------------------------------------------
eps  = EPS_START
steps = 0
returns = deque(maxlen=100)
CONSEC_TARGET = 3     # need 3 evals in a row to stop
consec_hits   = 0

for ep in trange(N_EPISODES):
    obs, _ = env.reset()
    state  = discretise(obs)
    done   = truncated = False
    ep_ret = 0

    while not (done or truncated):
        # eps-greedy policy
        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_obs, reward, done, truncated, _ = env.step(action)
        next_state = discretise(next_obs)

        ep_ret += reward
        steps  += 1

        # # Q-learning update
        td_target = reward + GAMMA * np.max(Q[next_state])
        Q[state, action] += ALPHA * (td_target - Q[state, action])

        state = next_state

        # epsilon linear decay
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * steps / EPS_DECAY_STEPS)

    returns.append(ep_ret)

    # Logging every 100 episodes
    if (ep + 1) % 100 == 0:
        mean_return = np.mean(returns)
        q_max = np.max(np.abs(Q))         
        print(f"Episode {ep+1:4d} | eps={eps:.3f} | mean_return (last 100): {mean_return:.1f} | Q_max: {q_max:.2f}")
        # --- 2. epsilon-greedy evaluation every 500 episodes -------------------------------
        if (ep + 1) % 500 == 0:
            # eval_return = 0
            # obs, _ = env.reset()
            # state = discretise(obs)
            # done = truncated = False
            # while not (done or truncated):
            #     action = np.argmax(Q[state])          # greedy (epsilon=0)
            #     obs, r, done, truncated, _ = env.step(action)
            #     eval_return += r
            #     state = discretise(obs)
            # print(f"Greedy-policy return: {eval_return}")
            # evaluate_policy(Q.copy(), env, n_episodes=500)  
            mean_ret, perfect = evaluate_policy(Q.copy(), env, n_episodes=100)
            print(f"[Eval] mean_return={mean_ret:.1f} | perfect {perfect}/100")

            # --- early-stop check -----------------------------
            if perfect >= 95 or mean_ret >= 475:    # good enough?
                consec_hits += 1
            else:
                consec_hits = 0

            if consec_hits >= CONSEC_TARGET:
                print("Early stopping — policy consistently solves CartPole.")
                np.save("cartpole_qtable.npy", Q)   # save for later use
                break



env.close()
