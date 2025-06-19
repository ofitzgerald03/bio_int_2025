# robust_framework.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch   
from collections import defaultdict
import gymnasium as gym
import torch
from wrappers import ParamWrapper
from lunar_wrappers import LunarParamWrapper
from cartpole_DQN import DQNAgent, evaluate_policy   
from lunar_DQN import DQNAgent_lander, evaluate_policy_lander
import pickle
# import seaborn as sns



PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------------------------------------
def load_agent(weights_path="dqn_cartpole.pt"):
    """Rebuild agent wrapper and load trained weights."""
    dummy = gym.make("CartPole-v1")
    agent = DQNAgent(dummy.observation_space.shape[0],
                     dummy.action_space.n)
    agent.online.load_state_dict(torch.load(weights_path, map_location="cpu"))
    agent.online.eval()
    dummy.close()
    return agent


# robust_framework.py  (add below the existing load_agent for CartPole)


def load_agent_lander(weights_path="lunar_dqn.pt",
                      env_id="LunarLander-v3"):
    """
    Rebuild a Lunar-Lander DQNAgent and load trained weights.
    Only *.pt files (state-dict) are supported.
    """
    if not weights_path.endswith(".pt"):
        raise ValueError("load_agent_lander expects a .pt file (DQN state-dict)")

    # dummy env to get observation & action dimensions
    dummy_env = gym.make(env_id)
    obs_dim   = dummy_env.observation_space.shape[0]
    n_actions = dummy_env.action_space.n
    dummy_env.close()

    agent = DQNAgent_lander(obs_dim, n_actions)
    agent.online.load_state_dict(torch.load(weights_path, map_location="cpu"))
    agent.online.eval()
    return agent

# ------------------------------------------------
def sweep_1d(agent, param_name, values, episodes=100, fixed_params=None, verbose=True):
    """
    Evaluate agent over a 1-D list of parameter values.
    param_name  : e.g. "gravity" or "length"
    values      : iterable of scalars
    fixed_params: dict of other params to keep fixed
    Returns dict {val: [episode returns]}
    """
    fixed_params = fixed_params or {}
    results = defaultdict(list)
    for v in values:
        env = ParamWrapper(gym.make("CartPole-v1"),
                           **{param_name: (v, v)},  # vary this param
                           **fixed_params)          # plus fixed others
        ep_returns = [evaluate_policy(agent, env, 1) for _ in range(episodes)]
        env.close()
        results[v] = ep_returns
        if verbose:
            print(f"{param_name} = {v:6.2f} | mean = {np.mean(ep_returns):6.1f} "
                  f"| std = {np.std(ep_returns):6.1f}")

    # if verbose:
    #     print("\nSummary:")
    #     for v in values:
    #         print(f"{v:6.1f} → mean {np.mean(results[v]):6.1f}")
    return results

# ---------------------------------------------------------------------------
# sweep_1d_lander
#   • Works exactly like sweep_1d but targets Lunar-Lander environments.
#   • param_name    : "gravity", "main_engine", "lander_mass", etc.
#   • values        : iterable of scalars to test
#   • episodes      : greedy episodes per value
#   • fixed_params  : dict of other params held constant, e.g. {"gravity": (9.8, 9.8)}
#   • env_id        : "LunarLander-v3"  (discrete)  or  "LunarLanderContinuous-v3"
# ---------------------------------------------------------------------------

def sweep_1d_lander(agent,
                    param_name,
                    values,
                    episodes=200,
                    fixed_params=None,
                    env_id="LunarLander-v3",
                    verbose=True):
    fixed_params = fixed_params or {}
    results = defaultdict(list)

    for v in values:
        env = LunarParamWrapper(
                gym.make(env_id),
                **{param_name: (v, v)},     # vary param of interest
                **fixed_params)             # keep others fixed
        ep_returns = [evaluate_policy_lander(agent, env, 1) for _ in range(episodes)]
        env.close()

        results[v] = ep_returns
        if verbose:
            print(f"{param_name} = {v:6.2f} | "
                  f"mean = {np.mean(ep_returns):6.1f} | "
                  f"std = {np.std(ep_returns):6.1f}")
    return results


# ------------------------------------------------
def plot_1d(results, episodes, x_label, title, fname, save_fig=False):
    x      = np.array(sorted(results))
    mean   = np.array([np.mean(results[v]) for v in x])
    sem    = np.array([np.std(results[v])  for v in x]) / np.sqrt(episodes)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, mean, marker='', label='DQN mean return')
    ax.fill_between(x, mean-sem, mean+sem, alpha=0.25, label=r'$\pm$ 1 SEM')
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Mean episode return ({episodes} runs)")
    ax.set_title(title)
    ax.set_ylim(-20, 520)
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(PLOT_DIR, fname), dpi=300)
    # plt.show()

# ---------------------------------------------------------------------------
# plot_1d_cartpole_panel
#   curves : list of dicts, each
#       {
#         "results"  : {param_val: [returns]},
#         "episodes" : int,              # N runs per point
#         "x_label"  : str,
#         "title"    : str,
#         "xlim"     : (xmin, xmax) or None
#       }
#   save_path : if given, figure is saved; otherwise shown
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# plot_1d_cartpole_panel_2x2
#   curves : list of 4 dicts, each with keys
#       "results"  : {param_val: [returns]}
#       "episodes" : int             # N runs per point
#       "x_label"  : str
#       "title"    : str
#       "xlim"     : (xmin, xmax) or None  (optional)
#   save_path : if given, figure saved; else shown
# ---------------------------------------------------------------------------
def plot_1d_cartpole_panel_2x2(curves, save_path=None):

    assert len(curves) == 4, "Need exactly four curves for a 2x2 panel"

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axs = axs.flatten()

    for ax, cfg in zip(axs, curves):
        res = cfg["results"]; N = cfg["episodes"]
        x   = np.array(sorted(res))
        mean= np.array([np.mean(res[v]) for v in x])
        sem = np.array([np.std(res[v])  for v in x]) / np.sqrt(N)

        ax.plot(x, mean, marker='', label='DQN mean return', color="C0")
        ax.fill_between(x, mean-sem, mean+sem, alpha=0.4, label=r'$\pm$ SEM', color="C3")

        # vertical nominal line
        if "nominal" in cfg:
            ax.axvline(cfg["nominal"], color="red",
                       linestyle="--", linewidth=1.3,
                       label="Nominal")
            
        ax.set_xlabel(cfg["x_label"]); ax.set_title(cfg["title"])
        if cfg.get("xlim"): ax.set_xlim(*cfg["xlim"])
        ax.grid(alpha=0.4)
        ax.legend()

    # shared y-axis formatting
    for ax in axs[::2]:    # left column
        ax.set_ylabel("Mean episode return")
    axs[0].set_ylim(-20, 520)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300)
    else:
        pass
        # plt.show()


def plot_1d_lander(results, episodes, x_label, title, fname, save_fig=False):
    x      = np.array(sorted(results))
    mean   = np.array([np.mean(results[v]) for v in x])
    sem    = np.array([np.std(results[v])  for v in x]) / np.sqrt(episodes)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, mean, marker='', label='DQN mean return')
    ax.fill_between(x, mean-sem, mean+sem, alpha=0.25, label=r'$\pm$ 1 SEM')
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Mean episode return ({episodes} runs)")
    ax.set_title(title)
    ax.set_ylim(-200, 300)
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(PLOT_DIR, fname), dpi=300)
    # plt.show()

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

def save_results(obj, filename):
    """Pickle `obj` to results/<filename>.pkl."""
    path = os.path.join(RESULT_DIR, f"{filename}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[saved] {path}")

def load_results(filename):
    """Load object previously saved with save_results()."""
    path = os.path.join(RESULT_DIR, f"{filename}.pkl")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[loaded] {path}")
    return obj

# ---------------------------------------------------------------------------
# plot_1d_lander_pair
#   gravity_res : dict {g: [returns]}
#   mass_res    : dict {m: [returns]}
#   episodes_g  : int   – # episodes per g point
#   episodes_m  : int   – # episodes per mass point
#   xlim_g      : tuple – (xmin, xmax) for gravity axis   (optional)
#   xlim_m      : tuple – (xmin, xmax) for mass   axis     (optional)
#   save_path   : str or None
# ---------------------------------------------------------------------------
def plot_1d_lander_pair(
        gravity_res, mass_res,
        episodes_g, episodes_m,
        xlim_g=None, xlim_m=None,
        nominal_g=10.0,           # ← default nominal gravity
        nominal_m=4.8167,         # ← default nominal mass
        save_path=None):

    def _prep(res, N):
        x   = np.array(sorted(res))
        avg = np.array([np.mean(res[v]) for v in x])
        sem = np.array([np.std(res[v])  for v in x]) / np.sqrt(N)
        return x, avg, sem

    gx, gmean, gsem = _prep(gravity_res, episodes_g)
    mx, mmean, msem = _prep(mass_res,    episodes_m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # ── gravity subplot ────────────────────────────────────────────────
    ax1.plot(gx, gmean, color="C0", label='DQN mean return')
    ax1.fill_between(gx, gmean-gsem, gmean+gsem, alpha=0.25, color="C3",
                     label=r'$\pm$ SEM')
    ax1.axvline(nominal_g, color="red", linestyle="--", linewidth=1.2,
                label="Nominal")
    ax1.set_xlabel("Gravity $g$ (m s$^{-2}$)")
    ax1.set_title("Gravity sweep"); ax1.grid(alpha=0.3)
    if xlim_g: ax1.set_xlim(*xlim_g)
    ax1.legend()

    # ── mass subplot ───────────────────────────────────────────────────
    ax2.plot(mx, mmean, color="C0", label='DQN mean return')
    ax2.fill_between(mx, mmean-msem, mmean+msem, alpha=0.25, color="C3",
                     label=r'$\pm$ SEM')
    ax2.axvline(nominal_m, color="red", linestyle="--", linewidth=1.2,
                label="Nominal")
    ax2.set_xlabel("Lander mass $m$ (kg)")
    ax2.set_title("Mass sweep"); ax2.grid(alpha=0.3)
    if xlim_m: ax2.set_xlim(*xlim_m)
    ax2.legend()

    # shared y-axis and layout
    for ax in (ax1, ax2):
        ax.set_ylabel("Mean return")
        ax.set_ylim(-200, 300)
    # fig.suptitle("Lunar-Lander DQN robustness (gravity vs mass)", fontsize=14)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300)


def plot_policy_heatmaps_cartpole(agent, var1_idx, var2_idx, fixed_state, var1_range, var2_range,
                         resolution=100, save_prefix=None):
    """
    Visualize Q-values and greedy policy over a 2D state slice.
    
    Args:
        agent: trained DQNAgent
        var1_idx: index of x-axis state variable
        var2_idx: index of y-axis state variable
        fixed_state: full 4D state with var1 and var2 to be varied
        var1_range: (min, max) for x-axis
        var2_range: (min, max) for y-axis
        resolution: grid resolution
        save_prefix: base filename (no extension); if None, does not save
    """
    q_map = np.zeros((resolution, resolution, agent.n_actions))
    v1_vals = np.linspace(*var1_range, resolution)
    v2_vals = np.linspace(*var2_range, resolution)
    device = next(agent.online.parameters()).device

    for i, v1 in enumerate(v1_vals):
        for j, v2 in enumerate(v2_vals):
            state = np.array(fixed_state)
            state[var1_idx] = v1
            state[var2_idx] = v2
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = agent.online(state_t).cpu().numpy().squeeze()
            q_map[j, i, :] = q_vals

    state_labels = [
        "Cart position $x$ (m)",
        "Cart velocity $\dot{x}$ (m/s)",
        "Pole angle $\\theta$ (rad)",
        "Pole angular velocity $\dot{\\theta}$ (rad/s)"
    ]
    x_label = state_labels[var1_idx]
    y_label = state_labels[var2_idx]

    action_titles = {
        0: "Q-values for action 0 (Push LEFT)",
        1: "Q-values for action 1 (Push RIGHT)"
    }

    os.makedirs("plots", exist_ok=True)

    for action in range(agent.n_actions):
        plt.figure(figsize=(6, 5))
        plt.imshow(q_map[:, :, action], extent=(*var1_range, *var2_range),
                   origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Q-value')
        plt.title(action_titles[action])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        if save_prefix:
            fname = os.path.join("plots", f"{save_prefix}_action{action}.png")
            plt.savefig(fname, dpi=300)
        else:
            plt.show()

    policy_map = np.argmax(q_map, axis=2)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(policy_map, extent=(*var1_range, *var2_range),
                    origin='lower', aspect='auto', cmap='coolwarm')
    cbar = plt.colorbar(im)
    cbar.set_label("Greedy action (0 or 1)")
    plt.title("Greedy policy map (argmax Q)")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    if save_prefix:
        fname = os.path.join("plots", f"{save_prefix}_policy.png")
        plt.savefig(fname, dpi=300)
    else:
        pass
        # plt.show()


def plot_policy_subplots(agent, var1_idx, var2_idx, fixed_state,
                         var1_range, var2_range, resolution=100,
                         fig_title=None, save_path=None):
    """
    Create a 3-subplot figure: Q-value action 0, Q-value action 1, Greedy policy.
    Saves to save_path if given.
    """

    q_map = np.zeros((resolution, resolution, agent.n_actions))
    v1_vals = np.linspace(*var1_range, resolution)
    v2_vals = np.linspace(*var2_range, resolution)
    device = next(agent.online.parameters()).device

    for i, v1 in enumerate(v1_vals):
        for j, v2 in enumerate(v2_vals):
            state = np.array(fixed_state)
            state[var1_idx] = v1
            state[var2_idx] = v2
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = agent.online(state_t).cpu().numpy().squeeze()
            q_map[j, i, :] = q_vals

    policy_map = np.argmax(q_map, axis=2)

    vmin = q_map[:, :, :2].min()      # min over both actions
    vmax = q_map[:, :, :2].max()      # max over both actions

    # Labels
    state_labels = [
        "Cart position $x$ (m)",
        "Cart velocity $\dot{x}$ (m/s)",
        "Pole angle $\\theta$ (rad)",
        "Pole angular velocity $\dot{\\theta}$ (rad/s)"
    ]
    x_label = state_labels[var1_idx]
    y_label = state_labels[var2_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Q-value: action 0 (Push LEFT)",
              "Q-value: action 1 (Push RIGHT)",
              "Greedy policy (argmax Q)"]

    cmaps = ['viridis', 'viridis', 'coolwarm']
    datas = [q_map[:, :, 0], q_map[:, :, 1], policy_map]

    for ax, data, title, cmap in zip(axes, datas, titles, cmaps):
        if title.startswith("Q-value"):               # share scale
            im = ax.imshow(data, extent=(*var1_range,*var2_range),
                           origin='lower', aspect='auto',
                           cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.85)
        else:                                         # policy map
            # im = ax.imshow(data, extent=(*var1_range,*var2_range),
            #                origin='lower', aspect='auto', cmap=cmap)
            im = ax.imshow(data, extent=(*var1_range,*var2_range),
                        origin='lower', aspect='auto', cmap=cmap)

            colors  = [plt.cm.coolwarm(0.), plt.cm.coolwarm(1.)]
            labels  = ["LEFT", "RIGHT"]
            patches = [Patch(color=c, label=l) for c, l in zip(colors, labels)]
            ax.legend(handles=patches, loc="center left",
                    bbox_to_anchor=(1.02, 0.5), frameon=False)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        # fig.colorbar(im, ax=ax, shrink=0.85)

    # if fig_title:
    #     fig.suptitle(fig_title, fontsize=16)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        pass
        # plt.show()



# ---------------------------------------------------------------------------
# plot_policy_heatmaps_lander
#   • agent        : Lunar-Lander DQNAgent (4 actions)
#   • var1_idx/var2_idx : indices (0-7) of the state slice to visualise
#   • fixed_state  : 8-element list/array with baseline values
#   • var?_range   : (min,max) tuples for the slice axes
#   • resolution   : grid resolution along each axis
#   • save_prefix  : if given, PNGs saved to plots/<prefix>_*.png
# ---------------------------------------------------------------------------
# def plot_policy_heatmaps_lander(agent,
#                                 var1_idx, var2_idx,
#                                 fixed_state,
#                                 var1_range, var2_range,
#                                 resolution=120,
#                                 save_prefix=None):

#     q_map   = np.zeros((resolution, resolution, agent.n_actions))
#     v1_vals = np.linspace(*var1_range, resolution)
#     v2_vals = np.linspace(*var2_range, resolution)
#     device  = next(agent.online.parameters()).device

#     for i, v1 in enumerate(v1_vals):
#         for j, v2 in enumerate(v2_vals):
#             s = np.array(fixed_state, dtype=np.float32)
#             s[var1_idx] = v1
#             s[var2_idx] = v2
#             with torch.no_grad():
#                 q = agent.online(torch.tensor(s, device=device).unsqueeze(0)).cpu().numpy().squeeze()
#             q_map[j, i, :] = q  # rows = y, cols = x

#     # axis labels ---------------------------------------------------------
#     labels = ["x pos", "y pos", "x vel", "y vel",
#               "angle θ", "ang vel ω", "left contact", "right contact"]
#     x_lab, y_lab = labels[var1_idx], labels[var2_idx]

#     action_titles = {
#         0: "Q-value: NO-OP",
#         1: "Q-value: Main engine",
#         2: "Q-value: Left engine",
#         3: "Q-value: Right engine"
#     }

#     os.makedirs(PLOT_DIR, exist_ok=True)

#     for a in range(agent.n_actions):
#         plt.figure(figsize=(6,5))
#         plt.imshow(q_map[:,:,a],
#                    extent=(*var1_range,*var2_range),
#                    origin="lower", aspect="auto", cmap="viridis")
#         plt.colorbar(label="Q-value")
#         plt.xlabel(x_lab); plt.ylabel(y_lab)
#         plt.title(action_titles[a]); plt.tight_layout()
#         if save_prefix:
#             plt.savefig(os.path.join(PLOT_DIR,
#                          f"{save_prefix}_action{a}.png"), dpi=300)
#         else:
#             pass
#             # plt.show()

#     # greedy-policy map ---------------------------------------------------
#     policy = np.argmax(q_map, axis=2)
#     plt.figure(figsize=(6,5))
#     im = plt.imshow(policy, extent=(*var1_range,*var2_range),
#                     origin="lower", aspect="auto", cmap="coolwarm")
#     plt.colorbar(im).set_label("Greedy action (0–3)")
#     plt.xlabel(x_lab); plt.ylabel(y_lab)
#     plt.title("Greedy policy (argmax Q)"); plt.tight_layout()
#     if save_prefix:
#         plt.savefig(os.path.join(PLOT_DIR, f"{save_prefix}_policy.png"), dpi=300)
#     else:
#         pass
#         # plt.show()

def plot_policy_panel_lander(agent,
                             var1_idx, var2_idx,
                             fixed_state,
                             var1_range, var2_range,
                             resolution=300,
                             save_prefix=None):
    """
    2 x 3 panel of Q-maps + greedy policy for Lunar Lander.
    """

    # ---------- compute Q grid -------------------------------------------
    import numpy as np, torch, matplotlib.pyplot as plt, os
    from matplotlib.patches import Patch

    n_actions = agent.n_actions
    q_map = np.zeros((resolution, resolution, n_actions))
    v1_vals = np.linspace(*var1_range, resolution)
    v2_vals = np.linspace(*var2_range, resolution)
    device = next(agent.online.parameters()).device

    for i, v1 in enumerate(v1_vals):
        for j, v2 in enumerate(v2_vals):
            s = np.array(fixed_state, dtype=np.float32)
            s[var1_idx] = v1
            s[var2_idx] = v2
            with torch.no_grad():
                q = agent.online(torch.tensor(s, device=device).unsqueeze(0)
                                 ).cpu().numpy().squeeze()
            q_map[j, i, :] = q

    # ---------- descriptive axis labels ----------------------------------
    labels = [
        "Horizontal position $x$ (m)",        # 0
        "Vertical position $y$ (m)",          # 1
        "Horizontal velocity $\\dot{x}$ (m/s)",# 2
        "Vertical velocity $\\dot{y}$ (m/s)",  # 3
        "Angle $\\theta$ (rad)",               # 4
        "Angular velocity $\\dot{\\theta}$ (rad/s)",  # 5
        "Left-leg contact",                    # 6
        "Right-leg contact"                    # 7
    ]
    x_lab, y_lab = labels[var1_idx], labels[var2_idx]

    titles = ["Q: No Action (0)",
              "Q: Left Engine (1)",
              "Q: Main Engine (2)",
              "Q: Right Engine (3)",
              "Greedy policy"]

    # ---------- figure & axes (2 × 3 grid) -------------------------------
    fig, axmat = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axs = axmat.flatten()
    pos = {0: 0, 1: 1, 2: 3, 3: 4}     # action → subplot
    policy_pos, unused_pos = 2, 5

    vmin, vmax = q_map.min(), q_map.max()

    # -------- Q-value layers ---------------------------------------------
    for a in range(n_actions):
        ax = axs[pos[a]]
        im = ax.imshow(q_map[:, :, a],
                       extent=(*var1_range, *var2_range),
                       origin="lower", aspect="auto",
                       cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(titles[a])
        ax.set_xlabel(x_lab); ax.set_ylabel(y_lab)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # -------- greedy-policy map ------------------------------------------
    p_ax = axs[policy_pos]
    im2 = p_ax.imshow(np.argmax(q_map, axis=2),
                      extent=(*var1_range, *var2_range),
                      origin="lower", aspect="auto", cmap="coolwarm")
    p_ax.set_title(titles[-1])
    p_ax.set_xlabel(x_lab); p_ax.set_ylabel(y_lab)

    colors = [plt.cm.coolwarm(0.), plt.cm.coolwarm(1.0),
              plt.cm.coolwarm(0.25), plt.cm.coolwarm(0.75)]
    patches = [Patch(color=c, label=l) for c, l in zip(
        colors, ["No Action", "Left Engine", "Main Engine", "Right Engine"])]
    p_ax.legend(handles=patches, loc="center left",
                bbox_to_anchor=(1.03, 0.5), frameon=False)

    # -------- tidy up -----------------------------------------------------
    axs[unused_pos].axis("off")
    fig.tight_layout()

    os.makedirs(PLOT_DIR, exist_ok=True)
    if save_prefix:
        fig.savefig(os.path.join(PLOT_DIR,
                                 f"{save_prefix}_panel.png"), dpi=300)



# import gymnasium as gym
# import numpy as np, sys, pkg_resources

# # 1) Library version
# print("Gymnasium version:", gym.__version__)
# try:
#     print("Box2D-py version :", pkg_resources.get_distribution("box2d-py").version)
# except Exception:
#     print("Box2D-py not installed as package")

# # 2) Environment spec (id, entry-point, kwargs)
# env = gym.make("LunarLander-v3")
# print("Env spec :", env.spec)          # shows version tag (v3) and kwargs

# # 3) Observation-space limits actually used
# print("obs.low :", env.observation_space.low)
# print("obs.high:", env.observation_space.high)
# env.close()


# # Quick check of CartPole default (nominal) parameters
# import gymnasium as gym

# env = gym.make("CartPole-v1")
# u   = env.unwrapped          # raw CartPoleEnv

# print("Gravity g            :", u.gravity)     # 9.8  (m s⁻²)
# print("Pole half-length ℓ   :", u.length)      # 0.5  (m)  ← half-length
# print("Cart mass m_cart     :", u.masscart)    # 1.0  (kg)
# print("Pole mass m_pole     :", u.masspole)    # 0.1  (kg)

# env.close()


# # Inspect default (nominal) parameters in LunarLander-v3
# import gymnasium as gym
# from pprint import pprint

# env = gym.make("LunarLander-v3")
# obs, _ = env.reset()               # creates Box2D world & bodies
# u      = env.unwrapped             # raw LunarLander object

# # --- gravity -------------------------------------------------------------
# gravity = abs(u.world.gravity.y)   # positive value (m s⁻²)
# print("Gravity g           :", gravity)

# # --- lander mass ---------------------------------------------------------
# lander_mass = u.lander.mass        # kg
# print("Lander mass m       :", lander_mass)

# --- engine powers -------------------------------------------------------
# Newer Gym versions store them in a dict; fall back to attr names if present
# main_power = (u.engine_power["main"] if hasattr(u, "engine_power")
#               else getattr(u, "MAIN_ENGINE_POWER",
#                    getattr(u, "main_engine_power")))
# side_power = (u.engine_power["side"] if hasattr(u, "engine_power")
#               else getattr(u, "SIDE_ENGINE_POWER",
#                    getattr(u, "side_engine_power")))

# print("Main engine power   :", main_power)
# print("Side engine power   :", side_power)

# env.close()
