# run_sweeps_lander.py
# ──────────────────────────────────────────────────────────────────────────
# Robustness evaluation for Lunar Lander using the reusable framework
# (works for the discrete DQN; swap weights file for SAC).
# ──────────────────────────────────────────────────────────────────────────

import os
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from robust_framework import *
# load_agent_lander, sweep_1d_lander, plot_1d_lander, save_results, PLOT_DIR
from lunar_wrappers import LunarParamWrapper

# ───── 0. Load trained agent ───────────────────────────────────────────────
agent = load_agent_lander("lunar_dqn.pt")           
# ===========================================================================
# 1. Gravity sweep  (g-range, engine power fixed)
# ===========================================================================
g_vals          = np.linspace(0.5, 20, 100)
episodes_per_g  = 200

gravity_res = sweep_1d_lander(
    agent,
    param_name="gravity",
    values=g_vals,
    episodes=episodes_per_g,
    fixed_params={},          # all else nominal
    verbose=True,
)

# save_results(gravity_res, "lander_gravity_DQN")

plot_1d_lander(
    results=gravity_res,
    episodes=episodes_per_g,
    x_label="Gravity $g$ (m s$^{-2}$)",
    title="Lunar-Lander robustness to gravity (DQN)",
    fname="lander_gravity.png",
    save_fig=False
)

# ===========================================================================
# 2. Lander-mass sweep  (gravity fixed, vary lander_mass)
# ===========================================================================
mass_vals        = np.linspace(0.1, 12, 100)   # kg   (default mass ≈ 4.8 kg)
episodes_per_m   = 200

mass_res = sweep_1d_lander(
    agent,
    param_name="lander_mass",
    values=mass_vals,
    episodes=episodes_per_m,
    fixed_params={},       
    env_id="LunarLander-v3",
    verbose=True
)

# save_results(mass_res, "lander_mass_DQN")

plot_1d_lander(
    results=mass_res,
    episodes=episodes_per_m,
    x_label="Lander mass $m$ (kg)",
    title="Lunar-Lander robustness to mass (DQN)",
    fname="lander_mass.png",
    save_fig=False
)


plot_1d_lander_pair(
    gravity_res, mass_res,
    episodes_g = episodes_per_g,
    episodes_m = episodes_per_m,
    xlim_g = (0, 20),          # customise as you like
    xlim_m = (0, 12),
    save_path = None
)

# # ===========================================================================
# # 2. Engine-power sweep  (gravity fixed, vary MAIN_ENGINE_POWER)
# # ===========================================================================
# engine_vals        = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]    # × default
# episodes_per_eng   = 100

# engine_res = sweep_1d_lander(
#     agent,
#     param_name="main_engine",
#     values=engine_vals,
#     episodes=episodes_per_eng,
#     fixed_params={"gravity": (9.8, 9.8)},   # hold g constant
#     verbose=True
# )
# save_results(engine_res, "lander_engine_DQN")

# plot_1d(
#     results=engine_res,
#     episodes=episodes_per_eng,
#     x_label="Main-engine power (x default)",
#     title="Lunar-Lander robustness to engine power (DQN)",
#     fname="lander_engine.png"
# )

# # ===========================================================================
# # 3. 2-D grid:  Gravity × Main-engine power
# # ===========================================================================
# g_grid      = [8, 9.8, 12, 15]          # rows
# eng_grid    = [0.6, 0.8, 1.0, 1.2]      # columns
# episodes_2d = 50

# heat = sweep_2d(
#     agent,
#     param1="gravity",     vals1=g_grid,
#     param2="main_engine", vals2=eng_grid,
#     episodes=episodes_2d
# )
# save_results(heat, "lander_g_vs_engine_DQN")

# # heat-map plot
# fig, ax = plt.subplots(figsize=(6,4))
# sns.heatmap(
#     heat, annot=True, fmt=".0f",
#     xticklabels=eng_grid, yticklabels=g_grid,
#     cbar_kws={'label': 'Mean return'}, ax=ax
# )
# ax.set_xlabel("Main-engine power (x default)")
# ax.set_ylabel("Gravity $g$ (m s$^{-2}$)")
# ax.set_title(f"Lunar-Lander DQN: return vs. gravity & engine ({episodes_2d} runs)")
# fig.tight_layout()
# fig.savefig(os.path.join(PLOT_DIR, "lander_g_vs_engine.png"), dpi=300)
plt.show()
