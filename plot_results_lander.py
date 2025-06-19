# plot_results_lander.py
# ---------------------------------------------------------------------------
# Re-draw Lunar-Lander robustness plots and Q-value heat-maps from saved data
# ---------------------------------------------------------------------------
from robust_framework import *
# load_results, plot_1d_lander, plot_policy_heatmaps_lander, load_agent_lander, PLOT_DIR, plot_policy_panel_lander
import numpy as np
import os
import matplotlib.pyplot as plt

# 1-D GRAVITY ---------------------------------------------------------------
g_res  = load_results("lander_gravity_DQN")
N_g    = len(next(iter(g_res.values())))
plot_1d_lander(g_res, N_g,
               x_label="Gravity $g$ (m s$^{-2}$)",
               title ="Lunar-Lander DQN: gravity sweep",
               fname ="lander_gravity.png",
               save_fig=True)

# 1-D MASS -----------------------------------------------------------------
m_res  = load_results("lander_mass_DQN")
N_m    = len(next(iter(m_res.values())))
plot_1d_lander(m_res, N_m,
               x_label="Lander mass $m$ (kg)",
               title ="Lunar-Lander DQN: mass sweep",
               fname ="lander_mass.png",
               save_fig=True)


plot_1d_lander_pair(
    g_res, m_res,
    episodes_g = N_g,
    episodes_m = N_m,
    xlim_g = (0, 20),          # customise as you like
    xlim_m = (0, 12),
    save_path = "plots/lander_gravity_mass_pair.png"
)



# POLICY HEAT-MAP SLICE ----------------------------------------------------
agent = load_agent_lander("lunar_dqn.pt", env_id="LunarLander-v3")

# Example slice: horizontal x (idx0) vs vertical y (idx1) with other vars 0
base_state = np.zeros(8, dtype=np.float32)
# plot_policy_heatmaps_lander(agent,
#                             var1_idx=0, var2_idx=1,
#                             fixed_state=base_state,
#                             var1_range=(-1.0, 1.0),
#                             var2_range=(0.0, 2.5),
#                             resolution=300,
#                             save_prefix="lander_xy_policy")

# -----------------------------------------------------------------------
# EXTRA POLICY SLICES  (add after the first slice block)
# -----------------------------------------------------------------------
slices = [
    # 0 – 1  : x   vs  y
    dict(var1=0, var2=1, v1_range=(-2.5,  2.5), v2_range=(-1,  2.5),
         prefix="lander_x_y",         desc="x vs y"),

    # 0 – 2  : x   vs  ẋ
    dict(var1=0, var2=2, v1_range=(-2.5,  2.5), v2_range=(-10.0,  10.0),
         prefix="lander_x_xdot",      desc="x vs ẋ"),

    # 0 – 3  : x   vs  ẏ
    dict(var1=0, var2=3, v1_range=(-2.5,  2.5), v2_range=(-10.0,  10.0),
         prefix="lander_x_ydot",      desc="x vs ẏ"),

    # 0 – 4  : x   vs  θ
    dict(var1=0, var2=4, v1_range=(-2.5,  2.5), v2_range=(-6.2831855, 6.2831855),
         prefix="lander_x_theta",     desc="x vs θ"),

    # 0 – 5  : x   vs  ω
    dict(var1=0, var2=5, v1_range=(-2.5,  2.5), v2_range=(-10.0,  10.0),
         prefix="lander_x_omega",     desc="x vs ω"),

    # 2–1 : ẋ vs y  (y vertical)
    dict(var1=2, var2=1, v1_range=(-10.0,  10.0), v2_range=(-1,  2.5),
         prefix="lander_xdot_y",      desc="ẋ vs y"),

    # 3–1 : ẏ vs y  (y vertical, ẏ horizontal)
    dict(var1=3, var2=1, v1_range=(-10.0,  10.0), v2_range=(-1,  2.5),
         prefix="lander_ydot_y",      desc="ẏ vs y"),

    # 4–1 : θ  vs y  (y vertical)
    dict(var1=4, var2=1, v1_range=(-6.2831855, 6.2831855), v2_range=(-1,  2.5),
         prefix="lander_theta_y",     desc="θ vs y"),

    # 5–1 : ω  vs y  (y vertical)
    dict(var1=5, var2=1, v1_range=(-10.0,  10.0), v2_range=(-1,  2.5),
         prefix="lander_omega_y",     desc="ω vs y"),

    # 2–3 : ẋ vs ẏ (ẏ vertical)
    dict(var1=2, var2=3, v1_range=(-10.0,  10.0), v2_range=(-10.0,  10.0),
         prefix="lander_xdot_ydot",   desc="ẋ vs ẏ"),

    # 2–4 : ẋ vs θ  (ẏ not involved, no change)
    dict(var1=2, var2=4, v1_range=(-10.0,  10.0), v2_range=(-6.2831855, 6.2831855),
         prefix="lander_xdot_theta",  desc="ẋ vs θ"),

    # 2–5 : ẋ vs ω  (no change)
    dict(var1=2, var2=5, v1_range=(-10.0,  10.0), v2_range=(-10.0,  10.0),
         prefix="lander_xdot_omega",  desc="ẋ vs ω"),

    # 4–3 : θ  vs ẏ (ẏ vertical)
    dict(var1=4, var2=3, v1_range=(-6.2831855, 6.2831855), v2_range=(-10.0,  10.0),
         prefix="lander_theta_ydot",  desc="θ vs ẏ"),

    # 5–3 : ω  vs ẏ (ẏ vertical)
    dict(var1=5, var2=3, v1_range=(-10.0,  10.0), v2_range=(-10.0,  10.0),
         prefix="lander_omega_ydot",  desc="ω vs ẏ"),

    # 4–5 : θ  vs ω  
    dict(var1=4, var2=5, v1_range=(-6.2831855, 6.2831855), v2_range=(-10.0,  10.0),
         prefix="lander_theta_omega", desc="θ vs ω"),
]



for sl in slices:
    print(f"Rendering slice: {sl['desc']}")
    plot_policy_panel_lander(
        agent,
        var1_idx = sl["var1"],
        var2_idx = sl["var2"],
        fixed_state = base_state,
        var1_range  = sl["v1_range"],
        var2_range  = sl["v2_range"],
        resolution  = 300,
        save_prefix = sl["prefix"]
    )


print(f"Figures saved to {PLOT_DIR}")

plt.show()