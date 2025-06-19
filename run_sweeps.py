# run_sweeps.py
from robust_framework import load_agent, sweep_1d, plot_1d, save_results
import matplotlib.pyplot as plt
import numpy as np


episodes_g = 200
episodes_L = 200
episodes_cm = 200
episodes_pm = 200

g_vals          = np.linspace(0.01, 60, 100)
L_vals          = np.linspace(0.01, 1.0, 100)
cartmass_vals   = np.linspace(0.01, 4, 100)
polemass_vals   = np.linspace(0.01, 4, 100)
# polemass_vals   = np.logspace(0, 1, 10)


agent = load_agent("dqn_cartpole.pt")

# ---- gravity sweep --------------------------------------------
g_res  = sweep_1d(
    agent, 
    "gravity", 
    g_vals, 
    episodes=episodes_g, 
    verbose=True
    )

# save_results(g_res, "gravity_DQN")         

plot_1d(
    g_res, 
    episodes_g,
    x_label="Gravity $g$ (m s$^{-2}$)",
    title="Robustness: gravity",
    fname="gravity_robustness.png",
    save_fig=False
)


# ---- pole-length sweep ----------------------------------------
L_res  = sweep_1d(
    agent, 
    "length", 
    L_vals, 
    episodes=episodes_L, 
    verbose=True
    )

# save_results(L_res, "length_DQN")         

plot_1d(L_res, 
        episodes_L,
        x_label="Pole half-length $\\ell$ (m)",
        title="Robustness: pole length",
        fname="length_robustness.png",
        save_fig=False
)

# ---- cart mass sweep ----------------------------------------


cartmass_results = sweep_1d(
    agent,
    param_name="masscart",
    values=cartmass_vals,
    episodes=episodes_cm,
    verbose=True               # set False to silence console
)

# save_results(cartmass_results, "cartmass_DQN")         

# 3) plot mean ± 1 SEM just like gravity/length figures
plot_1d(
    results=cartmass_results,
    episodes=episodes_cm,
    x_label="Cart mass $m_{cart}$ (kg)",
    title="CartPole DQN robustness to cart mass",
    fname="masscart_robustness.png",
    save_fig=False
)

# ---- pole mass sweep ----------------------------------------

polemass_results = sweep_1d(
    agent,
    param_name="masspole",
    values=polemass_vals,
    episodes=episodes_pm,
    verbose=True               # set False to silence console
)

# save_results(polemass_results, "polemass_DQN")         

# 3) plot mean ± 1 SEM just like gravity/length figures
plot_1d(
    results=polemass_results,
    episodes=episodes_pm,
    x_label="Pole mass $m_{pole}$ (kg)",
    title="CartPole DQN robustness to pole mass",
    fname="masspole_robustness.png",
    save_fig=False
)


plt.show()