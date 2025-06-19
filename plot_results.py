from robust_framework import *
import matplotlib.pyplot as plt

# Gravity plot
g_res = load_results("gravity_DQN")
episodes_per_g = len(next(iter(g_res.values())))   # infer N
plot_1d(g_res, episodes_per_g,
        x_label="Gravity $g$ (m s$^{-2}$)",
        title="CartPole DQN robustness to gravity",
        fname="gravity_robustness.png",
        save_fig=True)

# Length plot
L_res = load_results("length_DQN")
episodes_per_L = len(next(iter(L_res.values())))
plot_1d(L_res, episodes_per_L,
        x_label="Pole half-length $\\ell$ (m)",
        title="CartPole DQN robustness to pole length",
        fname="length_robustness.png",
        save_fig=True)

cartmass_res = load_results("cartmass_DQN")
episodes_per_cartmass = len(next(iter(cartmass_res.values())))
plot_1d(cartmass_res, episodes_per_cartmass,
        x_label="Cart mass $m_{cart}$ (kg)",
        title="CartPole DQN robustness to cart mass",
        fname="masscart_robustness.png",
        save_fig=True)

polemass_res = load_results("polemass_DQN")
episodes_per_polemass = len(next(iter(polemass_res.values())))
plot_1d(polemass_res, episodes_per_polemass,
        x_label="Pole mass $m_{pole}$ (kg)",
        title="CartPole DQN robustness to pole mass",
        fname="masspole_robustness.png",
        save_fig=True)


# ------- load robustness pickles -----------------------------------------
g_res        = load_results("gravity_DQN")
L_res        = load_results("length_DQN")
cartmass_res = load_results("cartmass_DQN")
polemass_res = load_results("polemass_DQN")

curves = [
    dict(results=g_res,
         episodes=len(next(iter(g_res.values()))),
         nominal = 9.8,
         x_label="Gravity $g$ (m s$^{-2}$)",
         title ="Gravity sweep",
         xlim  =(0, 60)),

    dict(results=L_res,
         episodes=len(next(iter(L_res.values()))),
         nominal = 0.5,
         x_label="Half-length $\\ell$ (m)",
         title ="Pole length sweep",
         xlim  =(0, 1.0)),

    dict(results=cartmass_res,
         episodes=len(next(iter(cartmass_res.values()))),
         nominal = 1.0,
         x_label="Cart mass $m_{cart}$ (kg)",
         title ="Cart mass sweep",
         xlim  =(0, 4)),

    dict(results=polemass_res,
         episodes=len(next(iter(polemass_res.values()))),
         nominal = 0.1, 
         x_label="Pole mass $m_{pole}$ (kg)",
         title ="Pole mass sweep",
         xlim  =(0, 4)),
]

plot_1d_cartpole_panel_2x2(curves, save_path="plots/cartpole_robustness_panel.png")


# agent = load_agent("dqn_cartpole.pt")
# # Fix x and x_dot, vary theta and theta_dot
# fixed_state = [0.0, 0.0, 0.0, 0.0]

# # θ (2) vs θ̇ (3)
# plot_policy_heatmaps_cartpole(
#     agent,
#     var1_idx=2, var2_idx=3,
#     fixed_state=fixed_state,
#     var1_range=(-0.3, 0.3),
#     var2_range=(-2.0, 2.0),
#     resolution=200,
#     save_prefix="theta_thetadot"
# )

# # x (0) vs ẋ (1)
# plot_policy_heatmaps_cartpole(
#     agent,
#     var1_idx=0, var2_idx=1,
#     fixed_state=fixed_state,
#     var1_range=(-2.4, 2.4),
#     var2_range=(-2.0, 2.0),
#     resolution=200,
#     save_prefix="x_xdot"
# )

# # ẋ (1) vs θ̇ (3)
# plot_policy_heatmaps_cartpole(
#     agent,
#     var1_idx=1, var2_idx=3,
#     fixed_state=fixed_state,
#     var1_range=(-2.0, 2.0),
#     var2_range=(-2.0, 2.0),
#     resolution=200,
#     save_prefix="xdot_thetadot"
# )

# # x (0) vs θ̇ (3)
# plot_policy_heatmaps_cartpole(
#     agent,
#     var1_idx=0, var2_idx=3,
#     fixed_state=fixed_state,
#     var1_range=(-2.4, 2.4),
#     var2_range=(-2.0, 2.0),
#     resolution=200,
#     save_prefix="x_thetadot"
# )

# # ẋ (1) vs θ (2)
# plot_policy_heatmaps_cartpole(
#     agent,
#     var1_idx=1, var2_idx=2,
#     fixed_state=fixed_state,
#     var1_range=(-2.0, 2.0),
#     var2_range=(-0.3, 0.3),
#     resolution=200,
#     save_prefix="xdot_theta"
# )

# plt.show()

agent = load_agent("dqn_cartpole.pt")
# Fix x and x_dot, vary theta and theta_dot
fixed_state = [0.0, 0.0, 0.0, 0.0]

# θ (2) vs θ̇ (3)
plot_policy_subplots(
    agent,
    var1_idx=2, var2_idx=3,
    fixed_state=fixed_state,
    var1_range=(-0.3, 0.3),
    var2_range=(-2.0, 2.0),
    resolution=200,
    fig_title="Pole angle $\\theta$ vs angular velocity $\dot{\\theta}$",
    save_path="plots/subplots_theta_thetadot.png"
)

# x (0) vs ẋ (1)
plot_policy_subplots(
    agent,
    var1_idx=0, var2_idx=1,
    fixed_state=fixed_state,
    var1_range=(-2.4, 2.4),
    var2_range=(-2.0, 2.0),
    resolution=200,
    fig_title="Cart position $x$ vs velocity $\dot{x}$",
    save_path="plots/subplots_x_xdot.png"
)

# ẋ (1) vs θ̇ (3)
plot_policy_subplots(
    agent,
    var1_idx=1, var2_idx=3,
    fixed_state=fixed_state,
    var1_range=(-2.0, 2.0),
    var2_range=(-2.0, 2.0),
    resolution=200,
    fig_title="Cart velocity $\dot{x}$ vs pole angular velocity $\dot{\\theta}$",
    save_path="plots/subplots_xdot_thetadot.png"
)

# x (0) vs θ̇ (3)
plot_policy_subplots(
    agent,
    var1_idx=0, var2_idx=3,
    fixed_state=fixed_state,
    var1_range=(-2.4, 2.4),
    var2_range=(-2.0, 2.0),
    resolution=200,
    fig_title="Cart position $x$ vs pole angular velocity $\dot{\\theta}$",
    save_path="plots/subplots_x_thetadot.png"
)

# ẋ (1) vs θ (2)
plot_policy_subplots(
    agent,
    var1_idx=1, var2_idx=2,
    fixed_state=fixed_state,
    var1_range=(-2.0, 2.0),
    var2_range=(-0.3, 0.3),
    resolution=200,
    fig_title="Cart velocity $\dot{x}$ vs pole angle $\\theta$",
    save_path="plots/subplots_xdot_theta.png"
)

# x (0) vs θ (2)
plot_policy_subplots(
    agent,
    var1_idx=0, var2_idx=2,
    fixed_state=fixed_state,
    var1_range=(-2.0, 2.0),
    var2_range=(-0.3, 0.3),
    resolution=200,
    fig_title="Cart position ${x}$ vs pole angle $\\theta$",
    save_path="plots/subplots_x_theta.png"
)

plt.show()