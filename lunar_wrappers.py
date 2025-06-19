# lunar_wrappers.py  — trimmed version
import numpy as np
import gymnasium as gym
from Box2D.b2 import vec2

class LunarParamWrapper(gym.Wrapper):
    def __init__(self, env, gravity=None, lander_mass=None):
        super().__init__(env)
        self.req = dict(gravity=gravity,
                        lander_mass=lander_mass)
        self.ranges = None  # filled on first reset

    @staticmethod
    def _norm_range(default, rng):
        if rng is None:
            return None
        if isinstance(rng, (int, float)):
            pct = abs(rng)
            return (default*(1-pct), default*(1+pct))
        return tuple(rng)

    # -------------------------------------------------------------------
    def reset(self, **kwargs):
        log = kwargs.pop("log_params", False)
        obs, info = super().reset(**kwargs)
        u = self.env.unwrapped

        # build ranges on first reset
        if self.ranges is None:
            self.ranges = {
                "gravity"     : self._norm_range(abs(u.world.gravity.y),
                                                 self.req["gravity"]),
                "lander_mass" : self._norm_range(u.lander.mass,
                                                 self.req["lander_mass"]),
            }

        # ----- gravity --------------------------------------------------
        if self.ranges["gravity"]:
            g = np.random.uniform(*self.ranges["gravity"])
            u.world.gravity = vec2(0.0, -g)     # ← set full vector
            if log: print(f"gravity → {g:.2f}")

        # ----- lander mass ---------------------------------------------
        if self.ranges["lander_mass"]:
            m = np.random.uniform(*self.ranges["lander_mass"])
            u.lander.mass = m
            if log: print(f"lander_mass → {m:.2f}")

        return obs, info


# env = LunarParamWrapper(gym.make("LunarLander-v3"), gravity=(100, 100))
# obs, _ = env.reset(log_params=True)
# print("world.gravity =", env.unwrapped.world.gravity)  # should show (0,-100)
# env.close()
