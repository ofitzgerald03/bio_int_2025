# wrappers.py
import numpy as np
import gymnasium as gym

class ParamWrapper(gym.Wrapper):
    """
    Overwrite CartPole dynamics parameters on every reset.
    Pass keyword arguments as {attr_name: value_or_(min,max)_range}.
    Attributes you can change: gravity, length, masspole, masscart, force_mag
    """
    def __init__(self, env, **param_ranges):
        super().__init__(env)
        self._ranges = {}
        for name, rng in param_ranges.items():
            default = getattr(env.unwrapped, name)
            if isinstance(rng, (int, float)):
                pct = abs(rng)
                self._ranges[name] = (default*(1-pct), default*(1+pct))
            else:
                self._ranges[name] = tuple(rng)

    def _recompute(self):
        u = self.env.unwrapped
        u.total_mass      = u.masspole + u.masscart
        u.polemass_length = u.masspole * u.length

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        for name, (lo, hi) in self._ranges.items():
            new_val = np.random.uniform(lo, hi)
            setattr(self.env.unwrapped, name, new_val)
        
        u = self.env.unwrapped
        if 'masscart' in self._ranges:
            u.masscart = np.random.uniform(*self._ranges['masscart'])
        if 'length' in self._ranges:
            u.length   = np.random.uniform(*self._ranges['length'])
        # add other params as needed

        self._recompute()          # ← make step() use new values
        return obs, info
