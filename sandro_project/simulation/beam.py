import numpy as np


class Beam(object):
    def __init__(self, coords=None, intensity=None,
                 dv: np.array(3, float) = None):
        self.coords = coords
        self.intensity = intensity
        self.dv = dv
