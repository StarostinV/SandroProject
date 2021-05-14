from typing import Tuple
import numpy as np

from numpy.core.umath_tests import inner1d

from .beam import Beam
from .plane import RectPlane


class Incoming(RectPlane):
    def __init__(self, length: float, width: float, shape: Tuple[str, float, float], inc_intensity: float,
                 gauss: bool = True, amp: float = 1.0, sigma: float = 1.0, ray_numb: float = 10000,
                 det_distance: float = 10.0):
        super().__init__(length, width)
        self.center = np.array([det_distance, 0, 0])
        self.rotate_y(90)
        self.shift_x(det_distance)
        self.set_mesh(ray_numb, det_distance)
        self.set_profile(shape, inc_intensity, amp, sigma, gauss)

    def set_mesh(self, ray_numb, det_distance):
        y, z = np.meshgrid(np.linspace(-self.edges[0][1], self.edges[0][1], int(np.sqrt(ray_numb))),
                           np.linspace(-self.edges[0][2], self.edges[0][2], int(np.sqrt(ray_numb))))
        z, y = z.flatten(), y.flatten()
        x = np.ones_like(z) * det_distance
        intensity = np.zeros_like(z)
        coords = np.stack([x, y, z]).T
        dv = np.array([-1, 0, 0])
        self.beam_profile = Beam(coords, intensity, dv)

    def set_profile(self, shape, inc_intensity, amp, sigma, gauss):
        if gauss:
            self.beam_profile.intensity = self.create_gauss_pattern(amp, sigma)
        else:
            self.beam_profile.intensity = np.ones(len(self.beam_profile.coords.T[0]))
        inc_center = self.beam_profile.coords - self.center
        if shape[0] == "round":
            self.beam_profile.intensity[inner1d(inc_center, inc_center) > shape[1]] = 0
        elif shape[0] == "elliptical":
            arr = inc_center / np.array([1, shape[1], shape[2]])
            self.beam_profile.intensity[inner1d(arr, arr) > 1] = 0
        elif shape[0] == "quadratic":
            pass
        else:
            print("Wrong input in shape of incoming beam.")
            print("Possible: round, elliptical, quadratic")
        self.beam_profile.intensity *= inc_intensity / np.sum(self.beam_profile.intensity)

    def create_gauss_pattern(self, amp: float, sigma: float):
        y, z = self.beam_profile.coords.T[1], self.beam_profile.coords.T[2]
        d = np.sqrt(y ** 2 + z ** 2)
        return amp * (np.exp(-(d ** 2 / (2.0 * sigma ** 2)))).flatten()
