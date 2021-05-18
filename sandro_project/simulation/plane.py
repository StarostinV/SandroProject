import cProfile

import numpy as np
from numpy.core.umath_tests import inner1d

from beam import Beam

import cProfile, pstats, io


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


class RectPlane(object):
    def __init__(self, length: float, width: float, thickness: float = 0.0):
        self.length = length
        self.width = width
        self.thickness = thickness
        self.edges = np.stack([[length / 2, width / 2, thickness],
                               [-length / 2, width / 2, thickness],
                               [-length / 2, -width / 2, thickness],
                               [length / 2, -width / 2, thickness]])
        self.parametric = self._set_parametric()
        self.beam_profile = Beam()

    def _set_parametric(self):
        return np.stack([self.edges[0],
                         self.edges[1] - self.edges[0],
                         self.edges[3] - self.edges[0]])

    def _actualize_parametric(self):
        self.parametric = self._set_parametric()

    def _rotate(self, function, angle):
        angle = angle * (np.pi / 180)
        return function(angle).dot(self.edges.T).transpose(1, 0)

    def rotate_x(self, angle):
        self.edges = self._rotate(self.rx_mtx, angle)
        self._actualize_parametric()

    def rotate_y(self, angle):
        self.edges = self._rotate(self.ry_mtx, angle)
        self._actualize_parametric()

    def rotate_z(self, angle):
        self.edges = self._rotate(self.rz_mtx, angle)
        self._actualize_parametric()

    def _shift(self, distance, axis):
        self.edges = self.edges.T
        self.edges[axis] += distance
        self.edges = self.edges.T
        self._actualize_parametric()

    def shift_x(self, distance):
        self._shift(distance, 0)

    def shift_y(self, distance):
        self._shift(distance, 1)

    def shift_z(self, distance):
        self._shift(distance, 2)

    @staticmethod
    def rx_mtx(angle):
        cs, sn = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, cs, -sn], [0, sn, cs]])

    @staticmethod
    def ry_mtx(angle):
        cs, sn = np.cos(angle), np.sin(angle)
        return np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])

    @staticmethod
    def rz_mtx(angle):
        cs, sn = np.cos(angle), np.sin(angle)
        return np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])

    @staticmethod
    def plane_line_intersect(plane: np.array([float]), coords: np.array([float]), dv: np.array([float])):
        """

        :param plane: [point on plane, normal vector]
        :param coords: [[x1, y1, z1], [x2, y2, z2], ... , [xn, yn, zn]]
        :param dv: direction vector of incoming ray [x, y, z]
        :return: intersection point of every incoming ray with the plane
        """
        w = coords - plane[0]
        fac = w.dot(-np.array(plane[1])) / (dv.dot(plane[1]))
        fac = np.reshape(fac, (-1, 1))
        u = dv * fac
        result = coords + u
        if len(coords.T[0].shape) > 0:
            result[np.where(fac.reshape(coords.T[0].shape[0]) < 0)] = np.array([-10000, -10000, -10000])
        return result

    def reflection(self, line: np.array([float])) -> np.array([float]):
        """

        :param line: first entry: point on line, second entry: direction vector
        :return: reflected direction vector
        """
        plane = np.stack([self.parametric[0], np.cross(self.parametric[1], self.parametric[2])])
        mirror_point_plane = self.plane_line_intersect(plane, line[0], plane[1])
        mirror_point = line[0] + 2 * (mirror_point_plane - line[0])
        reflection_point_plane = self.plane_line_intersect(plane, line[0], line[1])
        reflection_point = mirror_point + 2 * (reflection_point_plane - mirror_point)
        return (reflection_point - reflection_point_plane)[0]

    def get_intensity(self, incoming, apply_noise: bool = False):
        cross_product = np.cross(self.parametric[1], self.parametric[2])
        norm_cross_product = cross_product / np.sqrt(np.dot(cross_product, cross_product))
        sample_plane_in_normal = np.stack([self.parametric[0], norm_cross_product])
        intersect = self.plane_line_intersect(sample_plane_in_normal, incoming.beam_profile.coords,
                                              incoming.beam_profile.dv)
        intersect_new_coordinates = intersect - self.parametric[0]
        # Projection 1:
        fac1 = self.parametric[1] / np.dot(self.parametric[1], self.parametric[1])
        ca = (np.dot(intersect_new_coordinates, self.parametric[1]) * fac1.reshape(-1, 1)).T
        cond1 = inner1d(ca, ca) <= inner1d(self.parametric[1], self.parametric[1])
        cond2 = inner1d(ca, self.parametric[1]) >= 0

        # Projection 2:
        fac2 = self.parametric[2] / np.dot(self.parametric[2], self.parametric[2])
        cb = (np.dot(intersect_new_coordinates, self.parametric[2]) * fac2.reshape(-1, 1)).T
        cond3 = inner1d(cb, cb) <= inner1d(self.parametric[2], self.parametric[2])
        cond4 = inner1d(cb, self.parametric[2]) >= 0

        cond_together = cond1 & cond2 & cond3 & cond4
        indices = np.where(cond_together)
        self.beam_profile.coords = intersect[indices]
        self.beam_profile.dv = self.reflection(np.stack([incoming.parametric[0], incoming.beam_profile.dv]))
        if apply_noise:
            noise_level = 0.0001
            self.beam_profile.intensity = np.random.poisson(incoming.beam_profile.intensity[indices] / noise_level) * noise_level
        else:
            self.beam_profile.intensity = incoming.beam_profile.intensity[indices]
