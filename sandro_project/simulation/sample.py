import numpy as np

from simulation.plane import RectPlane


class Sample(RectPlane):
    def __init__(self, length, width, thickness, det_angle):
        super().__init__(length, width, thickness)
        """self.distance_local_origin = distance_origin_sample + np.array([0, 0, thickness])
        self.edges += self.distance_local_origin
        self.rotate_y(det_angle / 2)
        self._actualize_parametric()"""
        self.edges += np.array([0, 0, thickness])
        self.rotate_y(det_angle / 2)
        self._actualize_parametric()

    def set_position(self, misalignment):
        """

        :param misalignment: Tuple(chi, omega, x, y, z)
        :return: Set the misaligned position of the sample
        """
        self.rotate_x(misalignment[0])
        self.rotate_y(misalignment[1])
        self.shift_x(misalignment[2])
        self.shift_y(misalignment[3])
        self.shift_z(misalignment[4])
