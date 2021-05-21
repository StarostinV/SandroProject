from copy import deepcopy

import numpy as np
from numpy.core.umath_tests import inner1d

from incoming import Incoming
from plane import RectPlane
from sample import Sample


# step 00 : fix problem with target misalignment
# step 0 (easiest): train with theta == 0.
# if slow, optimize code
# if still slow, np.array -> torch.tensor (device='cuda')
# if still slow, simulate batches of different positions for the same sample at the same time.
# vector (x, y, z) -> (batch, x, y, z)


class Detector(RectPlane):
    def __init__(self, length: float, width: float, det_angle: float, det_distance: float = 10.0):
        super().__init__(length, width)
        self.rotate_y(90)
        self.shift_x(-det_distance)
        self.rotate_y(det_angle)
        self._actualize_parametric()

    @staticmethod
    def is_point_left(coords, line):
        coords_diff = coords - line[0]
        coords_diff = (coords_diff.T * np.array([[1], [1], [-1]])).T
        dv2 = np.array([line[1][0], line[1][2], line[1][1]])
        return coords_diff.dot(dv2) < 0

    def new_shadow_scan(self, incoming: Incoming, sample: Sample, apply_noise: bool):
        """

        First condition (above the sample):

        x axis
        -
        *-----* <- either these edges have higher z
        -------
        -------
        -------
        -------
        *-----* -------> y axis  <- or these edges have higher z

        (y, z) projection

        (x1, y1, z1)
        *
           -
             -
                -
                   -
                      * (x2, y2, z2)

                *
               ---
              *------*


        z - z1 >= (y - y1) * (z2 - z1) / (y2 - y1)

        Second condition (within the detector):

        - detector_width / 2 <= y <= detector_width / 2

        - detector_height / 2 <= z <= detector_height / 2


        :param incoming:
        :param sample:
        :param apply_noise:
        :return:
        """
        if sample.edges[0, 2] < sample.edges[1, 2]:
            idx = [1, 2]
        else:
            idx = [0, 3]

        y1, y2 = sample.edges[idx, 1]
        z1, z2 = sample.edges[idx, 2]

        y, z = incoming.beam_profile.coords[:, 1], incoming.beam_profile.coords[:, 2]

        cond1 = (z - z1) >= (y - y1) * (z2 - z1) / (y2 - y1)

        # TODO: add the second condition

        self.beam_profile.coords = self.beam_profile.coords[cond1]

        if apply_noise:
            noise_level = 0.0001
            self.beam_profile.intensity = \
                np.random.poisson(incoming.beam_profile.intensity[cond1] / noise_level) * noise_level
        else:
            self.beam_profile.intensity = incoming.beam_profile.intensity[cond1]

    def shadow_scan(self, incoming: Incoming, sample: Sample, apply_noise: bool):
        """
            1. Project sample edges onto y-z-plane (x = 0) -> saved in y_z_edges
            2. Find edge index with highest z value -> highest_edge_idx
            3. Now I want to get the two neighbour edges of the considered edge,
               by looking at the previous and next place in y_z_edges
                - in the cases of A or D this would throw an Error (for example: looking at the previous edge of A would
                  end in looking at position -1, which is impossible)
                - Trick: So adding D at the front of the list and A again at the end, delivers the desired functions
                  -> help_edges
                - Note: highest_edge_idx has to be increased by 1, to get the correct position in help_edges
            4. Take the edge with the highest z value and check if one of the neighbour edges has the same height
                - True: Check if y value of this neighbour is also identical
                    - True: Take line of highest edge and other neighbour
                    - False: Take line of highest edge and this neighbour
                - False: One has to consider two lines, which define the shadowed region
        """
        y_z_edges = deepcopy(sample.edges)
        y_z_edges[:, 0] = 0
        high_edge = y_z_edges.T[2].argmax() + 1  # Index of the highest edge
        y_z_edges = np.stack([y_z_edges[3], *y_z_edges, y_z_edges[0]])  # Trick
        y = np.array([y_z_edges[high_edge - 1][1], y_z_edges[high_edge + 1][1]])
        line1 = np.stack([y_z_edges[high_edge - 1 + 2 * y.argmax()],
                          y_z_edges[high_edge] - y_z_edges[high_edge - 1 + 2 * y.argmax()]])
        line2 = np.stack([y_z_edges[high_edge], y_z_edges[high_edge - 1 + 2 * y.argmin()] - y_z_edges[high_edge]])
        cond1 = self.is_point_left(incoming.beam_profile.coords, line1)
        cond2 = self.is_point_left(incoming.beam_profile.coords, line2)
        cond = ~(cond1 & cond2)

        indices = np.where(cond)
        det_plane_in_normal = np.stack([self.parametric[0], np.cross(self.parametric[1], self.parametric[2])])
        intersect = self.plane_line_intersect(det_plane_in_normal, incoming.beam_profile.coords,
                                              incoming.beam_profile.dv)
        self.beam_profile.coords = intersect[indices]

        if apply_noise:
            noise_level = 0.0001
            self.beam_profile.intensity = \
                np.random.poisson(incoming.beam_profile.intensity[indices] / noise_level) * noise_level
        else:
            self.beam_profile.intensity = incoming.beam_profile.intensity[indices]
