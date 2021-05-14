from copy import deepcopy

import numpy as np

from .incoming import Incoming
from .plane import RectPlane
from .sample import Sample

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

    def shadow_scan(self, incoming: Incoming, sample: Sample, apply_noise: bool):
        """
        Unprofessional documentation:
            1. Project sample edges onto y-z-plane (x = 0) -> saved in y_z_edges
            2. Find edge index with highest z value -> highest_edge_idx
            3. Now I want to get the two neighbour edges of considered edge, by looking at the previous and next place in y_z_edges
                - in the cases of A or D this would throw an Error (for example: looking at the previous edge of A would end in looking at position -1, which si impossible)
                - Trick: So adding D at the front of the list and A again at the end, delivers the desired functions -> help_edges
                - Note: highest_edge_idx has to be increased by 1, to get the correct position in help_edges
            4. Take the edge with the highest z value and check if one of the neighbour edges has the same height
                - True: Check if y value of this neighbour is also identical
                    - True: Take line of highest edge and other neighbour
                    - False: Take line of highest edge and this neighbour
                - False: One has to consider two lines, which define the shadowed region
        """
        y_z_edges = sample.edges
        for i in range(len(y_z_edges)):
            y_z_edges[i] = np.array([0, y_z_edges[i][1], y_z_edges[i][2]])
        highest_edge_idx = y_z_edges.T[2].argmax() + 1
        help_edges = np.stack([y_z_edges[3], *y_z_edges, y_z_edges[0]])
        if help_edges[highest_edge_idx][2] == help_edges[highest_edge_idx - 1][2] or help_edges[highest_edge_idx][2] == \
                help_edges[highest_edge_idx + 1][2]:
            if help_edges[highest_edge_idx][2] == help_edges[highest_edge_idx - 1][2] and help_edges[highest_edge_idx][
                1] != help_edges[highest_edge_idx - 1][1]:
                if help_edges[highest_edge_idx][1] < help_edges[highest_edge_idx - 1][1]:
                    line1 = np.stack([help_edges[highest_edge_idx - 1],
                                      help_edges[highest_edge_idx] - help_edges[highest_edge_idx - 1]])
                    line2 = deepcopy(line1)
                elif help_edges[highest_edge_idx][1] > help_edges[highest_edge_idx - 1][1]:
                    line1 = np.stack(
                        [help_edges[highest_edge_idx], help_edges[highest_edge_idx - 1] - help_edges[highest_edge_idx]])
                    line2 = deepcopy(line1)
                else:
                    line1 = np.stack([help_edges[highest_edge_idx + 1],
                                      help_edges[highest_edge_idx] - help_edges[highest_edge_idx + 1]])
                    line2 = deepcopy(line1)
            else:
                if help_edges[highest_edge_idx][1] < help_edges[highest_edge_idx + 1][1]:
                    line1 = np.stack([help_edges[highest_edge_idx + 1],
                                      help_edges[highest_edge_idx] - help_edges[highest_edge_idx + 1]])
                    line2 = deepcopy(line1)
                elif help_edges[highest_edge_idx][1] > help_edges[highest_edge_idx + 1][1]:
                    line1 = np.stack(
                        [help_edges[highest_edge_idx], help_edges[highest_edge_idx + 1] - help_edges[highest_edge_idx]])
                    line2 = deepcopy(line1)
                else:
                    line1 = np.stack([help_edges[highest_edge_idx - 1],
                                      help_edges[highest_edge_idx] - help_edges[highest_edge_idx - 1]])
                    line2 = deepcopy(line1)
        else:
            if help_edges[highest_edge_idx - 1][1] < help_edges[highest_edge_idx + 1][1]:
                line1 = np.stack(
                    [help_edges[highest_edge_idx + 1], help_edges[highest_edge_idx] - help_edges[highest_edge_idx + 1]])
                line2 = np.stack(
                    [help_edges[highest_edge_idx], help_edges[highest_edge_idx - 1] - help_edges[highest_edge_idx]])
            else:
                line1 = np.stack(
                    [help_edges[highest_edge_idx - 1], help_edges[highest_edge_idx] - help_edges[highest_edge_idx - 1]])
                line2 = np.stack(
                    [help_edges[highest_edge_idx], help_edges[highest_edge_idx + 1] - help_edges[highest_edge_idx]])
        cond1 = self.is_point_left(incoming.beam_profile.coords, line1)
        cond2 = self.is_point_left(incoming.beam_profile.coords, line2)
        cond = cond1 & cond2
        cond_reverse = cond == False
        indices = np.where(cond_reverse)
        det_plane_in_normal = np.stack([self.parametric[0], np.cross(self.parametric[1], self.parametric[2])])
        intersect = self.plane_line_intersect(det_plane_in_normal, incoming.beam_profile.coords,
                                              incoming.beam_profile.dv)
        self.beam_profile.coords = intersect[indices]
        if apply_noise:
            noise_level = 0.0001
            self.beam_profile.intensity = np.random.poisson(
                incoming.beam_profile.intensity[indices] / noise_level) * noise_level
        else:
            self.beam_profile.intensity = incoming.beam_profile.intensity[indices]
