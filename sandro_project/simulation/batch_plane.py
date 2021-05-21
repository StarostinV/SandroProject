from torch import (
    Tensor,
    cos,
    sin,
    tensor
)


class BatchPlane(object):

    def set_positions(self, positions: Tensor):
        pass

    # API
    def rotate_x(self, x_angles: Tensor):
        """
        :param x_angles: an array of shape (N_batch, ).
        :return:
        """
        self.edges: Tensor
        matrix = get_x_rotation_matrix(x_angles)  # shape = (N_batch, 3, 3)
        self.edges = matrix.dot(self.edges)


def get_x_rotation_matrix(x_angles: Tensor) -> Tensor:
    """

    Consider using Quaternions instead.

    :param x_angles: (N_batch, )
    :return: (N_batch, 3, 3)
    """
    cs, sn = cos(x_angles), sin(x_angles)

    matrix = tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).repeat(x_angles.shape[0])

    matrix[:, 1, 1] = cs
    matrix[:, 1, 2] = -sn
    matrix[:, 2, 1] = sn
    matrix[:, 2, 2] = cs
    return matrix
