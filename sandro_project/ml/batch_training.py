from typing import List

import torch
from torch import Tensor

from ..simulation.batch_plane import BatchPlane
from ..simulation import Detector, Incoming


def update_inputs(prev_inputs, intensities, new_positions) -> Tensor:
    """
    Add another position, another intensity, + normalize everything.
    :param prev_inputs:
    :param intensities:
    :param new_positions:
    :return:
    """


def train_epoch(models: List[torch.nn.Module]):
    incoming = Incoming(1, 1, ...)
    samples = BatchPlane()
    detector = Detector(1, 1, 0)
    num_of_steps = 100
    inputs = []  # len(models) = 32

    # inputs = init_inputs(inputs)

    for i in range(num_of_steps):
        new_positions: List[Tensor] = [model(env) for env, model in zip(inputs, models)]
        positions_tensor: Tensor = torch.stack(new_positions)
        samples.set_positions(positions_tensor)

        intensities: Tensor = detector.new_shadow_scan(incoming, samples, True)

        inputs = update_inputs(inputs, intensities, new_positions)

    return models
