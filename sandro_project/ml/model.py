import os

import numpy as np

import torch
from torch import nn

from .device import device
from ..simulation import Scene


class Linear_QNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = nn.Linear(6 * int(input_size), 64)
        self.linear2 = nn.Linear(64, 5)
        self.requires_grad_(False)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x)).to(device)
        x = torch.sigmoid(self.linear2(x)).to(device)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./saves"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def fill(self, pos1, pos2, agent):
        """

        :param pos1: first input state of neural network
        :param pos2: second input state of neural network
        :param agent: current agent
        :return: The denormalized motor positions, predicted by the network
        """
        x = torch.zeros(agent.num_inputs, 6, requires_grad=False, device=device)
        if pos1[-1] > pos2[-1]:
            max_intensity = pos1[-1]
            pos1[-1] = 1
            pos2[-1] = pos2[-1] / max_intensity
        else:
            max_intensity = pos2[-1]
            pos2[-1] = 1
            pos1[-1] = pos1[-1] / max_intensity

        x[0] = pos1
        x[1] = pos2
        for i in range(2, agent.num_inputs):
            normalized_action = self(x.flatten())
            true_action = agent.change_motor_pos(normalized_action, agent.motor_ranges, normalize=False)

            scene = Scene(*agent.props, agent.misalignment + true_action, agent.detection_angle)
            intensity = torch.tensor([np.sum(scene.detector.beam_profile.intensity).item()]).detach().to(device)
            result = torch.cat((normalized_action, intensity), dim=0).detach().to(device)

            if result[-1] > max_intensity:
                ratio = max_intensity / result[-1]
                x[:, -1] *= ratio
                max_intensity = result[-1]
                result[-1] = 1
            else:
                result[-1] = result[-1] / max_intensity
            x[i] = result

        return agent.change_motor_pos(self(x.flatten()), agent.motor_ranges, normalize=False)
