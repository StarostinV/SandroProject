import os
import random
from copy import deepcopy

import numpy as np

import torch

from ml.model import Linear_QNet
from ml.device import device

from simulation.scene import Scene


class Agent(object):
    def __init__(self,
                 inc_props,
                 sample_props,
                 det_props,
                 detection_angle,
                 motor_ranges,
                 num_networks,
                 num_inputs,
                 samples_per_generation,
                 num_best_models,
                 load):

        self.props = (inc_props, sample_props, det_props)
        self.detection_angle = detection_angle
        self.motor_ranges = motor_ranges
        self.misalignment = self.change_motor_pos(self.random_values(5), normalize=False)
        self.spg = samples_per_generation
        self.num_inputs = num_inputs
        self.num_best_models = num_best_models
        self.num_networks = num_networks

        models = []
        if load:
            saved_model = Linear_QNet(num_inputs).cuda()
            saved_model.load_state_dict(
                torch.load(os.path.dirname(__file__) + "\\saves\\model.pth")
            )
            saved_model.eval()
            models.append(saved_model)
            for i in range(1, num_networks):
                model = self.mutate(saved_model)
                models.append(model)
        else:
            for i in range(num_networks):
                model = Linear_QNet(num_inputs).cuda()
                models.append(model)
        self.models = models

    def generate_start_positions(self):
        """
        First 2 measured points

        :return: torch tensor with shape(samples_per_generation, 2, 6)
        """
        states = None

        for i in range(self.spg):
            misalignment = self.change_motor_pos(self.random_values(5), normalize=False)
            self.misalignment = misalignment
            pos1, pos2 = self.random_values(), self.random_values()
            pos1_true = self.change_motor_pos(pos1, normalize=False)
            pos2_true = self.change_motor_pos(pos2, normalize=False)
            scene1 = Scene(*self.props, misalignment - pos1_true, self.detection_angle)
            scene2 = Scene(*self.props, misalignment - pos2_true, self.detection_angle)
            intensity1 = np.sum(scene1.detector.beam_profile.intensity)
            intensity2 = np.sum(scene2.detector.beam_profile.intensity)
            state1 = np.append(pos1.cpu().numpy(), intensity1)
            state2 = np.append(pos2.cpu().numpy(), intensity2)
            if i == 0:
                states = (np.stack([state1, state2]))
            else:
                states = np.concatenate((states, np.stack([state1, state2])), axis=0)
        return torch.from_numpy(states.reshape((self.spg, 2, 6))).detach().to(device)

    def evaluate_models(self, start_positions):
        """

        :param start_positions: all start positions for this generation
        :return: losses for each network
        """
        losses = []
        count = 0
        for model in self.models:
            count += 1
            print(count)
            loss = 0.0
            for i in range(len(start_positions)):
                pred = model.fill(start_positions[i][0], start_positions[i][1], self)
                loss += torch.norm(pred + self.misalignment).item()
            losses.append(loss / len(start_positions))
            print(loss / len(start_positions))
        return losses

    def generate_new_models(self, losses: list, gen_number: int):
        """

        :param losses: losses for each network
        :return: Set the new generation of models
        """
        models_sorted = [x for _, x in sorted(zip(losses, self.models))]
        new_models = []
        modulo = (self.num_networks - self.num_best_models) % self.num_best_models
        for i in range(self.num_best_models):
            new_models.append(models_sorted[i])
            for j in range(int((self.num_networks - self.num_best_models) / self.num_best_models)):
                new_model = self.mutate(models_sorted[i], gen_number)
                new_models.append(new_model)
            if i < modulo:
                new_model = self.mutate(models_sorted[i], gen_number)
                new_models.append(new_model)
        self.models = new_models

    @staticmethod
    def mutate(model, gen_number: int):
        """

        :param model: neural network
        :param gen_number: current generation number
        :return: mutated neural network
        """
        if gen_number < 200:
            mutation_power = 0.1 - (int(gen_number / 20) * 0.01)
        else:
            mutation_power = 0.01

        new_model = deepcopy(model)
        for param in new_model.parameters():
            param.data += mutation_power * torch.randn_like(param)
        return new_model

    @staticmethod
    def random_values(num_values: int = 5):
        """

        :param num_values: number of random values that should be generated
        :return: torch float tensor with random values
        """
        lst = []
        for i in range(num_values):
            lst.append(random.uniform(0, 1))
        return torch.tensor(lst, device=device, dtype=torch.float)

    def change_motor_pos(self, current_positions, normalize: bool):
        """

        :param current_positions: current motor positions
        :param normalize: True, if you want to normalize; False if you want to "denormalize"
        :return: List with normalized (or denormalized) positions
        """
        motor_maxima = self.motor_ranges
        if len(current_positions) != len(motor_maxima):
            print("Number of positions and motor ranges do not fit")
        else:
            if normalize:
                normalized_positions = current_positions / (2 * motor_maxima) + 0.5
                return normalized_positions
            else:
                denormalized_positions = (current_positions - 0.5) * 2 * motor_maxima
                return denormalized_positions
