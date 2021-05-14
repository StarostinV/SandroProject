# %%
import os
import torch

from sandro_project import (
    Beam,
    Agent,
    Linear_QNet,
)
from sandro_project.ml.device import device

# Properties of Incoming Beam, Sample and Detector:
DETECTION_ANGLE = 0  # shadow scan - much faster

# IncomingBeam:
inc_length = 2
inc_width = 2
inc_beam_shape = ("round", 0.4, 0.4)
inc_intensity = 1
gauss = True
INC_PROPS = (inc_length, inc_width, inc_beam_shape, inc_intensity, gauss)

# Sample:
sample_length = 8
sample_width = 8
sample_thickness = 0
SAMPLE_PROPS = (sample_length, sample_width, sample_thickness)

# Detector:
detector_length = 2
detector_width = 2
DET_PROPS = (detector_length, detector_width)

"""
MOTOR_RANGES: Defines the maximal motor position of each motor in positive as well as in negative direction.
"""

CHI_RANGE = 1  # -1 .. 1
OMEGA_RANGE = 1  # -1 .. 1
X_RANGE = 0.1  # -0.1 .. 0.1
Y_RANGE = 0.1
Z_RANGE = 0.1
MOTOR_RANGES = torch.tensor([CHI_RANGE, OMEGA_RANGE, X_RANGE, Y_RANGE, Z_RANGE], device=device)

"""
Hyper Parameters:
NUM_INPUTS: Number of input positions of the neural network.
            Every position consists of 6 states (chi, omega, x, y, z, intensity),
            so the overall input size is 6 * NUM_INPUTS.
NUM_NETWORKS: Number of networks per generation.
SAMPLES_PER_GENERATION: Defines the amount of tests for each network. Every test gives one loss as output.
                        All these losses are summed up and are divided by SAMPLES_PER_GENERATION,
                        to get an average loss for each network.
NUM_BEST_MODELS: Defines the number of networks which are used as parents for the next generation.
"""
NUM_INPUTS = 100
NUM_NETWORKS = 20
SAMPLES_PER_GENERATION = 20
NUM_BEST_MODELS = 4

assert SAMPLES_PER_GENERATION * NUM_INPUTS * NUM_NETWORKS == 40000
# time_per_simulation = 0.001 sec
# time_per_generation = 40000 * 0.001 sec = 40 sec


# %%

agent = Agent(
    INC_PROPS,
    SAMPLE_PROPS,
    DET_PROPS,
    DETECTION_ANGLE,
    MOTOR_RANGES,
    NUM_NETWORKS,
    NUM_INPUTS,
    SAMPLES_PER_GENERATION,
    NUM_BEST_MODELS,
    load=False)

counter = 0

while True:
    start_positions = agent.generate_start_positions()
    counter += 1
    losses = agent.evaluate_models(start_positions)
    agent.generate_new_models(losses)
    if counter % 1 == 0:
        models_sorted = [x for _, x in sorted(zip(losses, agent.models))]
        best_model = models_sorted[0]
        best_model.save()
        with open(os.path.abspath('') + "/saves/plot_file.txt", "a") as file:
            file.write(str(counter) + " " + str(sum(losses) / NUM_NETWORKS) + "\n")
