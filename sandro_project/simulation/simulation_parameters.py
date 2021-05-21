import torch
from ml.device import device

DETECTION_ANGLE = 20
DETECTION_DISTANCE = 10

# IncomingBeam:
inc_length = 2
inc_width = 2
inc_beam_shape = ("round", 0.6, 0.6)
inc_intensity = 1
gauss = True
INC_PROPS = (inc_length, inc_width, inc_beam_shape, inc_intensity, gauss)

# Sample:
sample_length = 10
sample_width = 10
sample_thickness = 0
SAMPLE_PROPS = (sample_length, sample_width, sample_thickness)

# Detector:
detector_length = 2
detector_width = 2
DET_PROPS = (detector_length, detector_width)

"""
MOTOR_RANGES: Defines the maximal motor position of each motor in positive as well as in negative direction.
"""

CHI_RANGE = 1
OMEGA_RANGE = 1
X_RANGE = 0.1
Y_RANGE = 0.1
Z_RANGE = 0.1
MOTOR_RANGES = torch.tensor([CHI_RANGE, OMEGA_RANGE, X_RANGE, Y_RANGE, Z_RANGE], device=device, requires_grad=False)
