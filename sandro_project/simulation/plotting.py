import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from ml.device import device
from scene import Scene
from simulation_parameters import DETECTION_DISTANCE, DETECTION_ANGLE, INC_PROPS, SAMPLE_PROPS, DET_PROPS


def set_color(inc, ax):
    inc_pixel_intensities = inc.beam_profile.intensity / max(inc.beam_profile.intensity)
    color_map = cm.get_cmap("inferno")
    ax.scatter(inc.beam_profile.coords.T[0].reshape(-1, len(inc.beam_profile.coords))[0],
               inc.beam_profile.coords.T[1].reshape(-1, len(inc.beam_profile.coords))[0],
               inc.beam_profile.coords.T[2].reshape(-1, len(inc.beam_profile.coords))[0],
               c=color_map(inc_pixel_intensities))


def setup_plot(*args):
    fig = plt.figure(figsize=(24, 12))
    ax = Axes3D(fig)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel("x", fontsize=20, linespacing=10)
    ax.set_ylabel("y", fontsize=20)
    ax.set_zlabel("z", fontsize=20)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.zaxis.set_tick_params(labelsize=10)
    for el in args:
        set_color(el, ax)
    ax.dist = 11
    fig.savefig(os.path.dirname(__file__) + "\\..\\plots\\setup.png")
    #plt.show()


def scan_plot(inc_props, sample_props, det_props, detection_angle, apply_noise, rotation: bool):
    misalignment = torch.tensor([0.000008, -0.000002, -0.000006, 0.000005, 0.000004], device=device, dtype=torch.float)
    names = ["chi", "omega", "x", "y", "z"]
    colors = ["g", "c", "r", "yellow", "b"]
    path = os.path.dirname(__file__) + "\\..\\plots\\"
    plt.rcParams['axes.linewidth'] = 1.5
    plt.figure(figsize=(16, 12))
    plt.ylabel("$\\frac{I_{det}}{I_{inc}}$", fontsize=45)
    plt.xlabel("rotation in °", fontsize=35)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tick_params(width=3, length=10)
    plt.tight_layout()

    if rotation:
        x_plot = np.linspace(-10, 10, 148)
        rng = range(2)
        name = f"rotation_scan"
    else:
        x_plot = np.linspace(-2.5, 2.5, 201)
        if detection_angle == 0:
            rng = range(3, 5)
        else:
            rng = range(2, 5)
        name = f"translation_scan"

    for var in rng:
        misalignment_vary = deepcopy(misalignment)
        y_plot = []
        for el in x_plot:
            misalignment_vary[var] = el
            scene = Scene(inc_props, sample_props, det_props, misalignment_vary, detection_angle, apply_noise, DETECTION_DISTANCE)
            y_plot.append(np.sum(scene.detector.beam_profile.intensity))
        if detection_angle == 0 and var == 3:
            plt.plot(x_plot, y_plot, "orange", label=names[var - 1] + ", " + names[var], linewidth=3, )
        else:
            plt.plot(x_plot, y_plot, colors[var], label=names[var], linewidth=3)
    plt.legend(loc="center right", prop={"size": 20})
    if detection_angle == 0:
        name = "shadow_" + name
    else:
        name = f"theta_{detection_angle}_" + name
    plt.savefig(path + name + ".png")


if __name__ == "__main__":
    #scene = Scene(INC_PROPS, SAMPLE_PROPS, DET_PROPS, torch.tensor([0, 0, 0, 0, 0], device=device, dtype=torch.float),
    #              theta=DETECTION_ANGLE, apply_noise=False)
    #setup_plot(scene.incoming, scene.sample, scene.detector)
    scan_plot(INC_PROPS, SAMPLE_PROPS, DET_PROPS, DETECTION_ANGLE, apply_noise=False, rotation=True)
    scan_plot(INC_PROPS, SAMPLE_PROPS, DET_PROPS, DETECTION_ANGLE, apply_noise=False, rotation=False)
    scan_plot(INC_PROPS, SAMPLE_PROPS, DET_PROPS, 0, apply_noise=False, rotation=True)
    scan_plot(INC_PROPS, SAMPLE_PROPS, DET_PROPS, 0, apply_noise=False, rotation=False)
