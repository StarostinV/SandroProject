import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from scene import Scene
import os
import matplotlib.pylab as pylab
from simulation_parameters import DETECTION_ANGLE, INC_PROPS, SAMPLE_PROPS, DET_PROPS
from ml.device import device


def set_color(inc, ax):
    inc_pixel_intensities = inc.beam_profile.intensity / max(inc.beam_profile.intensity)
    color_map = cm.get_cmap("inferno")
    ax.scatter(inc.beam_profile.coords.T[0].reshape(-1, len(inc.beam_profile.coords))[0],
               inc.beam_profile.coords.T[1].reshape(-1, len(inc.beam_profile.coords))[0],
               inc.beam_profile.coords.T[2].reshape(-1, len(inc.beam_profile.coords))[0],
               c=color_map(inc_pixel_intensities))


def set_color_old(inc, ax):
    inc_pixel_intensities = inc.beam_profile.intensity
    inc_beam_spot1 = inc_pixel_intensities > 0
    inc_beam_spot2 = inc_pixel_intensities > 0
    inc_color = np.array(["blue"] * len(inc_pixel_intensities))
    inc_color[inc_beam_spot1] = "y"
    inc_color[inc_beam_spot2] = "red"
    ax.scatter(inc.beam_profile.coords.T[0].reshape(-1, len(inc.beam_profile.coords)),
               inc.beam_profile.coords.T[1].reshape(-1, len(inc.beam_profile.coords)),
               inc.beam_profile.coords.T[2].reshape(-1, len(inc.beam_profile.coords)),
               color=inc_color)


def setup_plot(*args):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    for el in args:
        set_color(el, ax)
    #fig.suptitle(f"Theta = 0°", fontsize=20)
    #fig.savefig(os.path.dirname(__file__) + "\\..\\plots\\setup.png")
    plt.show()


def scan_plot(inc_props, sample_props, det_props, detection_angle, apply_noise, var):
    misalignment = torch.tensor([0.000008, -0.000002, -0.000006, 0.000005, 0.000004], device=device, dtype=torch.float)
    names = ["chi", "omega", "x", "y", "z"]
    y_plot = []
    if var < 2:
        x_plot = torch.linspace(-10, 10, 148)
    else:
        x_plot = torch.linspace(-5, 5, 201)
    for el in x_plot:
        misalignment[var] = el
        scene = Scene(inc_props, sample_props, det_props, misalignment, detection_angle, apply_noise)
        y_plot.append(np.sum(scene.detector.beam_profile.intensity))

    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(bottom=0.15)
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (20, 10),
              'axes.labelsize': 30,
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)
    fig = plt.figure()
    fig.suptitle(f"Theta = {detection_angle}°", fontsize=30)
    plt.ylabel("$\\frac{I_{det}}{I_{inc}}$")
    plt.xlabel(names[var], fontsize=20)
    plt.plot(x_plot, y_plot)
    path = os.path.dirname(__file__) + "\\..\\plots\\"
    name = f"{names[var]}_scan"
    if detection_angle == 0:
        name = "shadow_" + name
    else:
        name = f"theta_{detection_angle}_" + name
    fig.savefig(path + name)


if __name__ == "__main__":
    scene = Scene(INC_PROPS, SAMPLE_PROPS, DET_PROPS, torch.tensor([0, 0, 0, 0, 0], device=device, dtype=torch.float),
                  theta=DETECTION_ANGLE, apply_noise=False)
    setup_plot(scene.incoming, scene.sample, scene.detector)
    for i in range(0, 5):
        #scan_plot(INC_PROPS, SAMPLE_PROPS, DET_PROPS, DETECTION_ANGLE, apply_noise=False, var=i)
        #scan_plot(INC_PROPS, SAMPLE_PROPS, DET_PROPS, 0, apply_noise=False, var=i)
        pass
