__author__ = "Garrett Dal Byrd"
__version__ = "0.0.1"
__email__ = "gbyrd4@vols.utk.edu"
__status__ = "Development"

from os import listdir, curdir, chdir
from os.path import isdir
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from numpy import genfromtxt
import h5py
import sys
from scipy.stats import beta
from scipy.optimize import curve_fit


def proportions_visualizer(seed, image_filetype="svg"):
    chdir("runs")
    chdir(seed)

    parameters = genfromtxt("parameters.csv", delimiter=",")[1]
    x_length = int(parameters[1])
    y_length = int(parameters[2])
    z_length = int(parameters[3])
    init_infection = int(parameters[4])
    local_prob = float(parameters[5])
    global_prob = float(parameters[6])
    run_length = int(parameters[7])
    run_quantity = int(parameters[8])
    population = x_length * y_length * z_length

    susceptible_total = np.asarray(
        [population - init_infection for _ in range(run_length)]
    )
    infected_total = np.asarray([init_infection for _ in range(run_length)])

    susceptible_frequencies = np.zeros([run_length, population])
    infected_frequencies = []

    for run in listdir(curdir):
        if isdir(run):
            chdir(run)
            with h5py.File("proportions.hdf5", "r") as file:
                proportions = np.asarray(file["default"])

            for step in range(run_length):
                susceptible_frequencies[step][int(proportions[step][0])] += 1

            susceptible_total = np.vstack(
                (
                    susceptible_total,
                    np.asarray([item[0] / population for item in proportions]),
                )
            )
            infected_total = np.vstack(
                (
                    infected_total,
                    np.asarray([item[1] / population for item in proportions]),
                )
            )

            plt.figure(1, figsize=(1.68, 1), facecolor="white")
            plt.subplot(211)
            plt.plot(
                range(run_length),
                [item[0] / population for item in proportions],
                color="grey",
            )
            plt.subplot(212)
            plt.plot(
                range(run_length),
                [item[1] / population for item in proportions],
                color="grey",
            )
            chdir("..")

    susceptible_median = np.median(susceptible_total, axis=0)
    infected_median = np.median(infected_total, axis=0)

    plt.subplot(211)
    plt.plot(range(run_length), susceptible_median, color="red")
    plt.plot(
        range(run_length),
        [0.5 for _ in range(run_length)],
        linestyle="--",
        color="blue",
    )
    plt.ylabel(f"Susceptible \n (Proportion of Agents)")
    plt.subplot(212)
    plt.plot(range(run_length), infected_median, color="red")
    plt.plot(
        range(run_length),
        [0.5 for _ in range(run_length)],
        linestyle="--",
        color="blue",
    )
    plt.xlabel("Time (Steps)")
    plt.ylabel("Infected \n (Proportion of Agents)")
    plt.savefig(f"{seed} Figure.{image_filetype}", facecolor="white")

    plt.savefig("median_graph.svg")

    chdir("..")

    plt.figure(dpi=1024)
    plt.imshow(
        np.transpose(susceptible_frequencies) / run_quantity,
        interpolation="none",
        vmin=0,
        vmax=0.05,
        origin="lower",
    )
    plt.ylabel(r"% of Susceptible Agents")
    plt.xlabel(r"Time (discrete steps)")
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999])
    plt.yticks([0, 100])
    plt.savefig(f"{seed} Chromatic Figure.{image_filetype}", facecolor="white")
