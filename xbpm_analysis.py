#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Analysis of an X-ray Beam Position Monitor (XBPM) simulation.

Usage:
    ./xbpm_sim.py -h
    ./xbpm_sim.py -f <file1>,<file2> ... [-i <from .. to>]

where
     -h : this help message
     -f : the files to be analysed
     -i : interval of fitting in indexes
          (e.g, for a vector with 20 positions, -i 3 means from 7 to 13)

"""

from copy import deepcopy
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import re
import sys


def plot_positions(data, kdx=(1, 1), kdy=(1, 1), fromto=(7, 14), title=""):
    """Plot the positions calculated by the XBPM simulation."""
    fig, (axall, axcut, axrc) = plt.subplots(1, 3, figsize=(18, 5))

    # Get real and calculated positions for each direction from 'data'.
    real_x = data[:, 0]
    real_y = data[:, 1]
    calc_x = data[:, 2]
    calc_y = data[:, 3]

    # Fitting parameters.
    kx, dx = kdx
    ky, dy = kdy
    # Slice interval.
    fr, to = fromto

    # Calculate adjusted parameters.
    adj_x = calc_x * kx + dx
    adj_y = calc_y * ky + dy

    # Plot original and adjusted coordinates.
    axall.plot(real_x, real_y, 'bo')
    axall.plot(adj_x, adj_y, 'ro')

    # Reshaping parameter.
    resh = int(np.sqrt(real_x.shape[0]))

    # Reshape vector to rectangular array and get slices.
    rrs_x = real_x.reshape(resh, resh)
    rrs_y = real_y.reshape(resh, resh)

    rsl_x = rrs_x[fr:to, fr:to]
    rsl_y = rrs_y[fr:to, fr:to]

    # Reshape vectors and apply correction.
    adjresh_x = adj_x.reshape(resh, resh)
    adjresh_y = adj_y.reshape(resh, resh)
    #
    adjreshsl_x = adjresh_x[fr:to, fr:to]
    adjreshsl_y = adjresh_y[fr:to, fr:to]
    #
    axcut.plot(rsl_x, rsl_y, 'bo-')
    axcut.plot(adjreshsl_x, adjreshsl_y, 'ro-')
    stddevx, stddevy, stddevall = standard_deviation(rsl_x, rsl_y,
                                                     adjreshsl_x, adjreshsl_y,
                                                     title)

    # Select a line and compare real x calculated.
    lr = int(len(rrs_x) / 2)
    # Reference line.
    axrc.plot(rrs_x[lr, :], rrs_x[lr, :], 'b-')
    axrc.plot(rrs_x[lr, :], adjresh_x[lr, :], 'yo', label="X")
    axrc.plot(rrs_y[:, lr], adjresh_y[:, lr], 'go', label="Y")

    # Axes parameters.
    for ax in [axall, axcut, axrc]:
        ax.set_xlabel(u"$x$ [mm]")
        ax.set_ylabel(u"$y$ [mm]")
        ax.grid()
        ax.margins(0.1)
        ax.set_title(title)
        ax.set_aspect("equal")
    axrc.set_xlabel("real [mm]")
    axrc.set_ylabel("calculated [mm]")
    ax.set_title(title + f", cut at line {lr}")
    axrc.legend()
    axrc.margins(0.1)

    fig.tight_layout()
    fig.savefig(f"{title}-grid.png")
    # plt.tight_layout()
    # plt.show()

    return stddevx, stddevy, stddevall


def standard_deviation(realx, realy, adjx, adjy, title=""):
    """Average square distance between real and measured positions.

    Args:
        realx (numpy array) : real x positions;
        realy (numpy array) : real y positions;
        adjx (numpy array)  : adjusted measured x positions;
        adjy (numpy array)  : adjusted measured y positions.
        title (str) : graph title.

    Return:
        sqrt(varx) (float) : the standard deviation of distances over
            fitted points in horizontal direction.
        sqrt(vary) (float) : the standard deviation of distances over
            fitted points in vertical direction.
        stddev (float) : the standard deviation of distances over all
            fitted points.
    """
    diff_x = realx - adjx
    diff_y = realy - adjy
    diff_x_2 = diff_x * diff_x
    diff_y_2 = diff_y * diff_y
    nh, nv = realx.shape
    nsite = nh * nv
    diff2 = (diff_x_2 + diff_y_2) * 0.5

    figna, axna = plt.subplots(1)
    axna.set_xlabel(u"$x$ [mm]")
    axna.set_ylabel(u"$y$ [mm]")
    # axna.set_title("inaccuracy")
    axna.set_title(u"Local standard deviations [$\mu$m]")  # noqa: W605

    dcol = (realx[0, 1] - realx[0, 0])
    dlin = (realy[1, 0] - realy[0, 0])
    frcol, tcol = realx[0, 0], realx[0, -1]  # + dc
    frlin, tlin = realy[0, 0], realy[-1, 0]  # + dl

    xticks = [f"{x:.2f}" for x in np.arange(frcol, tcol+dcol, dcol)]
    yticks = [f"{x:.2f}" for x in np.arange(frlin, tlin+dlin, dlin)]
    try:
        axna.set_xticks(range(len(xticks)))
        axna.set_yticks(range(len(yticks)))
        axna.set_xticklabels(xticks)
        axna.set_yticklabels(yticks)
    except Exception as err:
        print(f" (std dev) WARNING: {err}")
    imna = axna.imshow(np.sqrt(diff2) * 1000,
                       cmap="viridis", origin="lower")
    figna.colorbar(imna)
    figna.tight_layout()
    figna.savefig(f"{title}-inaccuracy.png")

    varx = np.sum(diff_x_2) / nsite
    vary = np.sum(diff_y_2) / nsite
    stddev = np.sqrt((varx + vary)/2)
    return np.sqrt(varx), np.sqrt(vary), stddev


"""Functions to solve the linear system of adjustment coefficients."""


def chi_square(real, calc, kk, delta):
    """Chi-square estimate from data and fitted parameters."""
    norm = 1. / (real.shape[0] - 1)
    return norm * np.sum(np.square(real - kk * calc - delta))


def correction_coefficients(real, calc, fromto=(0, 10)):
    """Calculate the coefficients to correct the positions of measured data.

    This function slices and reshapes 'data' to extract the central portion
    of the positions, where there are some linear correspondence between real
    and measured (simulated) data, then it calculates the coefficients K and
    delta  of the linear fitting between both sets and return them.
    """
    # Reshaping parameter.
    resh = int(np.sqrt(real.shape[0]))

    # Reshape data to rectangular array.
    realresh = real.reshape(resh, resh)
    calcresh = calc.reshape(resh, resh)

    # Get slices and reshape back to 1-d matrices.
    fr, to = fromto
    realslice = realresh[fr:to, fr:to].reshape(-1)
    calcslice = calcresh[fr:to, fr:to].reshape(-1)

    # Rescale array for weighting.
    wslice = np.abs(deepcopy(realslice))
    wmin, wmax = np.min(wslice), np.max(wslice)
    # Avoid infinities in weighing by adding an offset.
    offset = (wmax - wmin) / (np.max(wslice.shape))
    weight = 1. / (wslice + offset)
    # weight = 1. / (np.sqrt(wslice) + offset)

    # Fit data with heavier weight at the central region.
    # weight = 1. / np.sqrt(np.abs(calcslice))
    try:
        pf = np.polyfit(calcslice, realslice, deg=1, w=weight, cov=True)
    except Exception as err:
        weight = np.ones(len(calcslice))
        print(f" WARNING: {err}.\n Fitting weight set to 1.")
        pf = np.polyfit(calcslice, realslice, deg=1, w=weight, cov=True)

    (kk, delta), cov = pf
    chi2 = chi_square(realslice, calcslice, kk, delta)

    return kk, delta, cov, chi2


def xbpm_fit(data, fromto=(7, 14), title="XBPM positions"):
    """Calculate the Kx and Ky coefficients to correct XBPM data and plot it.

    Args:
        data (numpy array): the defined and calculated (by simulation) data
        fromto (tuple): the interval where to fit data
        step (int): index interval step to reshape data for plotting;
        title (string): graph title.
        fitrange (int) : the range of indexes from the midpoint of the lattice.

    Return:
        stddevx (float) : the standard deviation of distances over
            fitted points in horizontal direction.
        stddevy (float) : the standard deviation of distances over
            fitted points in vertical direction.
        stddevall (float) : the standard deviation of distances over all
            fitted points.

    """
    # Get real and calculated positions for each direction from 'data'.
    real_x = data[:, 0]
    real_y = data[:, 1]
    #
    calc_x = data[:, 2]
    calc_y = data[:, 3]

    print("\n\n##### Calculated coefficients "
          "(first degree polynomial fitting, K . x + delta):")

    # Calculate correction coefficients and plot results.
    # Horizontal.
    kx, deltax, covx, chi2x = correction_coefficients(real_x, calc_x,
                                                       fromto=fromto)
    print(f"\n Kx     = {kx:.6f}  ({np.sqrt(covx[0, 0]):.1e}), "
        f" deltax = {deltax:.6f}  ({np.sqrt(covx[1, 1]):.1e}), "
        f"\t chi2 = {chi2x:.4g}"
        f"\n Covariance =\n{covx}")

    # Vertical.
    ky, deltay, covy, chi2y = correction_coefficients(real_y, calc_y,
                                                      fromto=fromto)
    print(f"\n Ky     = {ky:.6f}  ({np.sqrt(covy[0, 0]):.1e}), "
        f" deltay = {deltay:.6f}  ({np.sqrt(covy[1, 1]):.1e}), "
        f"\t chi2 = {chi2y:.4g}"
        f"\n Covariance =\n{covy}")

    # Plot the real and measured positions as a grid.
    stddevx, stddevy, stddevall = plot_positions(data,
                                                 (kx, deltax), (ky, deltay),
                                                 fromto,
                                                 title=title)
    print("\n\n# Standard deviations (um):\n"
          f"    horizontal: {stddevx * 1000:.2f}\n"
          f"    vertical:   {stddevy * 1000:.2f}\n"
          f"    overall:    {stddevall * 1000:.2f}\n")


def data_read():
    """Read command line options and files.

    This function is intentionally thin: it delegates parsing and file
    reading to small helpers so the logic is easier to test and maintain.

    Returns:
        (data, fitrange) where data is a list of FileData objects.
    """
    files, fitrange = parse_options()
    data = read_data_files(files)
    return data, fitrange


@dataclass
class FileData:
    """Container for a read file and its display title."""
    array: np.ndarray
    title: str


def parse_options(argv=None):
    """Parse command line options and return (files, fitrange).

    Uses argparse for clearer help messages and validation. If argv is
    provided it should be a list of arguments (for testing), otherwise
    sys.argv[1:] is used.
    """
    if argv is None:
        argv = None  # argparse will default to sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="xbpm_analysis",
        description=(
            "Analyze XBPM simulation output files and "
            "compute correction coefficients"
        ),
    )
    parser.add_argument('-f', '--files', required=True,
                        help='Comma-separated list of files to analyse')
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=None,
        help=(
            'Interval of fitting in indexes (e.g. for 20 positions, '
            '-i 3 means from 7 to 13)'
        ),
    )

    args = parser.parse_args(argv)

    files = args.files.split(',') if args.files else []
    fitrange = args.interval

    return files, fitrange


def read_data_files(files):
    """Read the provided files and build the data list.

    Each entry is [numpy_array, title]. Any file read error will print
    an error and exit, matching previous behavior.
    """
    data = []
    basetitle = "XBPM positions"

    for file in files:
        tp = re.search("cross|pair", file)
        if tp is None:
            title = basetitle
        else:
            title = basetitle + ", " + tp.group()
            if tp.group() == "pair":
                title += "wise blades"
        try:
            data.append(FileData(np.genfromtxt(file), title))
        except Exception as err:
            print(f"ERROR reading file: {err}")
            sys.exit(1)

    return data


def flux_plot(data):
    """Plot data flux.

    Args:
        data (numpy array) : blades' fluxes.
    """
    rx = data[:, 0]
    ry = data[:, 1]
    bo = data[:, 4]
    # bi = data[:, 5]
    # ti = data[:, 6]
    # to = data[:, 7]

    fig, ax = plt.subplots(2)
    ax[0].plot(rx, bo, 'bo', label="BI vs x")
    ax[1].plot(ry, bo, 'go-', label="BI vs y")
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()


def main():
    """Read data from files, plot and fit calibration for XBPM simulation."""
    data, fitrange = data_read()

    # Data square size, in index value.
    step = int(np.sqrt(data[0].array.shape[0]))

    # Default value for fitting interval if not provided.
    nx = int(step / 2)
    if fitrange is None:
        fitrange = int(nx / 4)
    fromto = (nx - fitrange, nx + fitrange + 1)

    for fd in data:
        xbpm_fit(fd.array, fromto=fromto, title=fd.title)

    if len(data[0].array) > 4:
        flux_plot(data[0].array)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
