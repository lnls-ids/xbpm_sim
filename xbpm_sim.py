#!/bin/env python3
# -*- coding: utf-8 -*-

"""X-ray Beam Position Monitor (XBPM) simulation.

Simulate the incidence of a gaussian-like beam on a
X-ray Beam Position Monitor (XBPM).

Usage:
    ./xbpm_sim.py [-h][-i] -p <parameter file>

where
     -h : this help message
     -i : interactive, the user can move the beam
          up/down, left/right, using the keyboard arrows.
     -p <parameter file>    : the file which defines the parameters of the
          simulation, as the number of random points in the beam, its shape,
          the format of blades and their geometry etc.
     -d <distribution file> : import an externally generated distribution
           in numpy array file format. It allows for X-ray profiles simulated
           from bend magnets or undulators, or registered from DVF cameras.

The default is non-interactive, meaning a sweep is made over the
defined window. Length units are in mm; angles, in degrees.

The parameters (to be defined in the parameter file) are:

ngauss (int) : number of gaussian distributions to be superimposed.
           Emulate distortions in the beam by adding gaussian distributions;
           the mean and variances of each are randomly defined.

nsample (int) : number of gaussian random samples (2-d coordinates) to
           represent the incident 'photons' in each frame;

pixelsize : the length resolution of the system;

 - Blades' geometry.

windowsize [(float), (float)] : size of the area where blades are defined
           (related to the Cu mask);

bladelength, bladethickness (float): blade dimensions, length and thickness;

corneroffset (float) : distance between blade and box corner (horizontal);

phi (float) : azimuthal angle of the blades in the Box;

 - Simulation

nsweeps (int) : number equidistant measurements in each direction
           inside the box;

sweepinterval ([(float), (float)]) : interval inside the box for sweeping
           (a centralized square).

fwhm_x, fwhm_y (float) : beam width (FWHM);

thetadeg (float) : blade angle upon which photons incide, in deg;

addring (True/False):  should the simulation consider a gaussian 'ring'
           to be added to the gaussian distributions;

mean ([(float), (float)]) : the standard mean;

cxy, cyx (float) : crossed covariances.

histupdate (True/False) : do update histogram image while sweeping.

registerflux (True/False) : record the fluxes on the blades and write them to
           file; usefull for further adjusts in position calculations.

"""

# from networkx import bellman_ford_path_length
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import random as rnd

from datetime import datetime
from copy import deepcopy
import matplotlib.animation as mpanim
from functools import partial
from pynput.keyboard import Key, Listener
import getopt
import re
import sys

bibdir = "./"
sys.path.append(bibdir)
from positioncalc import BeamPosition as BmP  # type: ignore # noqa: F401, E402
from blademask import BladeMask               # type: ignore # noqa: E402

# Initialize random number generator.
rng = np.random.default_rng(seed=None)

# Ratio  in sigma/FWHM in a gaussian distribution.
FWHM2SIGMA = 1.0 / np.sqrt(np.log(256))


def on_press(key: Key, gprm: dict, step: float) -> None:
    """Update the mean of the gaussians on pressing the keyboard arrows."""
    for ng in range(gprm["ngauss"]):
        mean = gprm[ng]["mean"]
        if key == Key.left:
            mean[0] -= step
        if key == Key.right:
            mean[0] += +step
        if key == Key.up:
            mean[1] += step
        if key == Key.down:
            mean[1] -= step


def on_release(key: Key) -> bool:
    """What to do if a key is released.

    Args:
        key (Key): the pressed key.

    Returns:
        False
    """
    if key == Key.esc:
        return False


def mean_shift(mean: np.ndarray) -> None:
    """Define the centroid and the dispersion of the distributions.

    Set the mean in x (0) and y (directions) of gaussian distributions
    by shifting the original mean and variance by some random amount.
    """
    mean[0] += 0.5 + 0.2 * (rnd.random() - 0.5)  # noqa: S311
    mean[1] += 0.5 * rnd.random()  # noqa: S311
    return


def cov_shift(cov: np.ndarray) -> None:
    """Shift the covariance by some random amount."""
    shift = 1.0
    cov[0, 0] = np.abs(cov[0, 0] + shift * (rnd.random() - 0.5))  # noqa: S311
    cov[0, 1] = np.abs(cov[0, 1] + shift * (rnd.random() - 0.5))  # noqa: S311
    cov[1, 0] = cov[0, 1]
    cov[1, 1] = np.abs(cov[1, 1] + shift * (rnd.random() - 0.5))  # noqa: S311


def histogram_parameters_set(gprm: dict,
                             mean: np.ndarray,
                             cov: np.ndarray) -> None:
    """Set mean, covariance and number of samples for each histogram.

    Args:
        gprm (dict): general parameters of the simulation;
        mean (numpy array): the mean of the distribution in x and y;
        cov (numpy array): the covariance of the distribution in x and y;
    """
    for ng in range(gprm["ngauss"]):
        # Dictionary for gaussian parameters: mean, covariance and number
        # of samples.
        gh = {
            "mean"        : mean,
            "cov"         : cov,
            "nsample"     : gprm["nsample"],
            "NsampRing"   : gprm["nsample"] / 5,
            "meanradius"  : min(gprm["windowsize"]) / 2,
            "sigmaradius" : min(gprm["windowsize"]) / 10,
        }
        # If not the first gaussian.
        if ng > 0:
            mean_shift(gh["mean"])
            cov_shift(gh["cov"])
            gh["nsample"] = int(rnd.random() * gprm["nsample"])  # noqa: S311
        # Incorporate new distribution's parameters into general
        # parameters dictionary.
        gprm[ng] = deepcopy(gh)
    return


def mean_update(gprm: dict, mean: list, nh: int) -> None:
    """Update the mean in each gaussian distribution.

    This makes the beam sweeping inside the box.

    Args:
        gprm (dict): general parameters of the simulation;
        mean (list): the mean of the distribution (histogram);
        nh (int): the index of the nh-th distribution (histogram).
    """
    meanzero = gprm[0]["mean"]
    for ng in range(nh):
        if ng == 0:
            gprm[0]["mean"] = mean
        else:
            """Update the mean of the distribution preserving the distance
            from the first gaussian."""
            gprm[ng]["mean"] += mean - meanzero


def update_cov(cov: np.ndarray, pos: tuple = (0, 0),
               step: float = 0.01) -> None:
    """Update the covariance."""
    cov[pos[0], pos[1]] += step


def rebin_histogram(hist: np.ndarray, gprm: dict) -> None:
    """Rebin the histogram to the new number of bins.

    Args:
        hist (numpy array): 2-d histogram;
        gprm (dict): general parameters of the simulation.
    """
    # Rebin the histogram to the new number of bins.
    hlin, hcol = hist.shape
    nlin, ncol = gprm['nbins']

    # Check if rebinning is needed or not.
    if hlin == nlin and hcol == ncol:
        return hist

    # Rebinning by bicubic interpolation. Guarantee that the new histogram
    # has the same number of bins as defined in gprm.
    zfactor = (nlin / hlin, ncol / hcol)
    hist_rebinned = zoom(hist, zoom=zfactor, order=3,
                         output=np.zeros((nlin, ncol), dtype=hist.dtype))
    return hist_rebinned


def histogram_init(gprm: dict) -> list:
    """Initialize each histogram.

    Args:
        gprm (dict): general parameters of the distribution.

    Returns:
        beamhist (list): the gaussian histogram distributions which simulate
            the incidence of photons on the blades.
    """
    beamhist = list()
    if gprm['distributionfile'] is not None:
        hist = np.load(gprm['distributionfile'])
        hlin, hcol = hist.shape
        nlin, ncol = gprm['nbins']

        if hlin < nlin or hcol < ncol:
            print(">>> WARNING: beam distribution histogram"
                  f" (shape = {hlin}, {hcol})"
                  "\n>>> is smaller than window box"
                  f" (shape = {nlin}, {ncol}).\n"
                  ">>> Histogram will be rebinned from"
                  f"{hist.shape} to {gprm['nbins']} ...")
            hist = rebin_histogram(hist, gprm)
            print(f">>> done. Final shape = {hist.shape}.")
        beamhist.append(hist)
    elif gprm['randomhist']:
        beamhist.append(gaussian_2d_samples(gprm, 0)[0])
    else:
        beamhist.append(gaussian_2d_analytic(gprm, 0))

    # Create distributions for remaining histograms.
    for ng in range(1, gprm["ngauss"]):
        gh = gprm[ng]
        # New distributions are less intense than original.
        # gh['nsample'] = int(gh['nsample'] * rnd.random())
        if gprm['randomhist']:
            hist = gaussian_2d_samples(gprm, ng)[0]
        else:
            hist = gaussian_2d_analytic(gprm, ng)

        # Add a gaussian ring around the center of the beam.
        if gprm["addring"]:
            hist += histogram_ring(
                gprm,
                radius=gh["meanradius"],
                sradius=gh["sigmaradius"],
                center=gh["mean"],
            )
        beamhist.append(hist)
    return beamhist


def histogram_ring(gprm: dict, mradius: float, sradius: float,
                   center: tuple) -> np.ndarray:
    """Create a ring-like distribution in 2d.

    Its angular part (theta) has an uniform distribution and the radial
    part has gaussian one, with mean radius Radius and standard deviation
    Sradius. The ring's center is set to Center, nsample is the number of
    generated points and nbins is the number of histogram bins.
    """
    nsample = gprm["nsample"]

    # Radial and angular disrtibutions.
    # radius = np.random.Generator.normal(loc=mradius, scale=sradius,
    radius = rng.normal(loc=mradius, scale=sradius, size=nsample)
    phi = rng.random(size=nsample) * 2.0 * np.pi
    # phi = np.random.Generator.random(size=nsample) * 2.0 * np.pi

    # Polar to Cartesian coordinates.
    xpos = radius * np.cos(phi) + center[1]
    ypos = radius * np.sin(phi) + center[0]

    # Classify data as a histogram.
    dx, dy = gprm["PixelSize"], gprm["PixelSize"]
    hbs, vbs = 0.5 * gprm['windowsize'][0], 0.5 * gprm['windowsize'][1]
    hedges = np.arange(-hbs, hbs + dx, dx)
    vedges = np.arange(-vbs, vbs + dy, dy)
    h2d, _, _ = np.histogram2d(xpos, ypos, bins=[hedges, vedges])
    return h2d


def histogram_update(beamhist: list, gprm: dict) -> None:
    """Select method to update histogram, its type and number."""
    for ng in range(len(beamhist)):
        if gprm['distributionfile'] is not None:
            hist = histogram_shift(gprm['origbeam'],
                                   gprm[0]['mean'],
                                   pixelsize=gprm['virtualpixelsize'])
        elif gprm['randomhist']:
            hist = gaussian_2d_samples(gprm, ng)[0]
        else:
            hist = gaussian_2d_analytic(gprm, ng)

        # If a ring-like distribution must be added.
        if gprm["addring"]:
            gh = gprm[ng]
            hist += histogram_ring(
                gprm, mradius=gh["meanradius"],
                sradius=gh["sigmaradius"],
                center=gh["mean"]
            )
        beamhist[ng] = hist


def gaussian_2d_analytic(gprm: dict, ng: int) -> np.ndarray:
    """Create a 2d gaussian distribution.

    Args:
        gprm (dict) : parameters of the simulation, including the mean
        and the covariance matrix of the gaussian distribution, given by
        the keys 'windowsize' (xy-domain), 'pixelsize' (resolution),
        'mean' and 'cov'.
        ng (int) : index of ng-th gaussian (there might be superposition of
            gaussians).

    Returns:
        gauss_xy (numpy array) : the 2d gaussian distribution.
    """
    windowsize   = gprm['windowsize']
    pixelsize    = gprm['virtualpixelsize']
    nbinsx       = int(windowsize[0] / pixelsize)
    nbinsy       = int(windowsize[1] / pixelsize)
    sizex, sizey = windowsize / 2
    xlin         = np.linspace(-sizex, sizex, nbinsx)
    ylin         = np.linspace(-sizey, sizey, nbinsy)
    hx, hy       = np.meshgrid(xlin, ylin)
    # Mean and covariances.
    prm = gprm[ng]
    mx, my = prm['mean'][0], prm['mean'][1]
    [cx, cxy], [cyx, cy] = prm['cov']
    # cx, cy = prm['cx'], prm['cy']
    # cxy, cyx = prm['cxy'], prm['cyx']
    #
    rho = np.sqrt(cxy * cyx) / (cx * cy)
    norm = gprm['nsample'] / (2. * np.pi * np.sqrt(cx * cy * (1 - rho**2)))
    e_x = (hx - mx)**2 / cx
    e_y = (hy - my)**2 / cy
    e_xy = -2 * rho * (hx - mx) * (hy - my) / np.sqrt(cx * cy)
    gauss_xy = (norm * np.exp(- 0.5/(1 - rho**2) * (e_x + e_xy + e_y)))
    return gauss_xy


def gaussian_2d_samples(gprm: dict, idx: int) -> tuple:
    """Generate a multivariate gaussian random sample.

    Given the mean=[mx, my] and the covariance matrix, cov = [[cx, cxy],
    [cyx, cy]], defined in gprm; the specific distribution parameters set is
    selected by idx. The standard values of the parameters are zero mean and
    covariance = 1, with no correlation (cx=cy=1, cxy=cyx=0).
    The function returns an ndarray with gprm[idx]['nsample'] samples.

    Args:
        gprm (dict):  general parameters of the simulation;
        idx (int): the index of the distribution.

    Returns:
        beamhist.T (numpy array): the 2-d random gaussian distribtuion;
        xedges, yedges: the edges of the distribution.
    """
    vnb, hnb = gprm["nbins"]
    vlims = [0.5 * -gprm["windowsize"][1], 0.5 * gprm["windowsize"][1]]
    hlims = [0.5 * -gprm["windowsize"][0], 0.5 * gprm["windowsize"][0]]
    gh = gprm[idx]
    # data = np.random.multivariate_normal(gh["mean"], gh["cov"],
    #                                      size=gh["nsample"])
    data = rng.multivariate_normal(gh["mean"], gh["cov"], size=gh["nsample"])
    beamhist, xedges, yedges = np.histogram2d(
        data[:, 0], data[:, 1], bins=(hnb, vnb), range=[hlims, vlims]
    )
    return beamhist.T, xedges, yedges


def histogram_shift(beamhist: np.ndarray,
                    newbeamcenter: np.ndarray,
                    oldbeamcenter: np.ndarray = np.array([0, 0]),
                    pixelsize: float = 0.2,
                    windowsize: tuple = (1.0, 1.0)) -> np.ndarray:
    """Just shift histogram by mean vector.
    
    Args:
        beamhist (numpy array)       : the histogram to be shifted;
        newbeamcenter (numpy array)  : the new center of the distribution;
        oldbeamcenter (numpy array)  : the old center of the distribution;
        pixelsize (float)            : the pixel size of the histogram.

    Returns:
        newhist (numpy array): the shifted histogram.
    """
    # New histogram skeleton.
    newhist = np.zeros_like(beamhist)

    # Displacement.
    delta = newbeamcenter - oldbeamcenter
    dx, dy  = float(delta[0]), float(delta[1])

    # Histogram dimensions.
    # nlin, ncol = beamhist.shape

    # Work first with float values to avoid collapsing nearby centers,
    # then convert values back to int. 
    window_x, window_y = windowsize[0], windowsize[1]

    # DEBUG
    print("\n###\n### DEBUG histogram_shift() :"
          f"\n pixelsize = {pixelsize}"
          f"\n dx, dy = {dx}, {dy} (beam diff = {newbeamcenter - oldbeamcenter})"
          f"\n window_x, window_y = {window_x}, {window_y} \n")
    # DEBUG

    # Find the overlapping region between the original
    # and shifted histograms. b = original, n = new (shifted).
    bh_x_min = max(0, dx)
    bh_y_min = max(0, dy)
    bh_x_max = min(window_x, window_x - dx)
    bh_y_max = min(window_y, window_y - dy)

    nh_x_min = max(0, dx)
    nh_y_min = max(0, dy)
    nh_x_max = nh_x_min + (bh_x_max - bh_x_min)
    nh_y_max = nh_y_min + (bh_y_max - bh_y_min)

    # Back to integer indexes.
    bh_x_min = np.int64(np.ceil(bh_x_min / pixelsize))
    bh_y_min = np.int64(np.ceil(bh_y_min / pixelsize))
    bh_x_max = np.int64(np.ceil(bh_x_max / pixelsize))
    bh_y_max = np.int64(np.ceil(bh_y_max / pixelsize))
    #
    nh_x_min = np.int64(np.ceil(nh_x_min / pixelsize))
    nh_y_min = np.int64(np.ceil(nh_y_min / pixelsize))
    nh_x_max = np.int64(np.ceil(nh_x_max / pixelsize))
    nh_y_max = np.int64(np.ceil(nh_y_max / pixelsize))

    # DEBUG
    print(f"\n###\n### DEBUG histogram_shift() : beamcenter = {newbeamcenter}"
          f"\t (oldbeamcenter = {oldbeamcenter})")
    for limname in ["bh_x_min", "bh_x_max", "bh_y_min", "bh_y_max",
                    "nh_x_min", "nh_x_max", "nh_y_min", "nh_y_max"]:
        print(f"{limname} = {eval(limname)}")
    # DEBUG

    # Copy the overlapping region from the original histogram
    # to the new histogram by slicing.
    newhist[nh_y_min:nh_y_max,
            nh_x_min:nh_x_max] = beamhist[bh_y_min:bh_y_max,
                                          bh_x_min:bh_x_max]
    
    return newhist


def observables_calculate(img: np.ndarray, bmp: 'BmP') -> tuple:
    """Calculate observables from flux measurement.

    The differences between crossed blades and pairwised are evaluated from
    the formerly masked 2-d histogram Img in the slices arrayintervals
    corresponding to the blades positions.
    """
    # Calculate the flux on each blade.
    flux = bmp.calc_flux(img)

    # Differences between pairwise blades. Scale to box size.
    hpair, vpair = bmp.pair_difference()

    # Differences between neighbour blades. Scale to box size.
    hcross, vcross = bmp.cross_difference()

    # Differences between neighbour blades.
    ineigh = bmp.neighbour_difference()

    return flux, (hpair, vpair), (hcross, vcross), ineigh


def box_values_show(axval: plt.Axes, flux: np.ndarray,
                    mean: np.ndarray, pairpositions: tuple,
                    crosspositions: tuple, ineigh: list) -> plt.Text:
    """Show the values corresponding to incidence flux on the blades.

    The factors that define the position of the beam are calculated by
    different methods.
    """
    # mean = gprm[0]["mean"]
    hpos, vpos = pairpositions
    hcrosspos, vcrosspos = crosspositions

    # Text table with calculated values at each interaction.
    fluxtext = (
        f"{'[Flux 1 - TI]':^22}   "
        f"{'[Flux 0 - TO]':^18}\n"
        f"{flux[1]:^16.4f}        "
        f"{flux[0]:^18.4f}\n\n"
        f"{'[Flux 2 - BI]':^22}   "
        f"{'[Flux 3 - BO]':^18}\n"
        f"{flux[2]:^16.4f}"
        f"{flux[3]:^18.4f}"
    )
    # fluxtext = (
    #     f"{'[Flux 3 - TO]':<18}   {'[Flux 2 - TI]':>22}\n"
    #     f"{flux[3]:<18.4f}        {flux[2]:>16.4f}\n\n"
    #     f"{'[Flux 0 - BO]':<18}   {'[Flux 1 - BI]':>22}\n"
    #     f"{flux[0]:<18.4f}        {flux[1]:>16.4f}"
    # )

    # Real and calculated positions.
    positions = (
        f"{'     ':10}     {'H':>12}   {'V':>14}\n"
        f"{'Real ':<15}    {mean[0]:<12.2f}     {mean[1]:<12.2f}\n\n"
        f"{'Pair ':<15}     {hpos:<10.2f}     {vpos:<10.2f}\n\n"
        f"{'Cross':<15}    {hcrosspos:<10.2f}     {vcrosspos:<10.2f}\n"
    )

    # Neighbour pair positions.
    neightext = (
        f"{'[N. Top]':^50} \n {ineigh[0]:^50.2f} \n"
        f"{'[N. Left]':<20}        {'[N. Right]':>20}]\n"
        f"  {ineigh[3]:<18.2f}       {ineigh[1]:>18.2f} \n"
        f"  {ineigh[2]:^50.2f} \n {'[N. Bottom]':^50}\n\n\n"
    )

    # Table.
    current_table = (
        f"{fluxtext}"
        f"\n\n\n{positions}"
        f"\n\n\n{neightext}"
    )

    axval.clear()
    axval.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axval.tick_params(axis="y", which="both", left=False, right=False,
                      labelleft=False)
    # Center the table within the axes
    return axval.text(0.5, 0.95, current_table, fontsize=11,
                      verticalalignment="top",
                      horizontalalignment="center")


def beam_over_mask(beam: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Superimpose beam over mask, considering their boundaries.
    
    Args:
        beam (numpy array): the 2-d histogram of the distribution;
        mask (numpy array): the 2-d histogram of the blades.

    Returns:
        masked_beam (numpy array): the 2-d histogram of the distribution
            after being masked by the blades.
    """
    blin, bcol = beam.shape
    mlin, mcol = mask.shape
    startlin   = round((blin - mlin)/2)
    endlin     = startlin + mlin
    startcol   = round((bcol - mcol)/2)
    endcol     = startcol + mcol
    return beam[startlin:endlin, startcol:endcol] * mask


def image_show(count: int,
               beamhist: list, maskarray: np.ndarray,
               bmp: 'BmP', gprm: dict,
               axbeam: plt.Axes,
               axblades: plt.Axes,
               axval: plt.Axes) -> tuple:
    """Add histograms and plot resulting image.

    Args:
        count (int): default frame counter for FuncAnimation;
        beamhist (list of numpy arrays): 2-d histograms of the distributions;
        maskarray (numpy array): the array with weights corresponding to the
            presence of the blades;
        bmp (BmP object): methods to calculate the beam position;
        gprm (dict) : general parameters of the simulation;
        axbeam (pyplot axis): the figure axis on which the beam image will be
            shown;
        axblades (pyplot axis): the figure axis to show the intersection of
            distribution and blades;
        axval (pyplot axis): the figure axis to show a box with calculated
            values;

    Returns:
        imbeam (pyplot image): the beam image;
        imblades (pyplot image): the blades image.
    """
    # Update histogram.
    histogram_update(beamhist, gprm)
    imgbeam = beamhist[0] if gprm["ngauss"] == 1 else sum(beamhist)

    xt, yt = 0.5 * gprm["windowsize"][0], 0.5 * gprm["windowsize"][1]
    extent = (-xt, xt, -yt, yt)
    axbeam.clear()
    axbeam.set_xlabel(u"$x$ [mm]")
    axbeam.set_ylabel(u"$y$ [mm]", labelpad=4)
    imbeam = axbeam.imshow(imgbeam, origin="lower", extent=extent)

    # Apply the mask on the image created, so only the regions where
    # the distribution and the blades intersect are considered for the
    # measurements.
    imgmasked = beam_over_mask(imgbeam, maskarray)
    axblades.clear()
    axblades.set_xlabel(u"$x$ [mm]")
    axblades.set_ylabel(u"$y$ [mm]", labelpad=4)

    # Widen left margin once to avoid y-label clipping.
    if count == 0:
        fig = axbeam.get_figure()
        fig.subplots_adjust(left=0.045)
    imblades = axblades.imshow(imgmasked, origin="lower", extent=extent)

    # Measure the flux on the blades, calculate positions and show it.
    (flux, pairpositions,
     crosspositions, ineigh) = observables_calculate(imgmasked, bmp)
    box_values_show(axval=axval, flux=flux, mean=gprm[0]['mean'],
                    pairpositions=pairpositions,
                    crosspositions=crosspositions,
                    ineigh=ineigh)

    return imbeam, imblades


def measurement_record(mean: np.ndarray,
                       pairpositions: np.ndarray,
                       crosspositions: np.ndarray,
                       gprm: dict,
                       fluxes: list = None) -> None:
    """Write mean, crosspositions and pairpositions results to data file.

    Args:
        mean (array): current mean value of sweeping;
        pairpositions (array): pairwise measured positions;
        crosspositions (array): crossed-blades measured positions;
        gprm (dict): general parameters of the simulation.
        fluxes (list) : values of flux on the blades.
    """
    outfilename = gprm['outfilename']

    # Write out crossed-blades position measurements.
    crossfile = f"{outfilename}-cross-00.dat"
    with open(crossfile, "a") as cf:
        dataline = (f"{mean[0]:.6f} {mean[1]:.6f}   "
                    f"{crosspositions[0]:.6f} {crosspositions[1]:.6f}")
        if fluxes is not None:
            dataline += "  " + " ".join([f"{flux:.6f}" for flux in fluxes])
        cf.write(dataline + "\n")

    # Write out paired-blades position measurements.
    pairfile = f"{outfilename}-pair-00.dat"
    with open(pairfile, "a") as pf:
        dataline = (f"{mean[0]:.6f} {mean[1]:.6f}  "
                    f"{pairpositions[0]:.6f} {pairpositions[1]:.6f}")
        if fluxes is not None:
            dataline += "  " + "   ".join([f"{flux:.6f}" for flux in fluxes])
        pf.write(dataline + "\n")


def sweeping_points(gprm: dict) -> np.ndarray:
    """Define the points of measurement inside the Box.

    Args:
        gprm (dict): general parameters of the simulation.

    Returns:
        sweeppos (numpy array): the sites of the lattice to be swept.
    """
    xa, xb = gprm["sweepinterval"]
    ya, yb = gprm["sweepinterval"]

    x = np.linspace(xa, xb, gprm["nsweeps"] + 1)
    y = np.linspace(ya, yb, gprm["nsweeps"] + 1)

    # yy is rows (bottom->top), xx is cols (left->right)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    sweeppos = np.column_stack((xx.ravel(order="C"), yy.ravel(order="C")))
    return sweeppos


def parameters_write(gprm: dict, pfile: object) -> None:
    """Write simulation parameters to data file.

    Args:
        gprm (dict): general parameters of the simulation;
        pfile (file pointer): file to be written to.
    """
    head = ("# Set general parameters for XBPM simulation."
            " Measures in mm when suitable.\n"
            f"# {datetime.now()}\n")
    pfile.write(head)
    for key, val in gprm.items():
        # Skip if entry is a dictionary for gaussian distribution or
        # a copy of the beam distribution.
        if isinstance(key, int) or key == "origbeam":
            continue
        if key == 'cov':
            pfile.write(f"# {key:15} :  [{val[0]} {val[1]}]\n")
            continue
        pfile.write(f"# {key:15} :  {val}\n")
    pfile.write("\n")


def outfile_initialize(gprm: dict) -> None:
    """Initialize output data files.

    Args:
        gprm (dict): general parameters of the simulation.
    """
    cfile = f"{gprm['outfilename']}-cross-00.dat"
    pfile = f"{gprm['outfilename']}-pair-00.dat"
    for dfile in [cfile, pfile]:
        with open(dfile, "w") as df:
            parameters_write(gprm, df)


def sweep_make(fig: plt.Figure, beamhist: np.ndarray,
               imageshow: tuple, blades, bmp, gprm: dict) -> None:
    """Loop for the sweeping process of the beam in a rectangle inside 'Box'.

    Args:
        fig (pyplot figure): the figure canvas to be updated
        beamhist (numpy array): 2d histogram with total beam distribution
        imageshow (tuple): pyplot axes and images;
        blades (BladeMask object): mask array and its intervals;
        bmp (BeamPosition object): methods to calculate position;
        gprm (dict): general parameters
    """
    imshowbeam, imshowblades, axval = imageshow

    # Points where to center the distribution(s) for further sweeping.
    beam_centers = sweeping_points(gprm)

    # Set output files.
    outfile_initialize(gprm)

    # DEBUG
    print("\n###\n### DEBUG sweep_make() :\n"
          f" beam_centers =\n {beam_centers}")
    # DEBUG

    # Shift the mean of the distribution(s) to perform the sweeping.
    beamhist = beamhist[0]
    oldbeamcenter = gprm[0]["mean"]
    for beamcenter in beam_centers:

        # DEBUG
        print("\n###\n### DEBUG sweep_make() : "
            f" beam center now = {beamcenter}")
        # DEBUG

        shiftedbeam = histogram_shift(beamhist,
                                      beamcenter,
                                      oldbeamcenter,
                                      pixelsize=gprm["virtualpixelsize"],
                                      windowsize=gprm["windowsize"])

        # Update the 'mean' entry in gprm.
        # mean_update(gprm, mean, len(beamhist))


        # Update the histograms for the new mean.
        if gprm['histupdate']:
            histogram_update(beamhist, gprm)

        oldbeamcenter = beamcenter
        beamhist      = shiftedbeam
        imgbeam       = shiftedbeam if gprm["ngauss"] == 1 else sum(beamhist)
        imshowbeam.set_data(imgbeam)
        #
        # imgmasked = imgbeam * blades.maskarray
        imgmasked = beam_over_mask(imgbeam, blades.maskarray)
        imshowblades.set_data(imgmasked)

        # Update measured data and show it.
        (fluxes, pairpositions,
         crosspositions, ineigh) = observables_calculate(imgmasked, bmp)
        box_values_show(axval, fluxes, beamcenter, pairpositions,
                        crosspositions, ineigh)

        # Record values.
        registerflux = fluxes if gprm['registerflux'] else None
        measurement_record(beamcenter, pairpositions, crosspositions, gprm,
                           fluxes=registerflux)
        fig.canvas.draw_idle()
        plt.pause(0.1)


STEP_RESOLUTION_FACTOR: float = 0.4

def number_of_bins(gprm: dict) -> tuple:
    """Calculate the number of bins in x and y directions.

    Args:
        gprm (dict): general parameters of the simulation.
    """
    window_x, window_y = gprm['windowsize']
    roi_begin, roi_end = gprm['sweepinterval']

    # Check the smaller size between pixel size and sweep step.
    # Guarantee that the resolution of the histogram array is sufficient.
    stepsize  = (roi_end - roi_begin) / gprm["nsweeps"]
    pixelsize = min(stepsize, gprm["pixelsize"]) * STEP_RESOLUTION_FACTOR

    # Calculate the number of bins in x and y directions.
    nlin = window_y / pixelsize
    ncol = window_x / pixelsize

    return np.int64(np.ceil([nlin, ncol])), pixelsize


def parameters_read(parfilename: str, distributionfile: str = None) -> tuple:
    """Read simulation parameters from file.

    Args:
        parfilename (str): parameter's file name
        distributionfile (str) : file with previously generated image.

    Returns:
        prm (dict): read parameters and their respective values.
        mean (numpy array): initial gaussian beam mean;
        cov (numpy array): initial gaussian beam covariance.
    """
    prm = dict()
    with open(parfilename, 'r') as pf:
        for line in pf:
            # Skip comments
            if re.match('#', line) or re.match(r"^\ *$", line):
                continue
            # Get parameters and their values.
            parval = line.split()
            key, val = parval[0], parval[1:]

            if key in ['windowsize', 'mean', 'sweepinterval']:
                v1 = float(re.sub(r'[\[\,]', '', val[0]))
                v2 = float(re.sub(r'[\]\,]', '', val[1]))
                prm[key] = np.array([v1, v2])
            elif key in ['gains']:
                prm[key] = np.array([
                    float(re.sub(r'[\[\,\]]', '', g))
                    for g in val
                    ])
            elif key in ['addring', 'histupdate',
                         'randomhist', 'registerflux']:
                prm[key] = False if val[0] == 'False' else True
            elif key in ['ngauss', 'nsweeps', 'nsample']:
                prm[key] = int(float(val[0]))
            elif key == 'thetadeg':
                prm['theta'] = round(float(val[0]) * np.pi / 180.0, 6)
            else:
                prm[key] = float(val[0])

    # Number of histogram bins (number of pixels in image).
    # Guarantee that the resolution of the histogram array is sufficient.
    prm['nbins'], prm['virtualpixelsize'] = number_of_bins(prm)

    # DEBUG
    print("\n###\n### DEBUG number_of_bins() :"
          f"\n pixelsize (after factor) = {prm['virtualpixelsize']}"
          f"\n nlin, ncol = {prm['nbins'][0]}, {prm['nbins'][1]}\n###\n")
    # DEBUG

    # Set gains to 1 if not defined.
    if 'gains' not in prm:
        prm['gains'] = np.array([1.0, 1.0, 1.0, 1.0])

    # Set standard mean, covariance matrix and number of samples per frame
    # for each histogram.
    mean = deepcopy(prm['mean'])
    cx, cy = prm['fwhm_x'] * FWHM2SIGMA, prm['fwhm_y'] * FWHM2SIGMA
    cov = np.array([[cx, prm['cxy']], [prm['cyx'], cy]])
    prm['cov'] = cov

    # Define output file name base.
    sdx = (prm['sweepinterval'][1] - prm['sweepinterval'][0]) / prm['nsweeps']
    outfilename = (
        f"XBPM_mu_{0.0:04.1f}_FWHM_"
        f"x{prm['fwhm_x']:04.1f}_"
        f"y{prm['fwhm_y']:04.1f}_step{sdx:05.3f}"
    )
    prm['outfilename'] = outfilename
    prm['distributionfile'] = distributionfile
    prm['sweepstep'] = sdx

    return prm, mean, cov


def cmd_options() -> tuple:
    """Get command line options."""
    # Read options, if available.
    try:
        opts = getopt.getopt(sys.argv[1:], "hid:p:")
    except getopt.GetoptError as err:
        print("\n\n ERROR: ", str(err), "\b.")
        sys.exit(1)

    interactive = False
    parameterfile, distributionfile = None, None

    for op in opts[0]:
        if op[0] == "-h":
            """Help message."""
            help("xbpm_sim")
            sys.exit(0)

        if op[0] == "-i":
            # Interactive mode.
            interactive = True

        if op[0] == "-p":
            # Simulation parameters.
            parameterfile = op[1]

        if op[0] == "-d":
            # Import pre-generated distribution.
            distributionfile = op[1]

    if parameterfile is None:
        print("ERROR: no parameters were provided. "
              "Run with the -h option to see the help message."
              "Aborting.")
        sys.exit(1)

    return interactive, parameterfile, distributionfile


def main() -> int:
    """Simulate in real time the incidence of photons upon XBPM blades."""
    # Initialize random seed.
    rnd.seed()

    # Read command line and simulation parameters from file.
    # interactive (boolean): user interaction or automatic sweeping;
    # parameterfile (string): file to read parameters from;
    # gprm (dict): general parameters of the simulation;
    # mean (numpy array): default mean 1x2 matrix of the distribution;
    # cov (numpy array): default covariance 2x2 matrix of the distribution;
    interactive, parameterfile, distributionfile = cmd_options()
    gprm, mean, cov = parameters_read(parameterfile, distributionfile)

    # Add a dictionary of each gaussian's parameters to the general
    # parameters (gprm) dictionary.
    histogram_parameters_set(gprm, mean, cov)

    # Initialize histogram(s).
    if gprm['distributionfile'] is not None:
        print(" Reading beam distribution from file: "
              f"\n >>> {gprm['distributionfile']}")
    elif gprm['randomhist']:
        print(f" Creating a distribution with {gprm['nsample']:g} samples"
            " (this may take a while)... ", end="")
    beamhist = histogram_init(gprm)
    gprm['origbeam'] = deepcopy(beamhist[0])
    print("done.\n")

    # Create blades array, a 'mask'.
    blades = BladeMask(gprm)

    # Initialize beam position calculation methods.
    bmp = BmP(blades.bladescoordinates, gprm)

    # Initialize subplots.
    fig, (axbeam, axblades, axval) = plt.subplots(1, 3, figsize=(15, 5))
    fig.tight_layout(pad=2.0)

    # Listen to the keyboard arrows (interactive motion of the beam mean).
    if interactive:
        step = 0.2
        listener = Listener(
            on_press=lambda event: on_press(event, gprm=gprm, step=step),
            on_release=on_release,
        )
        listener.start()

        # Animation function caller.
        imshow = partial(image_show, beamhist=beamhist,
                         maskarray=blades.maskarray, bmp=bmp, gprm=gprm,
                         axbeam=axbeam, axblades=axblades, axval=axval)
        try:
            # Animate. Variable 'anim' prevents FuncAnimation
            # from being deleted without rendering.
            anim = mpanim.FuncAnimation(fig, imshow,         # noqa: F841
                                        repeat=False,
                                        repeat_delay=500,
                                        cache_frame_data=False)
            # writer = mpanim.PillowWriter(fps=2)
            # anim.save("xbpm_sweep.gif", writer=writer)

        except Exception as err:
            print("ERROR when calling FuncAnimation: ", err)
    else:
        # Show initial image.
        imbeam, imblades = image_show(None, beamhist, blades.maskarray, bmp,
                                      gprm, axbeam, axblades, axval)

        imshow = (imbeam, imblades, axval)
        sweep_make(fig, beamhist, imshow, blades, bmp, gprm)

    # Show images.
    plt.savefig("beam_and_blades.png")
    plt.tight_layout()
    plt.show()
    if interactive:
        listener.stop()
    return 0


if __name__ == "__main__":
    main()
    print("Done.")
