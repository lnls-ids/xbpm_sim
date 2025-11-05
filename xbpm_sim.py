#!/usr/bin/env python3
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
     -p <paramter file> : the file which defines the parameters of the
          simulation, as the number of random points in the beam, its shape,
          the format of blades and their geometry etc.
    -d <distribution file> : import an externally generated distribution
          in numpy array file format. It allows for X-ray profiles simulated
          from bend magnets or undulators, or registered from DVF cameras.


The default is non-interactive, meaning a sweep is made over the
defined window. Length units are in mm; angles, in degrees.

The parameters (to be defined in the parameter file) are:

ngauss (int) : number of gaussian distributions to be superimposed.
           Emulate distortions in the beam by adding up gaussian distributions;
           mean and variances of each are randomly defined.

nsample (int) : number of gaussian random samples (2-d coordinates) to
           represent the incident 'photons' in each frame;

pixelsize : the length resolution of the system;

 - Blades' geometry.

windowsize [(float), (float)] : size of the area where blades are defined
      (related to the Cu mask);

bladelength (float): length at which the the pin diodes are installed,
           replacing the previous blades;

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
from pinmask import PinMask                   # type: ignore # noqa: E402

# Initialize random number generator.
rng = np.random.default_rng(seed=None)

# Ratio  in sigma/FWHM in a gaussian distribution.
FWHM2SIGMA = 1.0 / np.sqrt(np.log(256))


def on_press(key, prm, step):
    """Update the mean of the gaussians on pressing the keyboard arrows."""
    for ng in range(prm["ngauss"]):
        mean = prm[ng]["mean"]
        if key == Key.left:
            mean[0] -= step
        if key == Key.right:
            mean[0] += +step
        if key == Key.up:
            mean[1] += step
        if key == Key.down:
            mean[1] -= step


def on_release(key):
    """What to do if a key is released.

    Args:
        key (string): the pressed key.

    Returns:
        False
    """
    if key == Key.esc:
        return False


def mean_shift(mean):
    """Define the centroid and the dispersion of the distributions.

    Set the mean in x (0) and y (directions) of gaussian distributions
    by shifting the original mean and variance by some random amount.
    """
    mean[0] += 0.5 + 0.2 * (rnd.random() - 0.5)  # noqa: S311
    mean[1] += 0.5 * rnd.random()  # noqa: S311
    return


def cov_shift(cov):
    """Shift the covariance by some random amount."""
    shift = 1.0
    cov[0, 0] = np.abs(cov[0, 0] + shift * (rnd.random() - 0.5))  # noqa: S311
    cov[0, 1] = np.abs(cov[0, 1] + shift * (rnd.random() - 0.5))  # noqa: S311
    cov[1, 0] = cov[0, 1]
    cov[1, 1] = np.abs(cov[1, 1] + shift * (rnd.random() - 0.5))  # noqa: S311


def histogram_parameters_set(prm):
    """Set mean, covariance and number of samples for each histogram.

    Args:
        prm (dict): general parameters of the simulation;
        mean (numpy array): the mean of the distribution in x and y;
        cov (numpy array): the covariance of the distribution in x and y;
    """
    for ng in range(prm["ngauss"]):
        # Dictionary for gaussian parameters: mean, covariance and number
        # of samples.
        gh = {
            "mean": deepcopy(prm["mean"]),
            "cov": deepcopy(prm["cov"]),
            "nsample": prm["nsample"],
            "NsampRing": prm["nsample"] / 5,
            "meanradius": min(prm["windowsize"]) / 2,
            "sigmaradius": min(prm["windowsize"]) / 10,
            # 'sigmaradius' : max(cov[0,0],
            #  cov[1,1]) * min(prm['windowsize']),
        }
        # If not the first gaussian.
        if ng > 0:
            mean_shift(gh["mean"])
            cov_shift(gh["cov"])
            gh["nsample"] = int(rnd.random() * prm["nsample"])  # noqa: S311
        # Incorporate new distribution's parameters into general
        # parameters dictionary.
        prm[ng] = deepcopy(gh)
    return


def mean_update(prm, mean, nh):
    """Update the mean in each gaussian distribution.

    This makes the beam sweeping inside the box.

    Args:
        prm (dict): general parameters of the simulation;
        mean (list): the mean of the distribution (histogram);
        nh (int): the index of the nh-th distribution (histogram).
    """
    meanzero = prm[0]["mean"]
    for ng in range(nh):
        if ng == 0:
            prm[0]["mean"] = mean
        else:
            """Update the mean of the distribution preserving the distance
            from the first gaussian."""
            prm[ng]["mean"] += mean - meanzero


def update_cov(cov, pos=(0, 0), step=0.01):
    """Update the covariance."""
    cov[pos[0], pos[1]] += step


def histogram_init(prm):
    """Initialize each histogram.

    Args:
        prm (dict): general parameters of the distribution.

    Returns:
        beamhist (list): the gaussian histogram distributions which simulate
            the incidence of photons on the blades.
    """
    beamhist = list()
    if prm['distributionfile'] is not None:
        print(" Reading beam distribution from file: "
              f"{prm['distributionfile']}...")
        hist = np.load(prm['distributionfile'])
        hlin, hcol = hist.shape
        nlin, ncol = [int(x) for x in prm['nbins'] + prm['sweepinterval']]

        if hlin < nlin or hcol < ncol:
            print(">>>>> WARNING: beam distribution histogram"
                  f" (shape = {hlin}, {hcol})is smaller than window box"
                  f" (shape = {nlin}, {ncol}).\n"
                  ">>>>> Superposition might be truncated at the borders.\n\n")
        beamhist.append(hist)
    elif prm['randomhist']:
        print(f" Creating a distribution with {prm['nsample']:g} samples"
            " (this may take a while)... ", end="")
        beamhist.append(gaussian_2d_samples(prm, 0)[0])
    else:
        beamhist.append(gaussian_2d_analytic(prm, 0))

    # Create distributions for remaining histograms.
    for ng in range(1, prm["ngauss"]):
        gh = prm[ng]
        # New distributions are less intense than original.
        # gh['nsample'] = int(gh['nsample'] * rnd.random())
        if prm['randomhist']:
            hist = gaussian_2d_samples(prm, ng)[0]
        else:
            hist = gaussian_2d_analytic(prm)

        # Add a gaussian ring around the center of the beam.
        if prm["addring"]:
            hist += histogram_ring(
                prm,
                radius=gh["meanradius"],
                sradius=gh["sigmaradius"],
                center=gh["mean"],
            )
        beamhist.append(hist)
    return beamhist


def histogram_ring(prm, mradius, sradius, center):
    """Create a ring-like distribution in 2d.

    Its angular part (theta) has an uniform distribution and the radial
    part has gaussian one, with mean radius Radius and standard deviation
    Sradius. The ring's center is set to Center, nsample is the number of
    generated points and nbins is the number of histogram bins.
    """
    nsample = prm["nsample"]

    # Radial and angular disrtibutions.
    # radius = np.random.Generator.normal(loc=mradius, scale=sradius,
    radius = rng.normal(loc=mradius, scale=sradius, size=nsample)
    phi = rng.random(size=nsample) * 2.0 * np.pi
    # phi = np.random.Generator.random(size=nsample) * 2.0 * np.pi

    # Polar to Cartesian coordinates.
    xpos = radius * np.cos(phi) + center[1]
    ypos = radius * np.sin(phi) + center[0]

    # Classify data as a histogram.
    dx, dy = prm["PixelSize"], prm["PixelSize"]
    hbs, vbs = 0.5 * prm['windowsize'][0], 0.5 * prm['windowsize'][1]
    hedges = np.arange(-hbs, hbs + dx, dx)
    vedges = np.arange(-vbs, vbs + dy, dy)
    h2d, _, _ = np.histogram2d(xpos, ypos, bins=[hedges, vedges])
    return h2d


def histogram_update(beamhist, prm):
    """Select method to update histogram, its type and number."""
    for ng in range(len(beamhist)):
        if prm['distributionfile'] is not None:
            hist = histogram_shift(prm['origbeam'],
                                   prm[0]['mean'],
                                   pixelsize=prm['pixelsize'])
        elif prm['randomhist']:
            hist = gaussian_2d_samples(prm, ng)[0]
        else:
            hist = gaussian_2d_analytic(prm, ng)

        # If a ring-like distribution must be added.
        if prm["addring"]:
            gh = prm[ng]
            hist += histogram_ring(
                prm, mradius=gh["meanradius"],
                sradius=gh["sigmaradius"], center=gh["mean"]
            )
        beamhist[ng] = hist


def gaussian_2d_analytic(prm, ng):
    """Create a 2d gaussian distribution.

    Args:
        prm (dict) : parameters of the simulation, including the mean
        and the covariance matrix of the gaussian distribution, given by
        the keys 'windowsize' (xy-domain), 'pixelsize' (resolution),
        'mean' and 'cov'.
        ng (int) : index of ng-th gaussian (there might be superposition of
            gaussians).

    Returns:
        gauss_xy (numpy array) : the 2d gaussian distribution.
    """
    windowsize = prm['windowsize']
    pixelsize  = prm['pixelsize']
    sizex, sizey = windowsize / 2
    nbinsx = int(windowsize[0] / pixelsize)
    nbinsy = int(windowsize[1] / pixelsize)
    xlin   = np.linspace(-sizex, sizex, nbinsx)
    ylin   = np.linspace(-sizey, sizey, nbinsy)
    hx, hy = np.meshgrid(xlin, ylin)
    # Mean and covariances.
    nprm   = prm[ng]
    mx, my = nprm['mean'][0], nprm['mean'][1]
    [cx, cxy], [cyx, cy] = nprm['cov']
    #
    rho = np.sqrt(cxy * cyx) / (cx * cy)
    norm = prm['nsample'] / (2. * np.pi * np.sqrt(cx * cy * (1 - rho**2)))
    e_x = (hx - mx)**2 / cx
    e_y = (hy - my)**2 / cy
    e_xy = -2 * rho * (hx - mx) * (hy - my) / np.sqrt(cx * cy)
    gauss_xy = (norm * np.exp(- 0.5/(1 - rho**2) * (e_x + e_xy + e_y)))
    return gauss_xy


def gaussian_2d_samples(prm, idx):
    """Generate a multivariate gaussian random sample.

    Given the mean=[mx, my] and the covariance matrix, cov = [[cx, cxy],
    [cyx, cy]], defined in prm; the specific distribution parameters set is
    selected by idx. The standard values of the parameters are zero mean and
    covariance = 1, with no correlation (cx=cy=1, cxy=cyx=0).
    The function returns an ndarray with prm[idx]['nsample'] samples.

    Args:
        prm (dict):  general parameters of the simulation;
        idx (int): the index of the distribution.

    Returns:
        beamhist.T (numpy array): the 2-d random gaussian distribtuion;
        xedges, yedges: the edges of the distribution.
    """
    vnb, hnb = prm["nbins"]
    vlims = [0.5 * -prm["windowsize"][1], 0.5 * prm["windowsize"][1]]
    hlims = [0.5 * -prm["windowsize"][0], 0.5 * prm["windowsize"][0]]
    gh = prm[idx]
    # data = np.random.multivariate_normal(gh["mean"], gh["cov"],
    #                                      size=gh["nsample"])
    data = rng.multivariate_normal(gh["mean"], gh["cov"], size=gh["nsample"])
    beamhist, xedges, yedges = np.histogram2d(
        data[:, 0], data[:, 1], bins=(hnb, vnb), range=[hlims, vlims]
    )
    return beamhist.T, xedges, yedges


def histogram_shift(beamhist, mean, oldmean=(0, 0), pixelsize=0.2):
    """Just shift histogram by mean vector."""
    # The image must extend to the whole box area.
    lin, col = beamhist.shape
    # New histogram image.
    newhist = np.zeros((lin, col))
    # Displacement.
    delta = (mean - oldmean) / pixelsize
    mx, my = int(delta[0]), int(delta[1])
    for iy in range(lin):
        for ix in range(col):
            nx, ny = ix + mx, iy + my
            if (nx < col and ny < lin and
                nx >= 0 and ny >= 0):
                newhist[ny, nx] = beamhist[iy, ix]
    return newhist


def observables_calculate(img, bmp):
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


def box_values_show(axval, fluxes, mean, beamposition):
    """Show the values corresponding to incidence flux on the blades.

    The factors that define the position of the beam are calculated by
    different methods.
    """
    # mean = prm[0]["mean"]

    # Text table with calculated values at each interaction.
    # fluxtext = f"{' [Fluxes]':^40}
    # for flux in fluxes:
    # (
    #     f"{'[Flux 3 - TO]':<18}   {'[Flux 2 - TI]':>22}\n"
    #     f"{flux[3]:<18.4f}        {flux[2]:>16.4f}\n\n"
    #     f"{'[Flux 0 - BO]':<18}   {'[Flux 1 - BI]':>22}\n"
    #     f"{flux[0]:<18.4f}        {flux[1]:>16.4f}"
    # )

    # Real and calculated positions.
    hpos, vpos = beamposition
    pos_text = (
        f"{'     ':10}     {'H':>12}   {'V':>14}\n"
        f"{'Real ':<15}    {mean[0]:<12.2f}     {mean[1]:<12.2f}\n\n"
        f"{'C.M. ':<15}    {hpos:<10.2f}     {vpos:<10.2f}\n\n"
    )

    # Table.
    current_table = (
        # f"{fluxtext}"
        f"\n\n\n{pos_text}"
    )

    axval.clear()
    axval.set_frame_on(False)
    axval.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
    axval.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
        )
    return axval.text(0.20, 0.90, current_table, fontsize=12,
                      verticalalignment="top")


def beam_over_mask(beam, mask):
    """Superimpose beam over mask, considering their boundaries."""
    blin, bcol = beam.shape
    mlin, mcol = mask.shape
    startlin = round((blin - mlin)/2)
    endlin = startlin + mlin
    startcol = round((bcol - mcol)/2)
    endcol = startcol + mcol
    return beam[startlin:endlin, startcol:endcol] * mask


def interactive_run(prm, beamhist, mask, bmp, fig, axes, step=0.2):
    """."""
    listener = Listener(
        on_press=lambda event: on_press(event, prm=prm, step=step),
        on_release=on_release,
    )
    listener.start()

    # Animation function caller
    axbeam, axblades, axval = axes
    imshow = partial(image_show, beamhist=beamhist,
                        mask=mask, bmp=bmp, prm=prm,
                        axes=(axbeam, axblades, axval))
    try:
        # Animate. Variable 'anim' prevents FuncAnimation
        # from being deleted without rendering.
        anim = mpanim.FuncAnimation(fig, imshow,         # noqa: F841
                                    repeat=False,
                                    repeat_delay=500)
        # writer = mpanim.PillowWriter(fps=2)
        # anim.save("xbpm_sweep.gif", writer=writer)

    except Exception as err:
        print("ERROR when calling FuncAnimation: ", err)

    return anim, listener


def image_show(count, beamhist, mask, bmp, prm, axes):  # noqa: ARG001
    """Add histograms and plot resulting image.

    Args:
        count (int): default frame counter for FuncAnimation;
        beamhist (list of numpy arrays): 2-d histograms of the distributions;
        mask (numpy array): the array with weights corresponding to the
            presence of the blades;
        bmp (BeamPosition object): methods to calculate the beam position;
        prm (dict) : general parameters of the simulation;
        axes (tuple): triple with the pyplot axes: axbeam (the figure axis
            on which the beam image will be shown), axblades (the figure axis
            to show the intersection of distribution and blades), axval (the
            figure axis to show a box with calculated values).

    Returns:
        imbeam (pyplot image): the beam image;
        imblades (pyplot image): the blades image.
    """
    # Update histogram.
    histogram_update(beamhist, prm)
    imgbeam = beamhist[0] if prm["ngauss"] == 1 else sum(beamhist)
    axbeam, axblades, axval = axes

    xt, yt = 0.5 * prm["windowsize"][0], 0.5 * prm["windowsize"][1]
    extent = (-xt, xt, -yt, yt)
    axbeam.clear()
    imbeam = axbeam.imshow(imgbeam, origin="lower", extent=extent)

    # Apply the mask on the image created, so only the regions where
    # the distribution and the blades intersect are considered for the
    # measurements.
    imgmasked = beam_over_mask(imgbeam, mask)
    axblades.clear()
    imblades = axblades.imshow(imgmasked, origin="lower", extent=extent)

    # Measure the flux on the blades, calculate positions and show it.
    (beam_pos, fluxes) = bmp.center_of_mass(imgmasked)
    box_values_show(axval=axval, fluxes=fluxes, mean=prm[0]['mean'],
                    beamposition=beam_pos)

    return imbeam, imblades


def measurement_record(mean, beam_pos, prm, fluxes=None):
    """Write mean, crosspositions and pairpositions results to data file.

    Args:
        mean (array): current mean value of sweeping;
        beam_pos (array): calculated positions;
        prm (dict): general parameters of the simulation.
        fluxes (list) : values of flux on the blades.
    """
    outfilename = prm['outfilename']

    # Write out crossed-blades position measurements.
    positionfile = f"{outfilename}-positions-00.dat"
    with open(positionfile, "a") as ff:
        dataline = (f"{mean[0]:.6f} {mean[1]:.6f}   "
                    f"{beam_pos[0]:.6f} {beam_pos[1]:.6f}")
        ff.write(dataline + "\n")

    # Write out fluxes.
    if fluxes is not None:
        fluxfile = f"{outfilename}-fluxes-00.dat"
        with open(fluxfile, "a") as ff:
            dataline += "  " + " ".join([f"{flux[0][0]:.6f}  "
                                         f"{flux[0][1]:.6f} "
                                         f"{flux[1]:.6f}"
                                         for flux in fluxes])
            ff.write(dataline + "\n")


def sweeping_points(prm):
    """Define the points of measurement inside the Box.

    Args:
        prm (dict): general parameters of the simulation.

    Returns:
        sweeppos (numpy array): the sites of the lattice to be swept.
    """
    xa, xb = prm["sweepinterval"]
    dx = (xb - xa) / prm["nsweeps"]
    #
    ya, yb = prm["sweepinterval"]
    dy = (yb - ya) / prm["nsweeps"]

    # Run through columns in a line, then move to next line.
    sweeppos = np.array(
        [
            [ii, jj]
            for jj in np.arange(xa, xb + dx, dx)
            for ii in np.arange(ya, yb + dy, dy)
        ]
    )
    return sweeppos


def parameters_write(prm, pfile):
    """Write simulation parameters to data file.

    Args:
        prm (dict): general parameters of the simulation;
        pfile (file pointer): file to be written to.
    """
    head = ("# Set general parameters for XBPM simulation."
            " Measures in mm when suitable.\n"
            f"# {datetime.now()}\n")
    pfile.write(head)
    for key, val in prm.items():
        # Skip if entry is a dictionary for gaussian distribution or
        # a copy of the beam distribution.
        if isinstance(key, int) or key == "origbeam":
            continue
        if key == 'cov':
            pfile.write(f"# {key:15} :  [{val[0]} {val[1]}]\n")
            continue
        pfile.write(f"# {key:15} :  {val}\n")
    pfile.write("\n")


def outfile_initialize(prm):
    """Initialize output data files.

    Args:
        prm (dict): general parameters of the simulation.
    """
    pfile = f"{prm['outfilename']}-positions-00.dat"
    ffile = f"{prm['outfilename']}-fluxes-00.dat"
    for dfile in [pfile, ffile]:
        with open(dfile, "w") as df:
            parameters_write(prm, df)


def sweep_make(fig, beamhist, imageshow, pindiodes, bmp, prm):
    """Loop for the sweeping process of the beam in a rectangle inside 'Box'.

    Args:
        fig (pyplot figure): the figure canvas to be updated
        beamhist (numpy array): 2d histogram with total beam distribution
        imageshow (tuple): pyplot axes and images;
        pindiodes (PinMask object): mask array and its intervals;
        bmp (BeamPosition object): methods to calculate position;
        prm (dict): general parameters
    """
    imshowbeam, imshowblades, axval = imageshow

    # Points where to center the distribution(s) for further sweeping.
    means = sweeping_points(prm)

    # Set output files.
    outfile_initialize(prm)

    # Shift the mean of the distribution(s) to perform the sweeping.
    for mean in means:
        shiftedbeam = histogram_shift(beamhist[0], mean,
                                      oldmean=prm['mean'],
                                      pixelsize=prm["pixelsize"])

        # Update the 'mean' entry in prm.
        # mean_update(prm, mean, len(beamhist))

        # Update the histograms for the new mean.
        if prm['histupdate']:
            histogram_update(beamhist, prm)

        imgbeam = shiftedbeam if prm["ngauss"] == 1 else sum(beamhist)
        imshowbeam.set_data(imgbeam)
        #
        # imgmasked = imgbeam * blades.mask
        imgmasked = beam_over_mask(imgbeam, pindiodes.mask)
        imshowblades.set_data(imgmasked)

        # Update measured data and show it.
        (beam_pos, fluxes) = bmp.center_of_mass(imgmasked)
        box_values_show(axval, fluxes, mean, beam_pos)

        # Record values.
        registerflux = fluxes if prm['registerflux'] else None
        measurement_record(mean, beam_pos, prm, fluxes=registerflux)

        fig.canvas.draw_idle()
        plt.pause(0.1)


def parameters_read(parfilename, distributionfile):
    """Read simulation parameters from file.

    Args:
        parfilename (str): parameter's file name
        distributionfile (str) : file with previously generated image.

    Returns:
        prm (dict): read parameters and their respective values.
        mean (numpy array): initial gaussian beam mean;
        cov (numpy array): initial gaussian beam covariance.
    """
    boollist = ['addring', 'histupdate', 'randomhist',
                'registerflux', 'centerpins']
    prm = {item : False for item in boollist}
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
            elif key in boollist:
                if val[0] == "True":
                    prm[key] = True
            elif key in ['ngauss', 'nsweeps', 'nsample', 'ndots']:
                prm[key] = int(float(val[0]))
            elif key == 'thetadeg':
                prm['theta'] = round(float(val[0]) * np.pi / 180.0, 6)
            else:
                prm[key] = float(val[0])

    # Number of histogram bins (number of pixels in image).
    prm['nbins'] = [int(prm['windowsize'][0] / prm['pixelsize']),
                    int(prm['windowsize'][1] / prm['pixelsize'])]

    # Set standard mean, covariance matrix and number of samples per frame
    # for each histogram.
    cx, cy = prm['fwhm_x'] * FWHM2SIGMA, prm['fwhm_y'] * FWHM2SIGMA
    prm['cov'] = np.array([[cx, prm['cxy']], [prm['cyx'], cy]])

    # Define output file name base.
    sdx = (prm['sweepinterval'][1] -
           prm['sweepinterval'][0]) / prm['nsweeps']
    outfilename = (
        f"XBPM_mu_{0.0:04.1f}_FWHM_"
        f"x{prm['fwhm_x']:04.1f}_"
        f"y{prm['fwhm_y']:04.1f}_step{sdx:05.3f}"
    )
    prm['outfilename'] = outfilename
    prm['distributionfile'] = distributionfile
    prm['sweepstep'] = sdx

    return prm


def cmd_options():
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


def main():
    """Simulate in real time the incidence of photons upon XBPM blades."""
    # Initialize random seed.
    rnd.seed()

    # Read command line arguments.
    interactive, parameterfile, distributionfile = cmd_options()

    # General parameters of the simulation.
    prm = parameters_read(parameterfile, distributionfile)

    # Add a dictionary of each gaussian's parameters to the general
    # parameters (prm) dictionary.
    histogram_parameters_set(prm)

    # Initialize histogram(s).
    beamhist = histogram_init(prm)
    prm['origbeam'] = deepcopy(beamhist[0])
    print("done.\n")

    # Create blades array, a 'mask'.
    pindiodes = PinMask(prm)

    # Initialize beam position calculation methods.
    bmp = BmP(pindiodes, prm)

    # Initialize subplots.
    fig, (axbeam, axblades, axval) = plt.subplots(1, 3, figsize=(15, 6))
    fig.tight_layout()

    # Listen to the keyboard arrows (interactive motion of the beam mean).
    if interactive:
        anim, listener = interactive_run(prm, beamhist, pindiodes.mask,
                                         bmp, fig,
                                         axes=(axbeam, axblades, axval),
                                         step=0.1)
    else:
        # Show initial image.
        imbeam, imblades = image_show(None, beamhist, pindiodes.mask, bmp,
                                      prm, axes=(axbeam, axblades, axval))

        imshow = (imbeam, imblades, axval)
        sweep_make(fig, beamhist, imshow, pindiodes, bmp, prm)

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
