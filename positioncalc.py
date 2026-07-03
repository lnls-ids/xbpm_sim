"""Classes to calculate beam position."""

import numpy as np


class BeamPosition():
    """Beam's position by different methods, given the flux on each blade."""
    def __init__(self, bladescoordinates: list, gprm: dict):
        # def __init__(self, intervals, windowsize=(10, 10), theta=0):
        """Initialize general parameters.

        Args:
            hist_img (numpy array): histogram array which defines the 'photons'
                    distribution image
            bladescoordinates (list): blades' corners coordinates;
            gprm (dict): general parameters to define the blades geometry.
        """
        # self.intervals = intervals
        self.bladescoordinates = bladescoordinates
        self.windowsize = gprm['windowsize']
        self.pixelsize  = gprm['virtualpixelsize']
        self.nbins      = gprm['nbins']
        self.theta      = gprm['theta']
        self.intervals  = self._blades_intervals()
        self.gains      = gprm['gains']

    def calc_flux(self, hist_img: np.ndarray) -> list:
        """Calculate the flux on each blade."""
        flux = list()
        # Incidence angle correction.
        angle_correction = 1.0 / np.cos(self.theta)
        # Calculate the flux on every blade.
        for interval in self.intervals:
            nlinmin, ncolmin = int(interval[0][0]), int(interval[0][1])
            nlinmax, ncolmax = int(interval[1][0]), int(interval[1][1])
            flux.append(np.sum(
                hist_img[nlinmin:nlinmax, ncolmin:ncolmax]
                ) * angle_correction)
        # Define fluxes with gains correction.
        self.flux = np.array(flux) * self.gains
        return self.flux

    def pair_difference(self) -> tuple:
        """The position of the beam from pairwised blades."""
        norm = 1.0 / sum(self.flux)
        xpos = norm * (
            (self.flux[0] + self.flux[3]) -
            (self.flux[1] + self.flux[2])
        )
        ypos = norm * (
            (self.flux[0] + self.flux[1]) -
            (self.flux[2] + self.flux[3])
        )
        # Normalize to box size.
        self.xppos = xpos * 0.5 * self.windowsize[0]
        self.yppos = ypos  * 0.5 * self.windowsize[1]
        return self.xppos, self.yppos

    def cross_difference(self) -> tuple:
        """The position of the beam from crossed blades."""
        to_bi = (
            (self.flux[0] - self.flux[2]) /
            (self.flux[0] + self.flux[2])
             )
        ti_bo = (
            (self.flux[1] - self.flux[3]) /
            (self.flux[1] + self.flux[3])
            )
        xpos = (to_bi - ti_bo) * 0.5 * self.windowsize[0]
        ypos = (to_bi + ti_bo) * 0.5 * self.windowsize[1]
        return xpos, ypos

    def neighbour_difference(self) -> list:
        """The flux difference between neighbour blades."""
        ftop = (
            (self.flux[0] - self.flux[1]) /
            (self.flux[0] + self.flux[1])
            )
        fbottom = (
            (self.flux[3] - self.flux[2]) /
            (self.flux[3] + self.flux[2])
            )
        fleft = (
            (self.flux[1] - self.flux[2]) /
            (self.flux[1] + self.flux[2])
            )
        fright = (
            (self.flux[0] - self.flux[3]) /
            (self.flux[0] + self.flux[3])
            )
        return [ftop, fright, fbottom, fleft]

    def _blades_intervals(self) -> list:
        """Find the boundaries of the surrounding box around each blade.

        Returns:
            bladesintervals (list): intervals defining the surrounding box.
                Each element in the list is a blade's corner's coordinate.
        """
        bladesintervals = list()
        for blade in self.bladescoordinates:
            xmin, xmax = np.min(blade[:, 0]), np.max(blade[:, 0])
            ncolmin = int(max(xmin / self.pixelsize, 0))
            ncolmax = int(min(xmax / self.pixelsize, self.nbins[1]))
            ymin, ymax = np.min(blade[:, 1]), np.max(blade[:, 1])
            nlinmin = int(max(ymin / self.pixelsize, 0))
            nlinmax = int(min(ymax / self.pixelsize, self.nbins[0]))
            bladesintervals.append([[nlinmin, ncolmin], [nlinmax, ncolmax]])
        return bladesintervals
