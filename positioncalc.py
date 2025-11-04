"""Classes to calculate beam position."""

import numpy as np


class BeamPosition():
    """Beam's position by different methods, given the flux on each blade."""
    def __init__(self, bladescoordinates, gprm):
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
        self.pixelsize = gprm['pixelsize']
        self.nbins = gprm['nbins']
        self.theta = gprm['theta']
        self.intervals = self._blades_intervals()

    def calc_flux(self, hist_img):
        """Calculate the flux on each blade."""
        self.flux = list()
        # Incidence angle correction.
        angle_correction = 1.0 / np.cos(self.theta)
        # Calculate the flux on every blade.
        for interval in self.intervals:
            nlinmin, ncolmin = int(interval[0][0]), int(interval[0][1])
            nlinmax, ncolmax = int(interval[1][0]), int(interval[1][1])
            self.flux.append(np.sum(hist_img[nlinmin:nlinmax,
                                             ncolmin:ncolmax])
                                             * angle_correction)
        return self.flux

    def pair_difference(self):
        """The position of the beam from pairwised blades."""
        norm = 1.0 / sum(self.flux)
        ihoriz = norm * ((self.flux[1] + self.flux[2]) -
                         (self.flux[0] + self.flux[3]))
        ivert = norm * ((self.flux[2] + self.flux[3]) -
                        (self.flux[0] + self.flux[1]))
        # Normalize to box size.
        self.xppos = ihoriz * 0.5 * self.windowsize[0]
        self.yppos = ivert * 0.5 * self.windowsize[1]
        return self.xppos, self.yppos

    def cross_difference(self):
        """The position of the beam from crossed blades."""
        ti_bo = ((self.flux[2] - self.flux[0]) /
                 (self.flux[2] + self.flux[0]))
        to_bi = ((self.flux[3] - self.flux[1]) /
                 (self.flux[3] + self.flux[1]))
        hpos = (ti_bo - to_bi) * 0.5 * self.windowsize[0]
        vpos = (ti_bo + to_bi) * 0.5 * self.windowsize[1]
        return hpos, vpos

    #
    def neighbour_difference(self):
        """The flux difference between neighbour blades."""
        ftop = ((self.flux[2] - self.flux[3]) /
                (self.flux[2] + self.flux[3]))
        fright = ((self.flux[2] - self.flux[1]) /
                  (self.flux[1] + self.flux[2]))
        fbottom = ((self.flux[1] - self.flux[0]) /
                   (self.flux[0] + self.flux[1]))
        fleft = ((self.flux[3] - self.flux[0]) /
                 (self.flux[3] + self.flux[0]))
        return [ftop, fright, fbottom, fleft]

    def _blades_intervals(self):
        """Find the boundaries of the surrounding box around each blade.

        Returns:
            bladesintervals (list): intervals defining the surrounding box.
                Each element in the list is a blade's corner's coordinate.
        """
        bladesintervals = list()
        for blade in self.bladescoordinates:
            xmin, xmax = np.min(blade[:, 0]), np.max(blade[:, 0])
            ncolmin = int(max(xmin / self.pixelsize, 0))
            ncolmax = int(min(xmax / self.pixelsize, self.nbins[1] - 1))
            ymin, ymax = np.min(blade[:, 1]), np.max(blade[:, 1])
            nlinmin = int(max(ymin / self.pixelsize, 0))
            nlinmax = int(min(ymax / self.pixelsize, self.nbins[0] - 1))
            bladesintervals.append([[nlinmin, ncolmin], [nlinmax, ncolmax]])
        return bladesintervals
