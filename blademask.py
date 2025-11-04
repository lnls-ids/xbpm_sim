"""Class to create a mask of "blades" in a XBPM.

A square numpy array is created with the dimensions nbins_h x nbins_v,
corresponding to the number of pixels in horizontal and
vertical directions respectively.
"""

import numpy as np
# import matplotlib.pyplot as plt
from copy import deepcopy


class BladeMask:
    """Create a mask of "blades" in a XBPM.

    A square numpy array is created with the dimensions nbins[0] (horiz) x
    nbins[1] (vert), corresponding to the number of pixels in horizontal and
    vertical directions.

    The standard values are based on real dimensions of the structures, in mm.

    Version: 2025-08-14.
    """
    def __init__(self, gprm):
        """Define the main parameters of the mask."""
        self.windowsize = gprm['windowsize']
        self.pixelsize = gprm['pixelsize']
        self.nbins = gprm['nbins']
        self.corneroffset = gprm['corneroffset']
        self.bladelength = gprm['bladelength']
        self.bladethickness = gprm['bladethickness']
        self.phi = self._degtorad(gprm['phi'])
        self.bladescoordinates = self._blades_coordinates()
        self.maskarray = self.mask_array()

    def mask_array(self):
        """Create a mask to weight the intersection of pixels and blades.

        The mask values are the percentage of intersection of the blade with
        the pixel array (screen image). If a pixel is fully inside the blade
        area, the pixel value is 1, otherwise it is proportional to the
        intersected area.

        Notice that the calculations are made in real coordinates and then
        projected into the pixel array.

        Warning: the usual indexing of an array is from top to bottom, but
        the coodrinates were chosen accordingly to the Cartesian systems,
        so the current indexing is reversed.
        """
        # Create mask array with zeros.
        self.mask = np.zeros((self.nbins[0], self.nbins[1]))

        # Scale to array units.
        pxnorm = 1. / self.pixelsize

        # Run through blades to assign weights to pixels.
        for ib, blade in enumerate(self.bladescoordinates):
            # Equations of lines joining the corners of the blade.
            bladeequations = [
                self._edge_line(blade[0] * pxnorm, blade[1] * pxnorm),
                self._edge_line(blade[1] * pxnorm, blade[2] * pxnorm),
                self._edge_line(blade[2] * pxnorm, blade[3] * pxnorm),
                self._edge_line(blade[3] * pxnorm, blade[0] * pxnorm),
            ]

            # Set horizontal (x) interval, and top and bottom line
            # equations for each interval.
            intervals, top_eq, bot_eq = list(), list(), list()

            # Even and odd-number blades are defined within different
            # intervals by their respective corners.
            if ib % 2 == 0:
                intervals.append([blade[0][0], blade[1][0]])
                top_eq.append(bladeequations[3])
                bot_eq.append(bladeequations[0])
                intervals.append([blade[1][0], blade[3][0]])
                top_eq.append(bladeequations[3])
                bot_eq.append(bladeequations[1])
                intervals.append([blade[3][0], blade[2][0]])
                top_eq.append(bladeequations[2])
                bot_eq.append(bladeequations[1])
            else:
                intervals.append([blade[3][0], blade[2][0]])
                top_eq.append(bladeequations[2])
                bot_eq.append(bladeequations[3])
                intervals.append([blade[2][0], blade[0][0]])
                top_eq.append(bladeequations[1])
                bot_eq.append(bladeequations[3])
                intervals.append([blade[0][0], blade[1][0]])
                top_eq.append(bladeequations[1])
                bot_eq.append(bladeequations[0])

            # Treat corners of the blade: check whether bottom and top lines
            # cross each other inside the pixel.
            self._pixel_corner_weight(blade, bladeequations)

            # Run vertically in each interval to assign weight to pixels:
            # treat bulk and borders of the blades, except corners.
            for jj, interval in enumerate(intervals):
                # Define the interval corresponding to current blade. Take the
                # limits of the box, [0, windowsize[0/1]], into consideration.
                # xA, xB = max(interval[0], 0), min(interval[1], self.windowsize)
                # # xA, xB = interval[0], interval[1]

                # Horizontal range to be scanned, in pixels coordinates, not
                # physical, to guarantee each pixel will be analysed only once.
                # Nx = int((xB - xA) / pixelsize)  # Number of intervals
                # (round up).
                ncolmin = max(0, round(interval[0] * pxnorm))
                ncolmax = min(round(interval[1] * pxnorm), self.nbins[1])

                # Run over all pixels inside intervals defined by
                # blade's corners.
                # Horizontal range to be scanned.
                for ncol in range(ncolmin, ncolmax):
                    # Vertical limits.
                    ymin = min(
                        self._linear(ncol, *bot_eq[jj]),
                        self._linear(ncol + 1, *bot_eq[jj]),
                    )
                    ymax = max(
                        self._linear(ncol, *top_eq[jj]),
                        self._linear(ncol + 1, *top_eq[jj]),
                    )
                    # Limits in array coordinates.
                    nlinmin, nlinmax = (int(ymin), int(ymax + 2))

                    # Vertical range to be scanned.
                    nlinrange = range(max(nlinmin, 0),
                                      min(nlinmax, self.nbins[0]))

                    for nlin in nlinrange:
                        self.mask[nlin, ncol] = self.pixel_weight(
                            nlin, ncol, bot_eq[jj], top_eq[jj]
                        )

        return self.mask

    def _blades_coordinates(self):
        """Create a list of the coordinates of the four blades.

        Each element of the list is an array with the coordinates of the
        corners of the blade. Each blade is initially created at the center
        of the coordinates system, then it is rotated and shifted accodingly.
        At this stage, the coordinates and measures are set in mm. They are
        transformed onto array indices afterwards.

        phi is the azimuthal angle (in rad) of the blades; not to be mistaken
        with the theta angle, relative to the cut edge of the blades,
        upon which the x-ray is incident.

        Returns:
            blades (list): each blades' corners' coordinates.
        """
        # Define the blade corners. The order is counterclockwise,
        # starting from the bottom left of the blade. The initial position
        # of the blade is: length in vertical, thickness in horziontal,
        # geometric center at zero. The blade is thereafter rotated and
        # shifted to its final location.
        halfthickb = 0.5 * self.bladethickness
        halfheightb = 0.5 * self.bladelength
        bladecoordinates = [[-halfthickb, -halfheightb],
                            [halfthickb, -halfheightb],
                            [halfthickb, halfheightb],
                            [-halfthickb, halfheightb]]

        # List of all four blades.
        blades = list(range(4))

        # Rotate two blades counterclockwise.
        rotmat = self._matrix_rotation(self.phi)
        #
        rotbladecoord = list()
        for bc in bladecoordinates:
            rotbladecoord.append(np.matmul(rotmat, bc))
        # Create copies of the rotated blades.
        blades[1] = deepcopy(np.array(rotbladecoord))
        blades[3] = deepcopy(blades[1])

        # Rotate two blades clockwise.
        rotmat = self._matrix_rotation(-self.phi)
        #
        rotbladecoord = list()
        for bc in bladecoordinates:
            rotbladecoord.append(np.matmul(rotmat, bc))
        # Create copies of the rotated blades.
        blades[0] = np.array(rotbladecoord)
        blades[2] = deepcopy(blades[0])

        # Find the coordinates of the appropriate blade's corner to be set on
        # the box boundaries and shift the blade accordingly.
        # Map: Blades[a][b][c] corresponds to
        # a = # blade (0... 3);
        # b = # corner (counterclockwise);
        # c = coordinate x (0) or y (1).
        #
        # An offset relative to the corner is added as a horizontal shift,
        # since blades are not necessarily located at the corners of the box.

        # Bottom, left blade.
        dx = -blades[0][0][0] + self.corneroffset
        dy = -blades[0][0][1]
        blades[0] += np.array((dx, dy))

        # Bottom, right blade.
        dx = self.windowsize[0] - blades[1][1][0] - self.corneroffset
        dy = -blades[1][1][1]
        blades[1] += np.array((dx, dy))

        # Top, right blade.
        dx = self.windowsize[0] - blades[2][2][0] - self.corneroffset
        dy = self.windowsize[1] - blades[2][2][1]
        blades[2] += np.array((dx, dy))

        # Top, left blade.
        dx = -blades[3][3][0] + self.corneroffset
        dy = self.windowsize[1] - blades[3][3][1]
        blades[3] += np.array((dx, dy))

        return blades

    def _pixel_area(self, key, x0, xf, ya, yb):
        """Wrapper function to access pixel area dictionary.

        The letters in keys indicate the pixel borders crossed by
        the edge's line, namely, T: top, B: bottom, L: left, R: right.
        """
        if key == 'TB':
            return 0.5 * (x0 + xf)

        if key == 'LR':
            return 0.5 * (ya + yb)

        if key == 'BL':
            return 0.5 * x0 * ya

        if key == 'BR':
            return 0.5 * x0 * yb

        if key == 'TL':
            return 1. - (0.5 * xf * (1 - ya))

        if key == 'TR':
            return 1. - 0.5 * (1 - xf) * (1 - yb)

    def pixel_weight(self, nlin, ncol, bot_eq, top_eq):
        """Calculate the area of intersection of blade over the pixel.

        The intersected area is the weight of the pixel to be considered.

        Obs.: in this method, the pixel area is set to one and the
        bottom-left corner of the pixel is considered as (0, 0) coordinate.
        """
        # Shift equations for current pixel position (set bottom-left
        # coordinate to zero).
        boteq = [bot_eq[0], bot_eq[1] + bot_eq[0] * ncol - nlin]
        topeq = [top_eq[0], top_eq[1] + top_eq[0] * ncol - nlin]

        # Ordinates where the equations cross the interval.
        yabot, ybbot = self._linear(0, *boteq), self._linear(1, *boteq)
        yatop, ybtop = self._linear(0, *topeq), self._linear(1, *topeq)

        # If corners are outside the blade.
        if (yatop <= 0 and ybtop <= 0) or (yabot >= 1 and ybbot >= 1):
            return 0

        # If all corners are inside the blade.
        if yabot < 0 and ybbot < 0 and yatop > 1 and ybtop > 1:
            return 1.

        # If bottom line crosses the pixel.
        x0 = -yabot / (ybbot - yabot)
        xf = (1 - yabot) / (ybbot - yabot)
        if (0 <= x0 <= 1 or 0 <= xf <= 1 or
            0 <= yabot <= 1 or 0 <= ybbot <= 1):
            lineq = boteq
            toparea = True
        else:
            # Top line crosses the pixel.
            lineq = topeq
            toparea = False

        # Ordinates where the line crosses the interval [xa, xb].
        ya = self._linear(0, *lineq)
        yb = self._linear(1, *lineq)
        # Abscissas where the line crosses the bottom/top of the
        # pixel ordinates.
        x0 = -ya / (yb - ya)
        xf = (1. - ya) / (yb - ya)

        # Classify the case according to crossing lines.
        cross = ''
        if 0 <= xf <= 1:
            cross += 'T'
        if 0 <= x0 <= 1:
            cross += 'B'
        if 0 <= ya <= 1:
            cross += 'L'
        if 0 <= yb <= 1:
            cross += 'R'

        # Calculate intersection area (blade over pixel).
        area0 = self._pixel_area(cross[:2], x0, xf, ya, yb)
        if toparea:
            return 1.0 - area0
        else:
            return area0

    def _pixel_corner_weight(self, corners, blade_eqs):
        """Assign weights to the pixels at blade's corners."""
        for ic, corner in enumerate(corners):
            xc, yc = corner

            # Skip if corner lies outside the box array.
            if (xc < 0 or xc > self.windowsize[0] or
                yc < 0 or yc > self.windowsize[1]):
                continue

            # Identify pixel (in array) coordinates. It must be inbounds.
            nx = min(int(xc / self.pixelsize), self.nbins[0] - 1)
            ny = max(int(yc / self.pixelsize), self.nbins[1] - 1)
            if nx < 0:
                nx = 0
            if ny < 0:
                ny = 0
            xa, y0 = nx * self.pixelsize, ny * self.pixelsize
            xb, yf = xa + self.pixelsize, y0 + self.pixelsize

            # Calculate crossing point.
            topeq, boteq = blade_eqs[(ic - 1) % 4], blade_eqs[ic]
            xcross = (topeq[1] - boteq[1]) / (boteq[0] - topeq[0])
            ycross = self._linear(xcross, *topeq)

            # Next step, TO BE DONE: find the intersection area.
            # Idea: besides the crossing point coordinates (above), find the
            # side of the blade which intersects the pixel, calculate the
            # height of the triangle formed by the this blade corner
            # (crossing point) and the corresponding pixel edge intercepted
            # by the blade, calculate the area of the triangle and discount
            # the other two minor triangles outside the pixel.

            # Temporary dummy line to avoid warnings from CodeVS.
            ycross += xb + yf

            # WARNING: Temporary!
            self.mask[nx, self.nbins[1] - 1 - ny] = 0.25

        return

    """Mathematical methods: linear equation, integral of a line,
    matrix rotation in 2D."""

    def _linear(self, x, a, b):
        """Straight line equation."""
        return a * x + b

    def _edge_line(self, pp, qq):
        """Calculate the line equation joining two points, P and Q.

        Args:
            pp (tuple): first point;
            qq (tuple): second point.

        Returns:
            [a, b] (list): coefficients of the line ax + b.
        """
        a = (qq[1] - pp[1]) / (qq[0] - pp[0])
        b = pp[1] - a * pp[0]
        return [a, b]

    def _integral_linear(self, coefficients, interval):
        """Integral of a linear equation.

        Args:
            coefficients (tuple): coefficients of the line ax + b;
            interval (tuple): the interval of integration.

        Returns:
            the value of the integral in the interval.

        """
        a, b = coefficients
        xa, xb = interval
        return (xb - xa) * (a * (xa + xb) / 2.0 + b)

    def _matrix_rotation(self, phi):
        """Rotation matrix.

        Args:
            phi (float): angle (in rad).

        Returns:
            (numpy array) 2x2 rotation matrix.
        """
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        return np.array([[cphi, -sphi], [sphi, cphi]])

    def _degtorad(self, phi):
        """Convert from degree to radian."""
        return np.pi / 180 * phi
