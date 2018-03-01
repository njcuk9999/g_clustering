#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2018-03-01 at 16:58

@author: cook
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm
import warnings


# =============================================================================
# Define variables
# =============================================================================

# -----------------------------------------------------------------------------

# =============================================================================
# Define functions
# =============================================================================
def convert_XYZ(ra, dec, distance):
    """
    Convert ra, dec and distance to X Y and Z

    :param ra: numpy array of astropy quantities or floats, the right ascensions
               if floats, units must be in degrees
    :param dec: numpy array of astropy quantities or floats, the declinations
                if floats, units must be in degrees
    :param distance: numpy array of astropy quantities or floats, the distances
                     if floats, units must be in parsecs

    :return x: numpy array of astropy quantities, X in parsecs
    :return y: numpy array of astropy quantities, Y in parsecs
    :return z: numpy array of astropy quantities, Z in parsecs
    """

    # deal with ra units - should be in degrees
    ra = __validate_units__(ra, u.deg)
    # deal with dec units - should be in degrees
    dec = __validate_units__(dec, u.deg)
    # deal with distance units - should be in parsecs
    distance = __validate_units__(distance, u.parsec)
    # get coordinate array
    coords = SkyCoord(ra=ra, dec=dec, frame='icrs', unit=u.deg)
    # convert to galactic longitude and latitude
    l = coords.galactic.l.radian
    b = coords.galactic.b.radian
    # get X Y and Z
    x = distance * np.cos(b) * np.cos(l)
    y = distance * np.cos(b) * np.sin(l)
    z = distance * np.sin(b)
    # return x, y, z
    return x, y, z


def convert_ra_dec_distance(x, y, z):
    """
    Convert x, y and z into ra, dec and distance

    :param x: numpy array of astropy quantities or floats, X, if floats, units
              must be in parsecs
    :param y: numpy array of astropy quantities or floats, Y, if floats, units
              must be in degrees
    :param z: numpy array of astropy quantities or floats, Z, if floats, units
              must be in degrees

    :return ra: numpy array of astropy quantities, right ascension in degrees
    :return dec: numpy array of astropy quantities, declination in degrees
    :return distance: numpy array of astropy quantities, distance in parsecs
    """

    # deal with x units - should be in parsecs
    x = __validate_units__(x, u.parsec)
    # deal with y units - should be in parsecs
    y = __validate_units__(y, u.parsec)
    # deal with z units - should be in parsecs
    z = __validate_units__(z, u.parsec)
    # get distance
    distance = np.sqrt(x**2 + y**2 + z**2)
    # get l and b in radians
    lrad = np.arctan2(y, x)
    brad = np.arcsin(z/distance)
    # get coordinate array
    coords = SkyCoord(lrad, brad, frame='galactic', unit=u.rad)
    # convert to ra and dec
    ra = coords.icrs.ra.deg
    dec = coords.icrs.dec.deg
    # return ra, dec, distance
    return ra, dec, distance


def __validate_units__(value, unit):
    if hasattr(value, 'unit'):
        value = value.to(unit)
    else:
        value = value * unit
    return value

# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    print("Hello World!")

# =============================================================================
# End of code
# =============================================================================
