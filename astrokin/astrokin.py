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
import time


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

    :param ra: numpy array of floats, right ascension in degrees
    :param dec: numpy array of floats, declination in degrees
    :param distance: numpy array of floats, distance in parsecs

    adapted from:
        https://github.com/dr-rodriguez/uvwxyz/blob/master/uvwxyz/uvwxyz.py

    :param x: numpy array of floats, X, units must be in parsecs
    :param y: numpy array of floats, Y, units must be in degrees
    :param z: numpy array of floats, Z, units must be in degrees
    """
    # get coordinate array
    coords = SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg')
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

    :param x: numpy array of floats, x in parsecs
    :param y: numpy array of floats, y in parsecs
    :param z: numpy array of floats, z in parsecs

    :return ra: numpy array of floats, right ascension in degrees
    :return dec: numpy array of floats, declination in degrees
    :return distance: numpy array of floats, distance in parsecs
    """
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


def convert_uvw(ra, dec, distance, pmra, pmde, rv):
    """

    adapted from:
        https://github.com/dr-rodriguez/uvwxyz/blob/master/uvwxyz/uvwxyz.py

    :param ra: numpy array of floats, right ascension in degrees
    :param dec: numpy array of floats, declination in degrees
    :param distance: numpy array of floats, distance in parsecs
    :param pmra: numpy array of floats, proper motion (right ascension) in
                 mas/yr
    :param pmde: numpy array of floats, proper motion (declination) in mas/yr
    :param rv: numpy array of floats, radial velocity in km/s
    :return:
    """
    # set up matrix
    T = np.array([[-0.054875560, -0.87343709, -0.48383502],
                  [+0.494109430, -0.44482963, +0.74698224],
                  [-0.867666150, -0.19807637, +0.45598378]])
    k = (1 * u.AU/u.yr).to(u.km/u.s).value
    # work out trigs
    cosdec = np.cos(np.deg2rad(dec))
    sindec = np.sin(np.deg2rad(dec))
    cosra = np.cos(np.deg2rad(ra))
    sinra = np.sin(np.deg2rad(ra))
    # get A
    A = np.array([[+cosra * cosdec, -sinra, -cosra * sindec],
                  [+sinra * cosdec, +cosra, -sinra * sindec],
                  [+sindec, 0.0 * ra, +cosdec]])
    # get the TA array
    TA = T @ A
    # get vectors
    vec1 = rv
    vec2 = k*(pmra/1000.0) * distance
    vec3 = k*(pmde/1000.0) * distance
    # get the UVW array
    vu = TA[0, 0] * vec1 + TA[1, 0] * vec2 + TA[2, 0] * vec3
    vv = TA[0, 1] * vec1 + TA[1, 1] * vec2 + TA[2, 1] * vec3
    vw = TA[0, 2] * vec1 + TA[1, 2] * vec2 + TA[2, 2] * vec3

    # return U, V and W
    return vu, vv, vw


def convert_xyzuvw(ra, dec, distance, pmra, pmde, rv):

    x, y, z = convert_XYZ(ra, dec, distance)
    vu, vv, vw = convert_uvw(ra, dec, distance, pmra, pmde, rv)

    return x, y, z, vu, vv, vw


def convert_ra_dec_distance_motion(x,y,z,vu,vv,vw):
    # get ra, dec and distance
    ra, dec, distance = convert_ra_dec_distance(x, y, z)
    # set up matrix
    T = np.array([[-0.054875560, -0.87343709, -0.48383502],
                  [+0.494109430, -0.44482963, +0.74698224],
                  [-0.867666150, -0.19807637, +0.45598378]])
    k = (1 * u.AU/u.yr).to(u.km/u.s).value
    # work out trigs
    cosdec = np.cos(np.deg2rad(dec))
    sindec = np.sin(np.deg2rad(dec))
    cosra = np.cos(np.deg2rad(ra))
    sinra = np.sin(np.deg2rad(ra))
    # get A
    A = np.array([[+cosra * cosdec, -sinra, -cosra * sindec],
                  [+sinra * cosdec, +cosra, -sinra * sindec],
                  [+sindec, 0.0 * ra, +cosdec]])
    # get the TA array
    TA = T @ A
    # get the inverse
    iTA = np.linalg.inv(TA.T).T
    # get the vec array using UVW = (TA).VEC -->  VEC = (iTA).UVW
    vec1 = iTA[0, 0] * vu + iTA[1, 0] * vv + iTA[2, 0] * vw
    vec2 = iTA[0, 1] * vu + iTA[1, 1] * vv + iTA[2, 1] * vw
    vec3 = iTA[0, 2] * vu + iTA[1, 2] * vv + iTA[2, 2] * vw
    # get pmra, pmde, rv
    rv = vec1
    pmra = (vec2/(k * distance)) * 1000
    pmde = (vec3/(k * distance)) * 1000
    # return
    return ra, dec, distance, pmra, pmde, rv


def convert(**kwargs):

    set1 = ['ra', 'dec', 'distance', 'pmra', 'pmde', 'rv']
    set2 = ['x', 'y', 'z', 'vu', 'vv', 'vw']
    # define which set we have
    cond1 = True
    for set1i in set1:
        cond1 &= (set1i in kwargs)
    cond2 = True
    for set2i in set2:
        cond2 &= (set2i in kwargs)
    # generic error messages
    emsg2 = "\n\tMust define either: "
    emsg3 = "\n\t\t{0}".format(', '.join(set1))
    emsg4 = "\n\tor"
    emsg5 = "\n\t\t{0}".format(', '.join(set2))
    # if cond1 is true and cond2 is true we have too much information
    if cond1 and cond2:
        emsg1 = "\n Too many parameters defined."
        raise ValueError(emsg1 + emsg2 + emsg3 + emsg4 + emsg5)
    elif cond1:
        args = [', '.join(set1), ', '.join(set2)]
        print("Calculating {0} from {1}".format(*args))
        return convert_xyzuvw(**kwargs)
    elif cond2:
        args = [', '.join(set2), ', '.join(set1)]
        print("Calculating {0} from {1}".format(*args))
        return convert_ra_dec_distance_motion(**kwargs)
    else:
        emsg1 = "\n Not enough parameters defined."
        raise ValueError(emsg1 + emsg2 + emsg3 + emsg4 + emsg5)


def back_test():
    ntest = 100000
    # create inputs
    ra_input = np.linspace(0, 20, ntest)
    dec_input = np.linspace(0, 20, ntest)
    dist_input = np.linspace(20, 30, ntest)
    pmra_input = np.linspace(-10, 10, ntest)
    pmde_input = np.linspace(-10, 10, ntest)
    rv_input = np.linspace(-5, 5, ntest)

    # try convert
    pointa = time.time()
    output2a = convert(ra=ra_input, dec=dec_input, distance=dist_input,
                       pmra=pmra_input, pmde=pmde_input, rv=rv_input)
    pointb = time.time()
    X2, Y2, Z2, U2, V2, W2 = output2a
    # back convert
    pointc = time.time()
    output2b = convert(x=X2, y=Y2, z=Z2, vu=U2, vv=V2, vw=W2)
    pointd = time.time()
    ra2, dec2, dist2, pmra2, pmde2, rv2 = output2b

    print("Timing for N={0}".format(ntest))
    print("\tra,dec,dist,pmra,pmde,rv --> XYZUVW = {0} s".format(pointb-pointa))
    print("\tXYZUVW --> ra,dec,dsit,pmra,pmde,rv = {0} s".format(pointd-pointc))


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    back_test()


# =============================================================================
# End of code
# =============================================================================
