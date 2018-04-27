#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CODE NAME HERE

# CODE DESCRIPTION HERE

Created on 2018-04-25 at 14:14

@author: cook
"""

import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astropy.table import Table
from astroquery.gaia import Gaia
import numpy as np
from tqdm import tqdm
import os
import warnings


# =============================================================================
# Define variables
# =============================================================================
SOURCE = 'gaiadr2.gaia_source'
COLS = '*'
WHERES = ['(parallax > 5)']
RA = 'ra'
DEC = 'dec'
SKIP = True
SAVE_LOCATION = '/scratch/Projects/Gaia_clustering/data/Gaia_5plx_chunks/'
NAME = 'Gaia_plx5'
# -----------------------------------------------------------------------------

KEEP_COLS = [
 'solution_id',
 'designation',
 'source_id',
 'random_index',
 'ref_epoch',
 'ra',
 'ra_error',
 'dec',
 'dec_error',
 'parallax',
 'parallax_error',
 'parallax_over_error',
 'pmra',
 'pmra_error',
 'pmdec',
 'pmdec_error',
 'ra_dec_corr',
 'ra_parallax_corr',
 'ra_pmra_corr',
 'ra_pmdec_corr',
 'dec_parallax_corr',
 'dec_pmra_corr',
 'dec_pmdec_corr',
 'parallax_pmra_corr',
 'parallax_pmdec_corr',
 'pmra_pmdec_corr',
 'astrometric_n_obs_al',
 'astrometric_n_obs_ac',
 'astrometric_n_good_obs_al',
 'astrometric_n_bad_obs_al',
 'astrometric_gof_al',
 'astrometric_chi2_al',
 'astrometric_excess_noise',
 'astrometric_excess_noise_sig',
 'astrometric_params_solved',
 'astrometric_primary_flag',
 'astrometric_weight_al',
 'astrometric_pseudo_colour',
 'astrometric_pseudo_colour_error',
 'mean_varpi_factor_al',
 'astrometric_matched_observations',
 'visibility_periods_used',
 'astrometric_sigma5d_max',
 'frame_rotator_object_type',
 'matched_observations',
 'duplicated_source',
 'phot_g_n_obs',
 'phot_g_mean_flux',
 'phot_g_mean_flux_error',
 'phot_g_mean_flux_over_error',
 'phot_g_mean_mag',
 'phot_bp_n_obs',
 'phot_bp_mean_flux',
 'phot_bp_mean_flux_error',
 'phot_bp_mean_flux_over_error',
 'phot_bp_mean_mag',
 'phot_rp_n_obs',
 'phot_rp_mean_flux',
 'phot_rp_mean_flux_error',
 'phot_rp_mean_flux_over_error',
 'phot_rp_mean_mag',
 'phot_bp_rp_excess_factor',
 'phot_proc_mode',
 'bp_rp',
 'bp_g',
 'g_rp',
 'radial_velocity',
 'radial_velocity_error',
 'rv_nb_transits',
 'rv_template_teff',
 'rv_template_logg',
 'rv_template_fe_h',
 'phot_variable_flag',
 'l',
 'b',
 'ecl_lon',
 'ecl_lat',
 'priam_flags',
 'teff_val',
 'teff_percentile_lower',
 'teff_percentile_upper',
 'a_g_val',
 'a_g_percentile_lower',
 'a_g_percentile_upper',
 'e_bp_min_rp_val',
 'e_bp_min_rp_percentile_lower',
 'e_bp_min_rp_percentile_upper',
 'flame_flags',
 'radius_val',
 'radius_percentile_lower',
 'radius_percentile_upper',
 'lum_val',
 'lum_percentile_lower',
 'lum_percentile_upper',
 'datalink_url',
 'epoch_photometry_url']


# =============================================================================
# Define functions
# =============================================================================
def sky_grid_wheres(min_ra=0.0, max_ra=360.0, min_dec=-90.0, max_dec=90.0,
                    bits=36, overlap=1/3600.0, ra_col='ra', dec_col='dec',
                    return_query=True):

    gridsize = np.ceil(np.sqrt(bits)+1).astype(int)

    ras = np.linspace(min_ra, max_ra, gridsize)
    decs = np.linspace(min_dec, max_dec, gridsize)

    ramin = ras[:-1] - overlap
    ramax = ras[1:] + overlap
    decmin = decs[:-1] - overlap
    decmax = decs[1:] + overlap

    # make sure ra's loop between 0 and 360
    ramin[ramin < 0] = ramin[ramin < 0] + 360
    ramin[ramin > 360] = ramin[ramin > 360] - 360
    ramax[ramax < 0] = ramax[ramax < 0] + 360
    ramax[ramax > 360] = ramax[ramax > 360] - 360

    # make sure dec's stop at +/- 90
    decmin[decmin < -90] = -90.0
    decmin[decmin > 90] = 90.0
    decmax[decmax < -90] = -90.0
    decmax[decmax > 90] = 90.0

    # generate where conditions
    wheres = []
    ra_cent, dec_cent = [], []
    ra_mins, ra_maxs = [], []
    dec_mins, dec_maxs = [], []
    for it in tqdm(range(len(ramin))):
        for jt in range(len(decmin)):
            # get iteration values
            ra1, ra2 = ramin[it], ramax[it]
            dec1, dec2 = decmin[jt], decmax[jt]
            # calculate centers
            ra_cent.append(np.mean([ra1, ra2]))
            dec_cent.append(np.mean([dec1, dec2]))
            # store edges
            ra_mins.append(ra1), ra_maxs.append(ra2)
            dec_mins.append(dec1), dec_maxs.append(dec2)
            # start of condition
            cond = '('
            # add ra term
            if ra1 < ra2:
                cond += '({0} BETWEEN {1} AND {2})'.format(ra_col, ra1, ra2)
            else:
                cond += '({0} BETWEEN {1} AND 360.0)'.format(ra_col, ra1)
                cond += ' OR '
                cond += '({0} BETWEEN 0.0 AND {1})'.format(ra_col, ra2)
            # add an AND
            cond += ' AND '
            # add dec term
            cond += '({0} BETWEEN {1} AND {2})'.format(dec_col, dec1, dec2)
            # end of condition
            cond += ')'
            # add cond to wheres
            wheres.append(cond)
    if not return_query:
        return ra_mins, ra_maxs, dec_mins, dec_maxs
    else:
        return wheres, ra_cent, dec_cent


def make_query(source, cols, wheres):

    query = 'SELECT {0}'.format(cols)
    query += '\nFROM {0}'.format(source)
    query += '\nWHERE'
    for wi, w in enumerate(wheres):
        query += '\n' + w
        if (wi + 1) != len(wheres):
            query += '\nAND '
    return query


def construct_table(rawtable, columnnames):
    table = Table()
    for col in columnnames:
        table[col] = np.array(rawtable[col])
    return table


def plot_regions(ramin, ramax, decmin, decmax):


    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, frame = plt.subplots(ncols=1, nrows=1)
    frame.set(xlim=(0, 360), ylim=(-90, 90))

    for it in range(len(ramin)):

        width = ramax[it] - ramin[it]
        xcent = np.mean([ramax[it], ramin[it]])
        x = ramin[it]

        height = decmax[it] - decmin[it]
        ycent = np.mean([decmax[it], decmin[it]])
        y = decmin[it]

        # construct rectangle
        rect = patches.Rectangle((x, y), width, height, ec='r', fill=False)
        # add to plot
        frame.add_patch(rect)
        # add centers
        frame.scatter(xcent, ycent, marker='x', s=100, color='r')

    plt.show()
    plt.close()




# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # get ra/dec grid
    print('Creating grid')
    griddata = sky_grid_wheres(bits=36, overlap=0.0,
                               ra_col=RA, dec_col=DEC)
    gridwheres, racent, deccent = griddata

    # loop around grid and query gaia
    print('Looping around grid')
    for git in range(len(gridwheres)):
        # print progress
        pargs = [racent[git], deccent[git]]
        print('\n\n' + '='*60 + '\n')
        print('\t {0} of {1}'.format(git+1, len(gridwheres)))
        print('\tGrid center  ra={0:.2f}  dec={1:.2f}'.format(*pargs))
        print('\n' + '='*60 + '\n')
        # construct_name
        name = NAME + 'ra={0:.3f}_dec={1:.3f}.vo'.format(*pargs)
        fullpath = os.path.join(SAVE_LOCATION, name)
        # check if we have already done region
        if SKIP and os.path.exists(fullpath):
            print('\tSkipping region')
            continue
        # construct where
        wheres = WHERES + [gridwheres[git]]
        # make query
        print('\tRunning query...')
        myquery = make_query(SOURCE, COLS, wheres).replace('\n', ' ')
        # create async job
        job = Gaia.launch_job_async(myquery, dump_to_file=True)
        # print job
        print(job)
        # get results
        with warnings.catch_warnings(record=True) as w:
            r = job.get_results()
            # patch table
            print('\tConstructing table...')
            rnew = construct_table(r, KEEP_COLS)
            # save table to file
            print('\tWriting table...')
            rnew.write(fullpath, format='votable', overwrite=True)

        # clean up
        del r
        del rnew

# =============================================================================
# End of code
# =============================================================================
