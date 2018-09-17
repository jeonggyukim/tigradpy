import os
import pandas as pd
import numpy as np
import astropy.units as au
import astropy.constants as ac
import glob

from .calc_lum import calc_LFUV, calc_Qi
from .read_output import read_starpar_vtk

def analyze_starpar(star):
    """
    Calculate
       - FUV luminosity of star particles with age < agemax [Myr]
       - Mass- or luminosity-weighted average z position    
       - Mass- or luminosity-weighted standard deviation of z position
       - Sigma_FUV = L_FUV_tot/(Lx*Ly)

    Parameters
       star: vtk output
    """
        
    
    muH = 1.4273
    LxLy = 1024.0*1024.0
    agemax = 40.0
    J_FUV0_cgs = 2.1e-4
    
    # 1 code mass in Msun
    Munit = (muH*au.u*(au.pc/au.cm)**3).to(au.Msun).value
    # 1 code time in Myr
    Tunit = (au.pc/(au.km/au.s)).to(au.Myr)
    
    #*float((u.Lsun/u.pc**2).in_cgs().v)

    # Select source particles with mass-weighted age < 40 Myr
    # Runaways have zero mass
    isrc = np.logical_and(star['mage']*Tunit < agemax,
                          star['mass'] != 0.0)

    if np.sum(isrc) == 0:
        return None
    
    star_ = {}
    for k in star.keys():
        if k in ('nstars', 'time'):
            continue
        elif k == 'mass':
            star_[k] = star[k][isrc] * Munit
        elif k == 'age' or k == 'mage':
            star_[k] = star[k][isrc] * Tunit
        else:
            star_[k] = star[k][isrc]
    
    star_['time'] = star['time'] * Tunit
    star_['nstars'] = np.sum(isrc)
    
    star_['L_FUV'] = calc_LFUV(star_['mass'], star_['mage'])
    star_['Qi'] = calc_Qi(star_['mass'], star_['mage'])

    #print star_['mass'], star_['mage']
    
    # center of mass, luminosity, standard deviation
    star_['z_max'] = np.max(star_['x'][:,2])
    star_['z_min'] = np.min(star_['x'][:, 2])
    star_['z_mass'] = np.average(star_['x'][:, 2], weights=star_['mass'])
    star_['z_lum'] = np.average(star_['x'][:, 2], weights=star_['L_FUV'])
    star_['stdz_mass'] = np.sqrt(np.average((star_['x'][:,2] - star_['z_mass'])**2,
                                            weights=star_['mass']))
    star_['stdz_lum'] = np.sqrt(np.average((star_['x'][:,2] - star_['z_lum'])**2,
                                           weights=star_['L_FUV']))

    star_['L_FUV_tot'] = np.sum(star_['L_FUV'])
    star_['Qi_tot'] = np.sum(star_['Qi'])
    star_['Sigma_FUV'] = star_['L_FUV_tot'] / LxLy
    
    # J_FUV without correction factor (1.0 - E_2(tau/2))/tau
    star_['Sigma_FUV_over_fourpi'] = star_['Sigma_FUV']/(4.0*np.pi)

    # # G0prime0 = Sigma_FUV/(4pi)/J_FUV,0
    # # G0prime0 = 1 when Sigma_FUV ~ 6.9 Lsun/pc^2
    # star_['G0prime0'] = star_['Sigma_FUV_over_fourpi']/J_FUV0_cgs

    return star_


def analyze_starpar_all(dirname, problem_id, force_override=False):

    fpkl = os.path.join(dirname, 'analyze_starpar_all.p')

    # Read from pickle
    if not force_override and os.path.exists(fpkl):
        return pd.read_pickle(fpkl)
    
    fstars_base = '{0:s}.????.starpar.vtk'.format(problem_id)
    fstars = glob.glob(os.path.join(dirname, fstars_base))

    keys=['time', 'nstars',
          'z_max', 'z_min', 'z_mass', 'z_lum', 'stdz_mass', 'stdz_lum',
          'L_FUV_tot', 'Sigma_FUV', 'Qi_tot']
        
    res = {}
    for k in keys:
        res[k] = []

    for i, fstar in enumerate(fstars):
        sp = read_starpar_vtk(fstar)        
        spa = analyze_starpar(sp)
        if spa is None:
            #print i,
            continue
        for k in keys:
            res[k].append(spa[k])

    df = pd.DataFrame(res)
    df.to_pickle(fpkl)
    
    return df
