"""
Work in progress. This file should be updated.
"""

import numpy as np
import astropy.units as au
import astropy.constants as ac

from astropy.convolution import convolve, Box1DKernel

from .read_output import read_hst

def f_tau(tau_perp):
    """
    Calculate the correction factor f(tau) = 4pi*J_FUV/Sigma_FUV
    for the mean intensity at z = 0 in an infinitely extended uniform slab.
    
    Ref: Ostriker et al. (2010)

    Parameters
    ----------
       tau_perp: float
          dust optical depth of emitting slab in the vertical direction
    
    Returns
    -------
       f_tau: float
          correction factor
    """
    
    from scipy.special import expn
    return (1.0 - expn(2, 0.5*tau_perp))/tau_perp

def analyze_hst(filename):
    """
    Read TIGRESS history dataframe and add various derived fields to it.

    Work in progress.
    - Need to take simulation parameters as input.
    """
    
    Lx = 1024.0*au.pc
    Ly = 1024.0*au.pc
    ## Lz = 2048.0*au.pc
    Lz = 1796.0*au.pc
    vol = Lx*Ly*Lz
    Omega = 28.0*au.km/au.s/au.kpc
    torb_Myr = ((2.0*np.pi)/Omega).to('Myr').value
    kappa_d = (1000.0*au.cm**2/au.g).to('pc**2/Msun')

    hst = read_hst(filename)
    
    # constants from astropy.units
    #dunit = 1.4271*ac.m_p/au.cm**3
    #munit = (dunit*au.pc**3).to('Msun')
    #tunit = (1.0*au.pc/(au.km/au.s)).to('Myr')

    ## Use constants defined in athena-tigress/src/units.c
    dunit = 1.4271*1.6733e-24*au.g/au.cm**3
    munit = (dunit*(3.085678e18*au.cm)**3)/(1.9891e33)*au.Msun
    tunit = (3.085678e18/1e5)/(1e6*3.155815e7)*au.Myr
    lunit = 3.085678e18*au.cm
    vunit = lunit/tunit
    punit = (dunit*vunit**2).to('g/(cm*s**2)')
    punit_over_kB = (punit/ac.k_B).cgs
    
    #print lunit, vunit, tunit, punit
    
    phase = ['c', 'u', 'w', 'h1', 'h2']
    
    # time in code units to Myr
    hst['time'] = hst['time']*tunit.value
    hst['t_over_torb'] = hst['time']/torb_Myr
    
    # scale height in parsec
    hst['H'] = np.sqrt(hst['H2']/hst['mass']) # whole medium
    for ph in phase:
        hst['H' + ph] = np.sqrt(hst['H2' + ph]/hst['M' + ph])

    # total gas mass in units of Msun
    hst['Mass_tot'] = hst['mass']*vol.value*munit.value
    hst['Mass_star'] = hst['msp']*vol.value*munit.value
    
    # surface density of total gas in units of Msun/pc**2
    hst['Surf_tot'] = hst['Mass_tot']/Lx/Ly
    
    hst['tau_perp'] = hst['Surf_tot']*kappa_d.value
    hst['ftau'] = f_tau(hst['tau_perp'])
    
    hst['M_2p'] = hst['Mc'] + hst['Mu'] + hst['Mw']
    
    # velocity dispersions of the C+U+W medium (with T < 20000K) in units of km/s
    hst['sigma_x2p'] = np.sqrt(2.0*hst['x1KE_2p']/hst['M_2p'])
    hst['sigma_y2p'] = np.sqrt(2.0*hst['x2KE_2p']/hst['M_2p'])
    hst['sigma_z2p'] = np.sqrt(2.0*hst['x3KE_2p']/hst['M_2p'])
    
    #velocity dispersion of whole medium
    hst['sigma_z']=np.sqrt(2.0*hst['x3KE']/hst['mass'])
    
    # average number density (cm^-3), total mass (Msun), mass, volume fraction
    for ph in phase:
        hst['n' + ph] = hst['M' + ph]/(1.e-15+ hst['V' + ph])
        # Mass in units of Msun
        hst['Mass_' + ph] = hst['M' + ph]*vol.value*munit.value
        hst['Mfrac_' + ph] = hst['Mass_' + ph]/hst['Mass_tot']

    
    # midplane pressure (all gas)
    hst['Pth'] = hst['Pth']*punit_over_kB.cgs.value
    hst['Pth_smooth'] = convolve(hst['Pth'].values,Box1DKernel(20))
    hst['Pturb'] = hst['Pturb']*punit_over_kB.cgs.value
    hst['Pturb_smooth'] = convolve(hst['Pturb'].values,Box1DKernel(20))

    # two-phase midplane pressure (T < Twarm = 5050 K)
    hst['Pth_2p'] = hst['Pth_2p']*punit_over_kB.cgs.value/hst['Vmid_2p']
    hst['Pth_2p_smooth'] = convolve(hst['Pth_2p'].values,Box1DKernel(20))
    hst['Pturb_2p'] = hst['Pturb_2p']*punit_over_kB.cgs.value/hst['Vmid_2p']
    hst['Pturb_2p_smooth'] = convolve(hst['Pturb_2p'].values,Box1DKernel(20))
    
    # two-phase midplane density
    hst['nmid_2p'] = hst['nmid_2p']/hst['Vmid_2p']
    
    # correct history variables for two phase medium (temporary fix)
    try:
        hst['heat_ratio_mid_2p'] = hst['heat_ratio_mid_2p']/hst['Vmid_2p_corr']
        hst['nmid_2p_corr'] = hst['nmid_2p_corr']/hst['Vmid_2p_corr']
        hst['Pturb_2p_corr'] = hst['Pturb_2p_corr']*punit_over_kB.cgs.value/hst['Vmid_2p_corr']
        hst['Pth_2p_corr'] = hst['Pth_2p_corr']*punit_over_kB.cgs.value/hst['Vmid_2p_corr']
    except KeyError:
        pass
    
    return hst
