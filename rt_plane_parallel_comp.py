import os
import scipy
import astropy.units as au
import astropy.constants as ac
import numpy as np
import matplotlib.pyplot as plt

from .read_output import *
from .analyze_starpar import *
from .rt_plane_parallel import *
from tigradpy.rt_plane_parallel import *
from tigradpy.read_rad_lost import *

def compare_Jrad_zprof(dirname, problem_id, da, num, z0='mid', dmax=512.0):

    from scipy.integrate import cumtrapz
    
    dunit = 1.4271*1.6733e-24*au.g/au.cm**3
    munit = ((dunit*(3.085678e18*au.cm)**3)/(1.9891e33)).value*au.Msun
    tunit = (3.085678e18/1e5)/(1e6*3.155815e7)*au.Myr
    lunit = (3.085678e18*au.cm).to(au.pc)
    vunit = lunit/tunit
    Eunit = (dunit*vunit**2).to('g/(cm*s**2)')
    Eunit_over_kB = (Eunit/ac.k_B).cgs
    Eunit_cgs = Eunit.cgs.value

    dx = dy = 8.0
    Nx = Ny = 128.0
    Lx = Nx*dx
    Ly = Ny*dy
    Area = Lx*Ly
    norm_factor = Area
    c_over_4pi = ac.c.cgs.value/(4.0*np.pi)

    dz_pc = 8.0
    dz = dz_pc*lunit
    muH = (1.4272*au.u).cgs.value
    sigma_d = muH*1000.0
    kappa_d = 1000.0*au.cm**2/au.g
    Sunit = munit/lunit**2
    c_over_4pi = ac.c.cgs.value/(4.0*np.pi)
    
    fstar = '{0:s}.{1:04d}.starpar.vtk'.format(problem_id, num)
    fstar = os.path.join(dirname, fstar)
    sp = read_starpar_vtk(fstar)
    spa = analyze_starpar(sp)
    #spall = analyze_starpar_all(dirname, problem_id)
    if spa is None or spa['nstars'] is None:
        return

    # energy density in cgs units
    Erad_da = da.loc['Erad0', :, :]*Eunit_cgs/norm_factor
    # energy density in cgs units
    den_da = da.loc['d', :, :]/norm_factor
    
    z = np.array(den_da.z)
    den = np.array(den_da[:, num])
    Erad = np.array(Erad_da[:, num])

    # z0: reference point from which dust optical depth is measured
    z0_orig = z0
    if z0 == 'z_lum':
        z0 = spa['z_lum']
        z0_title = r'$z_0=\langle z_*\rangle_{L}$'
    elif z0 == 'peak':
        z0 = z[np.argwhere(Erad == Erad.max())[0][0]]
        z0_title = r'$z_0=$ peak $G_0$'    
    elif z0 == 'mid':
        z0 = 0.0
        z0_title = r'$z_0=0$'

    zmin = z0 - spa['stdz_lum']
    zmax = z0 + spa['stdz_lum']
    #zmin = z0 - spa['z_min']
    #zmax = z0 + spa['z_max']
        
    #print 'z0', z0,
    
    SF_idx = np.logical_and(z >= zmin, z < zmax)
    #print zmin,zmax
    #return spa
    
    if SF_idx.sum() == 0: # star layer is too thin
        tau_SF = 0.001
    else:
        surf = (den[SF_idx].sum()*muH*dz)*au.g/au.cm**2
        tau_SF = (surf*kappa_d).value
        print tau_SF
    z0_idx = np.where(z > z0)[0][0]
    z_ = np.insert(z, z0_idx, z0)
    den_ = np.insert(den, z0_idx, np.interp(z0, z, den))
    #print z_[z0_idx]
    zp = z_[z0_idx:]
    zm = np.flipud(z_[:z0_idx + 1])
    
    taup = cumtrapz((den_[z0_idx:]*muH), x=zp, initial=0.0)
    taup = taup*au.pc.to(au.cm)*kappa_d.value
    taum = cumtrapz((np.flipud(den_[:z0_idx + 1])*muH), x=-zm, initial=0.0)
    taum = taum*au.pc.to(au.cm)*kappa_d.value

    Eradp = Erad[z0_idx:]
    Eradm = np.flipud(Erad[:z0_idx + 1])

    half_dmax_idxp = np.argwhere(np.abs(zp - z0) > 0.5*dmax)[0][0]
    half_dmax_idxm = np.argwhere(np.abs(zm - z0) > 0.5*dmax)[0][0]
    #print zp[dmax_idxp], taup[dmax_idxp]
    #print zm[dmax_idxm], taum[dmax_idxm]

    #### analytic approximation
    Jpa = []
    Jma = []
    for taum_ in taum[1:]:
        if taum_ < 0.5*tau_SF:
            Jma.append(J_over_JUV_inside_slab(taum_, tau_SF))
        else:
            Jma.append(J_over_JUV_outside_slab(taum_, tau_SF))

    for taup_ in taup[1:]:
        if taup_ < 0.5*tau_SF:
            Jpa.append(J_over_JUV_inside_slab(taup_, tau_SF))
        else:
            Jpa.append(J_over_JUV_outside_slab(taup_, tau_SF))

    Jpa = np.array(Jpa)
    Jma = np.array(Jma)
    
    Jradp = Eradp*(ac.c.cgs.value/(4.0*np.pi))
    Jradm = Eradm*(ac.c.cgs.value/(4.0*np.pi))

    Sigma_FUV_over_fourpi_cgs = (spa['Sigma_FUV']/(4.0*np.pi)*\
                                 ((1.0*au.Lsun/au.pc**2).cgs).value)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ms = 2.0
    plt.sca(axes[0])
    plt.plot(z, den, 'ko-', ms=ms,
             label=r'$\langle n_{\rm H}\rangle$')
    plt.plot(z, Erad*c_over_4pi/2.1e-4, 'go-', ms=ms,
             label=r'$\langle G^{\prime}_0\rangle$')
    plt.plot(zm[1:], Jma*Sigma_FUV_over_fourpi_cgs/2.1e-4,
             c='gray', ls='--', lw=3.0, alpha=0.7, label='analytic')
    plt.plot(zp[1:], Jpa*Sigma_FUV_over_fourpi_cgs/2.1e-4,
             c='gray', ls='--', lw=3.0, alpha=0.7)

    plt.plot(zm, taum, 'o-', c='b', ms=ms, label=r'$\tau_{\rm d}(z<z_0)$')
    plt.plot(zp, taup, 'o-', c='r', ms=ms, label=r'$\tau_{\rm d}(z>z_0)$')
    
    plt.axvline(zmin, 0, 1, c='m', ls=':', label=r'$z_0 \pm \sigma_{z_*}$')
    plt.axvline(zmax, 0, 1, c='m', ls=':')
    plt.xlabel('z [pc]')
    plt.yscale('log')
    plt.xlim(-896, 896)
    plt.ylim(1e-2, 2e1)
#    plt.legend(bbox_to_anchor=(0.05,1.05), loc="upper left", fontsize=13)
#     l = plt.legend(bbox_to_anchor=(0, 1.05), loc="lower left", 
#                    bbox_transform=fig.transFigure, ncol=3)
    l = plt.legend(loc="upper left", fontsize=13)

    plt.sca(axes[1])
    plt.plot(taup[1:], Jradp/Sigma_FUV_over_fourpi_cgs,
             'ro-', ms=ms, label=r'$z>z_0$')
    plt.plot(taum, Jradm/Sigma_FUV_over_fourpi_cgs,
             'bo-', ms=ms, label=r'$z<z_0$')
    plt.plot(taum[1:], Jma,
             c='gray', ls='--', lw=3.0, alpha=0.7, label='analytic')
    
    plt.axvline(tau_SF/2.0, 0, 1, c='m', ls=':', label=r'$\tau_{*}/2$')
    plt.axvline(taup[half_dmax_idxp], 0, 1, c='r', ls=':',
                label=r'$\tau_{\rm d}(z_0+d_{\rm max}/2)$')
    plt.axvline(taum[half_dmax_idxm], 0, 1, c='b', ls=':',
                label=r'$\tau_{\rm d}(z_0-d_{\rm max}/2)$')
    
    plt.xlabel(r'$\tau_{\rm d} = \int_{z_0}^{z} \kappa_{\rm d}\langle\rho\rangle |dz|$')
    #plt.ylabel(r'$\langle G^{\prime}_0\rangle/\langle G^{\prime}_0\rangle_{\rm max}$')
    plt.ylabel(r'$\langle J_{\rm FUV}\rangle/(\Sigma_{\rm FUV}/4\pi)$')
    plt.legend(loc=1, fontsize=13)
    plt.xlim(0.0, 1.4)
    #plt.ylim(0.0, 1.2)

    plt.ylim(1e-2, 1e1)
    plt.yscale('log')
    
    plt.tight_layout()
    subtit = 'model: {0:s}, $t$={1:5.1f} [Myr], '.format(problem_id, sp['time'])
    plt.suptitle(subtit + ' ' + z0_title + r', $d_{{\rm max}}={0:g}$ pc'.format(dmax))
    plt.subplots_adjust(top=0.88)

    savname = os.path.join(os.path.expanduser('~'),
                           'Dropbox/tigrad/plot/comparison-rt-plane-parallel/')
    savname = os.path.join(savname, '{0:s}_z0_{1:s}_{2:04d}.png'.format(\
                    problem_id, z0_orig, num))
    plt.savefig(savname, dpi=200)
    #plt.close()
    
    #return spa, Jm

def compare_Jrad_zprof2(dirname, problem_id, num, dmax=2048.0,
                        dirname2=None, dirname3=None):

    def compute_JFUV2(z, SFUV4pi, spa, fp, fm, dz_pc):

        SFUV4pi_tot = SFUV4pi.sum()
        z0 = spa['z_lum']
        Dz = spa['stdz_lum']
        tau_SF = fp(z0 + Dz) - fp(z0 - Dz)

        if z >= z0 + Dz:
            tau = fp(z) - fp(z0)
            J = SFUV4pi_tot*J_over_JUV_outside_slab(tau, tau_SF)
        elif z <= z0 - Dz:
            tau = fm(z) - fm(z0)
            J = SFUV4pi_tot*J_over_JUV_outside_slab(tau, tau_SF)
        else:
            if z >= z0:
                tau = fp(z) - fp(z0)
            else:
                tau = fm(z) - fm(z0)
            if tau > 0.5*tau_SF:
                tau = 0.5*tau_SF
            J = SFUV4pi_tot*J_over_JUV_inside_slab(tau, tau_SF)

        return J
    
    def compute_JFUV(z, SFUV4pi, zstar, fp, fm, dz_pc):
        J = 0.0
        for SFUV4pi_, zstar_ in zip(SFUV4pi, zstar):
            Dz = 1.0*dz_pc
            tau_SF = fp(zstar_ + Dz) - fp(zstar_ - Dz)

            if z >= zstar_ + Dz:
                tau = fp(z) - fp(zstar_)
                J += SFUV4pi_*0.5*expn(1, tau)
                #J += SFUV4pi_*J_over_JUV_outside_slab(tau, tau_SF)
            elif z <= zstar_ - Dz:
                tau = fm(z) - fm(zstar_)
                J += SFUV4pi_*0.5*expn(1, tau)
                #J += SFUV4pi_*J_over_JUV_outside_slab(tau, tau_SF)
            else:
                J += SFUV4pi_*J_over_JUV_avg_slab(tau_SF)
                #J += SFUV4pi_*J_over_JUV_inside_slab(0.0, tau_SF)
                pass

        return J

    dunit = 1.4271*1.6733e-24*au.g/au.cm**3
    munit = ((dunit*(3.085678e18*au.cm)**3)/(1.9891e33)).value*au.Msun
    tunit = (3.085678e18/1e5)/(1e6*3.155815e7)*au.Myr
    lunit = (3.085678e18*au.cm).to(au.pc)
    vunit = lunit/tunit
    Eunit = (dunit*vunit**2).to('g/(cm*s**2)')
    Eunit_over_kB = (Eunit/ac.k_B).cgs
    Eunit_cgs = Eunit.cgs.value
    Area = 1024.0**2
    norm_factor = Area
    c_over_4pi = ac.c.cgs.value/(4.0*np.pi)
    dz_pc = 8.0
    dz = dz_pc*lunit.cgs.value
    muH = (1.4271*au.u).cgs.value
    sigma_d = muH*1000.0
    kappa_d = (1000.0*au.cm**2/au.g).to('pc**2/Msun')
    Sunit = munit/lunit**2
    Lunit = (munit*vunit**2/tunit)
    Lconv = Lunit.cgs.value/((1.0*au.Lsun).cgs.value)

    fstar = '{0:s}.{1:04d}.starpar.vtk'.format(problem_id, num)
    fstar = os.path.join(dirname, fstar)
    # read starpar and analyze
    sp = read_starpar_vtk(fstar)
    spa = analyze_starpar(sp)
    
    if spa is None or spa['nstars'] is None:
        return

    # read all zprof as DataArray
    da = read_zprof_all(dirname, problem_id, phase='whole', force_override=False)
    den = da.sel(fields='d')[:, num]/norm_factor
    Erad = da.sel(fields='Erad0')[:, num]*Eunit_cgs/norm_factor
    G0pr_sim = Erad*c_over_4pi/2.1e-4
    if dirname2 is not None:
        da2 = read_zprof_all(dirname2, problem_id, phase='whole', force_override=False)
        Erad2 = da2.sel(fields='Erad0')[:, num]*Eunit_cgs/norm_factor
        G0pr_sim2 = Erad2*c_over_4pi/2.1e-4

    if dirname3 is not None:
        da3 = read_zprof_all(dirname3, 'rad_8pc', phase='whole', force_override=False)
        Erad3 = da3.sel(fields='Erad0')[:, num]*Eunit_cgs/norm_factor
        G0pr_sim3 = Erad3*c_over_4pi/2.1e-4

    # cell centers
    zpc = np.array(da.z)
    zmc = np.flipud(zpc)

    # cell edges
    zpe = zpc + 0.5*dz_pc
    zme = zmc - 0.5*dz_pc

    # optical depth at cell edges
    taup = sigma_d*np.cumsum(den*dz)
    taum = sigma_d*np.cumsum(np.flipud(den*dz))

    # interpolation function
    fp = scipy.interpolate.interp1d(zpe, taup)
    fm = scipy.interpolate.interp1d(zme, taum)

    SFUV4pi = spa['L_FUV']/Area/(4.0*np.pi)*(1.0*au.Lsun/au.pc**2).cgs.value
    zstar = spa['x'][:, 2]

    # plane-parallel approximation
    JFUV_anal = []
    JFUV_anal2 = []
    for z_ in zpc:
        JFUV_anal.append(compute_JFUV(z_, SFUV4pi, zstar, fp, fm, dz_pc))
        JFUV_anal2.append(compute_JFUV2(z_, SFUV4pi, spa, fp, fm, dz_pc))

    G0pr_anal = np.array(JFUV_anal)/2.1e-4
    G0pr_anal2 = np.array(JFUV_anal2)/2.1e-4

    fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                             gridspec_kw=dict(height_ratios=[5, 1.0], hspace=0.3))
    axes = axes.flatten()
    
    plt.sca(axes[0])
    plt.plot(zpc, den, c='gray', alpha=0.8,
             label=r'$\langle n_{\rm H}\rangle$')
    plt.plot(zpc, G0pr_sim, 'ro-', ms=3.0, label=r'$\langle G_0^{\prime} \rangle$')
    if dirname2 is not None:
        plt.plot(zpc, G0pr_sim2, 'co-', ms=3.0, alpha=0.5)
    if dirname3 is not None:
        plt.plot(zpc, G0pr_sim3, 'go-', ms=3.0, alpha=0.2)

    filename = os.path.join(dirname2.rstrip('id0'), 'rad_lost.txt')
    df, da = read_rad_lost(filename, force_override=False, verbose=True)

    SFUV4pi = df['L_tot0'][num]*Lconv/Area/(4.0*np.pi)*(1.0*au.Lsun/au.pc**2).cgs.value
    SFUV4pi_resp = df['L_lost0p'][num]*Lconv/Area/(4.0*np.pi)*(1.0*au.Lsun/au.pc**2).cgs.value
    SFUV4pi_resm = df['L_lost0m'][num]*Lconv/Area/(4.0*np.pi)*(1.0*au.Lsun/au.pc**2).cgs.value
    #print SFUV4pi, SFUV4pi_resm, SFUV4pi_resp

    zmax = 256.0
    zmin = -256.0

    JFUV_anal3p = []
    JFUV_anal3m = []
    tau_SF = 0.1
    for z_ in zpc[zpc > zmax]:
        taup = fp(z_) - fp(zmax)
        JFUV_anal3p.append(SFUV4pi_resp*expn(2, taup))
        # if taup > 0.5*tau_SF:
        #     JFUV_anal3p.append(SFUV4pi_resp*J_over_JUV_outside_slab(taup, tau_SF))
        # else:
        #     JFUV_anal3p.append(SFUV4pi_resp*J_over_JUV_inside_slab(taup, tau_SF))

    for z_ in zmc[zmc < zmin]:
        taum = fm(z_) - fm(zmin)
        JFUV_anal3m.append(SFUV4pi_resm*expn(2, taum))
        # if taum > 0.5*tau_SF:
        #     JFUV_anal3m.append(SFUV4pi_resm*J_over_JUV_outside_slab(taum, tau_SF))
        # else:
        #     JFUV_anal3m.append(SFUV4pi_resm*J_over_JUV_inside_slab(taum, tau_SF))

    G0pr_anal3p = np.array(JFUV_anal3p)/2.1e-4
    G0pr_anal3m = np.array(JFUV_anal3m)/2.1e-4

    #print zpc[zpc > zmax], G0pr_anal3p
    plt.plot(zpc, G0pr_anal, c='k', lw=3, ls='--', alpha=0.8,
             label='analytic')
    plt.plot(zpc, G0pr_anal2, c='m', lw=1.5, ls='--', alpha=1.0,
             label='analytic2')
    plt.plot(zpc[zpc > zmax], G0pr_anal3p, c='c', lw=1.5, ls='--', alpha=1.0)
    plt.plot(zmc[zmc < zmin], G0pr_anal3m, c='c', lw=1.5, ls='--', alpha=1.0)

    plt.yscale('log')
    plt.ylim(1e-2, 2e1)
    plt.xlim(-1024,1024)
    l = plt.legend(loc="upper left", fontsize=14)

    for tick in plt.gca().axes.get_xticklabels():
        tick.set_visible(False)
    
    plt.sca(axes[1])
    plt.plot(da.mu, da[:,num]/df.L_tot0[num], alpha=0.4)
    plt.xlim(-1,1)
    plt.ylim(0.0,0.005)
    plt.xlabel(r'$\mu = \cos \theta$')
    plt.ylabel(r'$d L_{\rm lost}/d\mu$')
    
    plt.sca(axes[2])
    for zstar_, L_FUV_ in zip(zstar, spa['L_FUV']):
        plt.plot([zstar_, zstar_], [0, L_FUV_/1e6], c='r', lw=0.5, alpha=0.6)
    plt.ylim(0, 10e0)
    plt.xlabel('z [pc]')
    plt.ylabel(r'$L_{*}\,[10^6 L_{\odot}]$')
    plt.xlim(-892,892)

    plt.sca(axes[3])
    plt.plot(df.time, df.L_lost0/df.L_tot0, c='b')
    plt.plot([df.time[num]], [df.L_lost0[num]/df.L_tot0[num]], c='r', marker='o', ms=5.0)
    plt.xlabel('time [Myr]')
    plt.ylabel(r'$L_{\rm lost}/L_{\rm tot}$')
    
    subtit = 'model: {0:s}, $t$={1:5.1f} [Myr], '.format(problem_id, sp['time'])
    plt.suptitle(subtit + ' ' + r'$d_{{\rm max}}={0:g}$ pc'.format(dmax),
                 fontsize=16)
    plt.subplots_adjust(top=0.90,wspace=0.35)
    
    savname = os.path.join(os.path.expanduser('~'),
                           'Dropbox/tigrad/plot/comparison-rt-plane-parallel/')
    savname = os.path.join(savname, '{0:s}_{1:04d}.png'.format(\
                    problem_id, num))
    plt.savefig(savname, dpi=200)
