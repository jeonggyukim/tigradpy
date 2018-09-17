"""
Read athena hst, zprof, starpar vtk files using pandas, dictionary, xarray
"""

import os
import re
import glob
import struct

import numpy as np
import pandas as pd
import xarray as xr

def read_hst(filename, force_override=False, verbose=False):
    """
    Function to read athena history file and pickle
    
    Parameters:
       filename : string
           Name of the file to open, including extension
       force_override: bool
           Flag to force read of hst file even when pickle exists

    Returns:
       hst : pandas dataframe
    """

    skiprows = 3

    fpkl = filename + '.p'
    if not force_override and os.path.exists(fpkl) and \
       os.path.getmtime(fpkl) > os.path.getmtime(filename):
        hst = pd.read_pickle(fpkl)
        if verbose:
            print('[read_hst]: reading from existing pickle.')
    else:
        if verbose:
            print('[read_hst]: pickle does not exist or hst file updated.' + \
                      ' Reading {0:s}'.format(filename))
        vlist = get_hst_var(filename)

        # c engine does not support regex separators
        hst = pd.read_table(filename, names=vlist, skiprows=skiprows,
                            comment='#', sep='\s*', engine='python')
        hst.to_pickle(fpkl)

    return hst
      
def get_hst_var(filename):
    """
    Read variable names from history file

    Parameters:
       filename : string
           Name of the file to open, including extension

    Returns:
       vlist : list
           list of variables

    """

    with open(filename, 'r') as f:
        # For the moment, skip the first line which contains information about
        # the volume of the simulation domain
        # "Athena history dump for level=.. domain=0 volume=..."
        # "#   [1]=time      [2]=dt         [3]=mass ......"
        h = f.readline()
        h = f.readline()

    vlist=re.split("\[\d+]\=|\n", h)
    for i in range(len(vlist)):
        vlist[i] = re.sub("\s|\W", "", vlist[i])

    return vlist[1:-1]

def read_zprof_all(dirname, problem_id, phase='whole', force_override=False):
    """
    Read all zprof files and make a DataArray object and write to a NetCDF
    file

    Parameters:
       dirname : string
           Name of the directory where zprof files are located
       problem_id: string
           Prefix of zprof files
       phase: string
           whole, phase1, ..., phase5 (cold, intermediate, warm, hot1, hot2)
  
    Returns:
       da: xarray dataarray
    """

    ### Find all files with "/dirname/problem_id.xxxx.phase.zprof"    
    fname_base = '{0:s}.????.{1:s}.zprof'.format(problem_id, phase)
    fnames = glob.glob(os.path.join(dirname, fname_base))

    fnetcdf = '{0:s}.{1:s}.zprof.nc'.format(problem_id, phase)
    fnetcdf = os.path.join(dirname, fnetcdf)

    # check if netcdf file exists and compare last modified times
    mtime_max = np.array([os.path.getmtime(fname) for fname in fnames]).max()
    if not force_override and os.path.exists(fnetcdf) and \
        os.path.getmtime(fnetcdf) > mtime_max:
        da = xr.open_dataarray(fnetcdf)
        return da
    
    # if here, need to create a new dataarray
    time = []
    df_all = []
    for fname in fnames:
        # Read time
        with open(fname, 'r') as f:
            h = f.readline()
            time.append(float(h[h.rfind('t=') + 2:]))

        # read pickle if exists
        df = read_zprof(fname, force_override=False)
        df_all.append(df)

    z = (np.array(df['z'])).astype(float)
    fields = np.array(df.columns)

    # Combine all data
    df_all = np.stack(df_all, axis=0)
    # print df_all.shape
    da = xr.DataArray(df_all.T,
                      coords=dict(fields=fields, z=z, time=time),
                      dims=('fields', 'z', 'time'))

    # Somehow overwriting using mode='w' doesn't work..
    if os.path.exists(fnetcdf):
        os.remove(fnetcdf)
        
    da.to_netcdf(fnetcdf, mode='w')

    return da

def read_zprof(filename, force_override=False, verbose=False):
    """
    Function to read one zprof file and pickle
    
    Parameters:
       filename : string
           Name of the file to open, including extension
       force_override: bool
           Flag to force read of zprof file even when pickle exists

    Returns:
       zp : pandas dataframe
    """

    skiprows = 2

    fpkl = filename + '.p'
    if not force_override and os.path.exists(fpkl) and \
       os.path.getmtime(fpkl) > os.path.getmtime(filename):
        zp = pd.read_pickle(fpkl)
        if verbose:
            print('[read_zprof]: reading from existing pickle.')
    else:
        if verbose:
            print('[read_zprof]: pickle does not exist or zprof file updated.' + \
                      ' Reading {0:s}'.format(filename))

        with open(filename, 'r') as f:
            # For the moment, skip the first line which contains information about
            # the time at which the file is written
            # "# Athena vertical profile at t=xxx.xx"
            h = f.readline()
            time = float(h[h.rfind('t=') + 2:])
            h = f.readline()
            vlist = h.split(',')
            if vlist[-1].endswith('\n'):
                vlist[-1] = vlist[-1][:-1]    # strip \n

        # c engine does not support regex separators
        zp = pd.read_table(filename, names=vlist, skiprows=skiprows,
                           comment='#', sep=',', engine='python')
        zp.to_pickle(fpkl)

    return zp


def read_starpar_vtk(filename):
    """
    Read athena starpar vtk output.
    Returns a dictionary containing mass, position, velocity, age, etc.

    Parameters:
       filename : string
           Name of the file to open, including extension
    """
    
    def _parse_starpar_vtk_line(spl, grid):
        if "vtk" in spl:
            grid['vtk_version'] = spl[-1]
        elif "time=" in spl:
            time_index = spl.index("time=")
            grid['time'] = float(spl[time_index+1].rstrip(','))
        elif "POINTS" in spl:
            grid['nstars'] = int(spl[1])
        elif "SCALARS" in spl:
            field = spl[1]
            grid['read_field'] = field
            grid['read_type'] = 'scalar'
            grid['data_type'] = spl[-1]
        elif "VECTORS" in spl:
            field = spl[1]
            grid['read_field'] = field
            grid['read_type'] = 'vector'
            grid['data_type'] = spl[-1]

    def _convert_field_name(name):
        if name == 'star_particle_id':
            return 'id'
        elif name == 'star_particle_mass':
            return 'mass'
        elif name == 'star_particle_age':
            return 'age'
        elif name == 'star_particle_mage':
            return 'mage'
        elif name == 'star_particle_position':
            return 'x'
        elif name == 'star_particle_velocity':
            return 'v'
        elif name == 'star_particle_flag':
            return 'flag'
        elif name == 'star_particle_metal_mass[0]':
            return 'metal_mass[0]'
        elif name == 'star_particle_metal_mass[1]':
            return 'metal_mass[1]'
        elif name == 'star_particle_metal_mass[2]':
            return 'metal_mass[2]'

    # Check for existance of file
    if not os.path.isfile(filename):
        raise IOError('starpar vtk file {0:s} is not found'.format(filename))

    star = {}
    with open(filename, 'rb') as f:
        grid = {}
        line = f.readline()

        # Read time, nstars
        while line != '':
            spl = line.strip().split()
            _parse_starpar_vtk_line(spl, grid)
            line = f.readline()
            if "POINT_DATA" in spl:
                break

        time = grid['time']
        nstars = grid['nstars']

        # Read field info
        _field_map = {}
        while line != '':
            spl = line.strip().split()
            _parse_starpar_vtk_line(spl, grid)
            if "SCALARS" in spl:
                field = grid['read_field']
                datatype = grid['data_type']
                line = f.readline()  # Read the lookup table line
                _field_map[field] = ('scalar', f.tell(), datatype)
            elif "VECTORS" in spl:
                field = grid['read_field']
                datatype = grid['data_type']
                _field_map[field] = ('vector', f.tell(), datatype)
            line = f.readline()

        # Read all fields
        for k, v in _field_map.iteritems():
            if v[0] == 'scalar':
                nvar = 1
                shape = [nstars, 1]
            elif v[0]=='vector':
                nvar = 3
                shape = [nstars, 3]
            else:
                raise ValueError('Unknown variable type')

            if v[2] == 'float':
                fmt = '>{}f'.format(nvar*nstars)
            elif v[2] == 'int':
                fmt = '>{}i'.format(nvar*nstars)

            f.seek(v[1])
            size = struct.calcsize(fmt)
            data = np.array(struct.unpack(fmt, f.read(size)))
            name = _convert_field_name(k)
            star[name] = np.transpose(np.reshape(data, shape), (0, 1))
            if nstars > 1:
                star[name] = np.squeeze(star[name])
            elif nstars == 1:
                if v[0] != 'vector':
                    star[name] = star[name][0]

    # Sort id in an ascending order (or age in an descending order)
    if nstars > 1:
        idsrt = star['id'].argsort()
        for k, v in star.iteritems():
            star[k] = v[idsrt]

    # Add time, nstars keys at the end
    star['time'] = time
    star['nstars'] = nstars
    
    return star
