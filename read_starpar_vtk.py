"""
Read athena hst, zprof, starpar vtk files using pandas, dictionary, xarray
"""

import os
import struct

import numpy as np

def read_starpar_vtk(filename):
    """
    Read athena starpar vtk output.
    Returns a dictionary containing mass, position, velocity, age, etc.

    Parameters
    ----------
       filename : string
           Name of the file to open, including extension
    
    Returns
    -------
       star: dict
           Dictionary containining star particle information
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
