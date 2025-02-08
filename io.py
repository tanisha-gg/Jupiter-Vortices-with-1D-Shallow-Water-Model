import netCDF4 as nc4
from datetime import datetime

class IO:
    def __init__(self, params):
        self.params = params

    def create_netcdf_file(self, uvh, vort):
        today = datetime.today()
        filenc = 'data_sw_Nx%d_Ny%d' % (self.params.Nx, self.params.Ny) + '.nc'
        self.rootgrp = nc4.Dataset(filenc, 'w', format='NETCDF4')

        # Create group
        solngrp = self.rootgrp.createGroup('soln')

        # Specify dimensions
        Nt = self.rootgrp.createDimension('time', len(self.params.tt))
        timeplot = self.rootgrp.createDimension('timeplot', len(self.params.ttpl))
        Nx = self.rootgrp.createDimension('x-dim', len(self.params.x))
        Ny = self.rootgrp.createDimension('y-dim', len(self.params.y))

        # Build variables
        time = self.rootgrp.createVariable('Time', 'f4', 'time')
        timeplot = self.rootgrp.createVariable('TimePlot', 'f4', 'timeplot')
        xvar = self.rootgrp.createVariable('x', 'f4', 'x-dim')
        yvar = self.rootgrp.createVariable('y', 'f4', 'y-dim')
        u = self.rootgrp.createVariable('x-velocity', 'd', ('timeplot', 'x-dim', 'y-dim'))
        v = self.rootgrp.createVariable('y-velocity', 'd', ('timeplot', 'x-dim', 'y-dim'))
        h = self.rootgrp.createVariable('depth', 'd', ('timeplot', 'x-dim', 'y-dim'))
        vort = self.rootgrp.createVariable('vort', 'd', ('timeplot', 'x-dim', 'y-dim'))

        uB = self.rootgrp.createVariable('x-velocityB', 'd', ('x-dim', 'y-dim'))
        vB = self.rootgrp.createVariable('y-velocityB', 'd', ('x-dim', 'y-dim'))
        hB = self.rootgrp.createVariable('depthB', 'd', ('x-dim', 'y-dim'))
        vortB = self.rootgrp.createVariable('vortB', 'd', ('x-dim', 'y-dim'))

        time.units = 's'
        timeplot.units = 's'
        xvar.units = 'm'
        yvar.units = 'm'
        u.units = 'm/s'
        v.units = 'm/s'
        h.units = 'm'
        vort.units = '1/s'

        time[:] = self.params.tt
        timeplot[:] = self.params.ttpl
        xvar[:] = self.params.x
        yvar[:] = self.params.y

        # Write ICs to a file and background state
        self.cnt = 0
        u[self.cnt, :, :], v[self.cnt, :, :], h[self.cnt, :, :], vort[self.cnt, :, :] = uvh[0, :, :], uvh[1, :, :], uvh[2, :, :], vort
        uB[:, :], vB[:, :], hB[:, :], vortB[:, :] = self.params.uB, self.params.vB, self.params.hB, self.params.vortB

    def save_to_netcdf(self, uvh, vort, ii):
        self.cnt += 1
        self.rootgrp['x-velocity'][self.cnt, :, :] = uvh[0, :, :]
        self.rootgrp['y-velocity'][self.cnt, :, :] = uvh[1, :, :]
        self.rootgrp['depth'][self.cnt, :, :] = uvh[2, :, :]
        self.rootgrp['vort'][self.cnt, :, :] = vort
