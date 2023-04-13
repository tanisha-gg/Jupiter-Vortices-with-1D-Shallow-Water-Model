#!/usr/bin/env python
#  SW_channel.py
#
# Solve the 1-Layer Rotating Shallow Water (SW) Model
#
# Fields: 
#   u : zonal velocity
#   v : meridional velocity
#   h : fluid depth
#
# Evolution Eqns:
#   B = g*h + 0.5*(u**2 + v**2)     Bernoulli function
#   q = (v_x - u_y + f)/h           Potential Vorticity
#   [U,V] = h[u,v]                  Transport velocities
#
#	u_t =  (q*V) - B_x
#	v_t = -(q*U) - B_y
#	h_t = - div[U,V]
#
# Geometry: periodic in x and channel in y
#
# Numerical Method:
# 1) Pseudospectral (Fourier) method
# 2) Adams-Bashforth for time stepping
#

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
import sys
from datetime import datetime

today = datetime.today()

try:
    import pyfftw
    from numpy import zeros as nzeros

    # Keep fft objects in cache for efficiency
    nthreads = 1
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1e8)
    def empty(N, dtype="float", bytes=16):
        return pyfftw.n_byte_align_empty(N, bytes, dtype=dtype)

    def zeros(N, dtype="float", bytes=16):
        return pyfftw.n_byte_align(nzeros(N, dtype=dtype), bytes)
    
    # Monkey patches for fft
    ifft  = pyfftw.interfaces.numpy_fft.ifft
    fft   = pyfftw.interfaces.numpy_fft.fft
    ifftn = pyfftw.interfaces.numpy_fft.ifftn
    fftn  = pyfftw.interfaces.numpy_fft.fftn
    fftfreq  = pyfftw.interfaces.numpy_fft.fftfreq

except:
    print(" ")
    print(Warning("Install pyfftw, it is much faster than numpy fft"))
    print(" ")
    from numpy.fft import fft, ifft, fftn, ifftn, fftfreq

class wavenum:
    pass

#######################################################
#        Create netCDF info                           #
#######################################################

def build_netcdf(rootgrp, tt, ttpl, x, y):

    #FJP: update
    # Open netcdf4 file
    # rootgrp
    #        -> solngrp
    #                     -> ugrp : zonal velocity
    #                     -> vgrp : meridional velocity
    #                     -> hgrp : depth
   
    # move into function
    rootgrp.description = "Data for one-layer RSW model"
    rootgrp.history = "Created " + today.strftime("%d/%m/%y")

    # Create group
    solngrp    = rootgrp.createGroup('soln')

    # Specify dimensions
    Nt       = rootgrp.createDimension('time',      len(tt))
    timeplot = rootgrp.createDimension('timeplot',  len(ttpl))
    Nx       = rootgrp.createDimension('x-dim',     len(x)) 
    Ny       = rootgrp.createDimension('y-dim',     len(y))

    # Build variables
    time     = rootgrp.createVariable('Time', 'f4','time')
    timeplot = rootgrp.createVariable('TimePlot', 'f4','timeplot')
    xvar     = rootgrp.createVariable('x', 'f4','x-dim')
    yvar     = rootgrp.createVariable('y', 'f4','y-dim')
    u        = rootgrp.createVariable('x-velocity', 'd', ('timeplot','x-dim','y-dim') )
    v        = rootgrp.createVariable('y-velocity', 'd', ('timeplot','x-dim','y-dim') )
    h        = rootgrp.createVariable('depth',      'd', ('timeplot','x-dim','y-dim') )
    vort     = rootgrp.createVariable('vort',       'd', ('timeplot','x-dim','y-dim') )

    uB       = rootgrp.createVariable('x-velocityB', 'd', ('x-dim','y-dim') )
    vB       = rootgrp.createVariable('y-velocityB', 'd', ('x-dim','y-dim') )
    hB       = rootgrp.createVariable('depthB',      'd', ('x-dim','y-dim') )
    vortB    = rootgrp.createVariable('vortB',       'd', ('x-dim','y-dim') )

    time.units     = 's'
    timeplot.units = 's'
    xvar.units     = 'm'
    yvar.units     = 'm'
    u.units        = 'm/s'
    v.units        = 'm/s'
    h.units        = 'm'
    vort.units     = '1/s'

    #FJP: is this stuff necessary?
    time[:]     = tt
    timeplot[:] = ttpl
    xvar[:]     = x
    yvar[:]     = y
    
    return u, v, h, vort, uB, vB, hB, vortB

def flux_sw_spec(uvh, parms):

    # Define parameters
    Ny, Nx   = uvh[0,:,:].shape
    ikx, iky = parms.ikx, parms.iky
    
    # Compute algebraic fields
    u, v, h  = uvh[0,:,:], uvh[1,:,:], parms.H0 + uvh[2,:,:]
    U, V, B  = h*u, h*v, parms.gp*h + 0.5*(u**2 + v**2)

    # single derivatives (1D transforms)
    vx   =  (ifft(ikx[0:Ny,:]*fft(v, axis=1), axis=1)).real
    Ux   =  (ifft(ikx[0:Ny,:]*fft(U, axis=1), axis=1)).real
    uy   =  (ifft(iky*fft(np.vstack((u, np.flipud(u))), axis=0), axis=0))[0:Ny,:].real
    Vy   =  (ifft(iky*fft(np.vstack((V,-np.flipud(V))), axis=0), axis=0))[0:Ny,:].real
    vort = vx - uy
    q    =  (vort + parms.f0)/h

    # gradient of B (2D transform)
    Bk =  fftn(np.vstack((B, np.flipud(B))))
    Bx = (ifftn( ikx*Bk))[0:Ny,:].real
    By = (ifftn( iky*Bk))[0:Ny,:].real
            
    # fluxes
    flux = np.zeros((3, Ny, Nx))
    flux[0,:,:] =  q*V - Bx
    flux[1,:,:] = -q*U - By
    flux[2,:,:] = - Ux - Vy

    # energy and enstrophy
    energy, enstrophy = 0.5*np.mean(parms.gp*h**2+h*(u**2+v**2)),0.5*np.mean(q**2*h)

    return flux, vort, energy, enstrophy

def plot_quiver(xx, yy, u, v, t, parms, title):

    plt.figure(figsize=(12,8))
    plt.quiver(xx,yy,u, v)
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.title(title+" at t = %7.5f days" % t)
    #plt.xlim([-parms.Lx/2e3, parms.Lx/2e3])
    #plt.ylim([-parms.Ly/2e3, parms.Ly/2e3])
    plt.draw()
    plt.pause(0.001)
    
def plot_field(xx, yy, field, t, parms, title):

    plt.clf()
    plt.pcolormesh(xx/1e3,yy/1e3,field)
    plt.colorbar()
    plt.title(title+"at t = %7.5f days" % t)
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.xlim([-parms.Lx/2e3, parms.Lx/2e3])
    plt.ylim([-parms.Ly/2e3, parms.Ly/2e3])
    plt.draw()
    plt.pause(0.001)
    

    
def main():
    
    # Grid Parameters
    Lx,Ly  = 200e3, 200e3
    Nx,Ny  = 128, 128
    dx,dy  = Lx/Nx,Ly/Ny

    method = flux_sw_spec
    domain = 'channel'
    
    # Physical parameters
    f0, gp, H0 = 2e-4, 23.1516, 500

    # Temporal Parameters
    mins, hours, days = 60., 3600., 0.414*24*3600.
    t0, tf, dt = 0.0, 0.4*days, 3
    tp = 0.1*hours
    Nt, npt  = int(tf/dt), int(tp/dt)
    tt = np.arange(0, tf+dt, dt)
    ttpl = np.arange(0, tf+tp, tp)

    # Define Grid (staggered grid)
    x = np.linspace(-Lx/2+dx/2,Lx/2-dx/2,Nx)
    y = np.linspace(-Ly/2+dy/2,Ly/2-dy/2,Ny)
    xx,yy = np.meshgrid(x,y)

    #  Define wavenumber (frequency)
    kx = 2*np.pi/Lx*Nx*fftfreq(Nx)
    ky = 2*np.pi/Ly*Ny*fftfreq(2*Ny)
    kxx, kyy = np.meshgrid(kx,ky)

    # Filter Parameters
    kmax = max(max(kx),max(ky));
    ks = 0.8*kmax;
    km = 0.9*kmax; 
    alpha = 0.69*ks**(-1.88/np.log(km/ks));
    beta  = 1.88/np.log(km/ks);
    sfilt = np.exp(-alpha*(kxx**2 + kyy**2)**(beta/2.0));

    # Modify class
    parms = wavenum()
    parms.f0, parms.gp, parms.H0  = f0, gp, H0
    parms.ikx, parms.iky = 1j*kxx, 1j*kyy
    parms.Lx,  parms.Ly  = Lx, Ly
    parms.x,   parms.y   = x,  y
    
    # Geostrophic Vortex
    # u = -g/f*dh/dy
    # v =  g/f*dh/dx
    Lv1 = Lx/20
    Lv2 = Lx/10
    xshift1 = 15e3
    xshift2 = 30e3
    yshift = 30e3
    hB    =  3*np.exp(-((xx-xshift1)**2 + (yy+yshift)**2)/(Lv1)**2) - 3*np.exp(-((xx-xshift2)**2 + (yy)**2)/(Lv1)**2)  - 3*np.exp(-((xx-xshift1)**2 + (yy-yshift)**2)/(Lv1)**2) - 3*np.exp(-((xx+xshift1)**2 + (yy+yshift)**2)/(Lv1)**2) + 3*np.exp(-((xx+xshift2)**2 + (yy)**2)/(Lv1)**2)  + 3*np.exp(-((xx+xshift1)**2 + (yy-yshift)**2)/(Lv1)**2)
    
    hBx   =  (ifft(parms.ikx[0:Ny,:]*fft(hB, axis=1), axis=1)).real
    hBy   =  (ifft(parms.iky*fft(np.vstack((hB, np.flipud(hB))), axis=0), axis=0))[0:Ny,:].real

    uB    = -parms.gp/parms.f0*hBy  #2*parms.gp/parms.f0/(Lv)**2*yy*hB
    vB    = +parms.gp/parms.f0*hBx  #-2*parms.gp/parms.f0/(Lv)**2*xx*hB
    #FJP: this is not computed correctly.  Should be fixed if you want to use it
    vortB = 0

    # Initial Conditions
    uvh = np.zeros((3, Ny, Nx))
    energy, enstr = np.zeros(Nt), np.zeros(Nt)
    
    uvh[0,:,:] = uB
    uvh[1,:,:] = vB
    uvh[2,:,:] = hB + 1e-3*np.random.randn(Nx,Ny)

    # Compute vorticity numerically
    u,v  = uvh[0,:,:], uvh[1,:,:]
    vx   =  (ifft(parms.ikx[0:Ny,:]*fft(v, axis=1), axis=1)).real
    uy   =  (ifft(parms.iky*fft(np.vstack((u, np.flipud(u))), axis=0), axis=0))[0:Ny,:].real
    vort = vx - uy

    # Create netCDF file
    filenc   = 'data_sw_Nx%d_Ny%d' % (Nx, Ny) + '.nc'
    rootgrp = nc4.Dataset(filenc,'w', format='NETCDF4')
    gr_u, gr_v, gr_h, gr_vort, gr_uB, gr_vB, gr_hB, gr_vortB = build_netcdf(rootgrp, tt, ttpl, parms.x, parms.y)
 
    # Write ICs to a file and background state
    cnt = 0
    gr_u[cnt,:,:], gr_v[cnt,:,:], gr_h[cnt,:,:], gr_vort[cnt,:,:] = uvh[0,:,:], uvh[1,:,:], uvh[2,:,:], vort 
    gr_uB[:,: ], gr_vB[:,:], gr_hB[:,:], gr_vortB[:,:] = uB, vB, hB, vortB


    # Begin Plotting
    t = t0
    plt.figure(figsize=(12,8))
    #plot_field(xx, yy, vort, t, parms, "Vorticity ")
    #plot_field(xx,yy, np.sqrt(u**2+v**2), t, parms, 'Velocity ')
    #plot_field(xx, yy, uvh[2,:,:], t, parms, "Depth ")
    plot_quiver(xx[0::3], yy[0::3], u[0::3], v[0::3], t, parms, 'Velocity Quiver Plot')



    # Euler step 
    NLnm, vort, energy[0], enstr[0] = method(uvh, parms)
    uvh  = uvh + dt*NLnm;

    # AB2 step
    NLn, vort, energy[1], enstr[1] = method(uvh, parms)
    uvh  = uvh + 0.5*dt*(3*NLn - NLnm)


    for ii in range(3,Nt+1):

        # AB3 step
        t = ii*dt/days
        NL, vort, energy[ii-1], enstr[ii-1] = method(uvh, parms);
        uvh  = uvh + dt/12*(23*NL - 16*NLn + 5*NLnm)

        #print('At t = %6.4f days: Energy = %12.10f and Enstrophy = %12.10f' % (t, energy[ii-1], enstr[ii-1]))
        # Filter
        uvh[0,:,:]  = (ifftn(sfilt*fftn(np.vstack((uvh[0,:,:], np.flipud(uvh[0,:,:]))))))[0:Ny,:].real
        uvh[1,:,:]  = (ifftn(sfilt*fftn(np.vstack((uvh[1,:,:],-np.flipud(uvh[1,:,:]))))))[0:Ny,:].real
        uvh[2,:,:]  = (ifftn(sfilt*fftn(np.vstack((uvh[2,:,:], np.flipud(uvh[2,:,:]))))))[0:Ny,:].real
    
        # Reset fluxes
        NLnm, NLn = NLn, NL

        if (ii-0)%npt==0:
            # Save data to a file
            cnt += 1
            #FJP: combine groups to gr_uvh?  Faster to write?
            gr_u[cnt,:,:], gr_v[cnt,:,:], gr_h[cnt,:,:], gr_vort[cnt,:,:] = uvh[0,:,:], uvh[1,:,:], uvh[2,:,:], vort 
            
            # Update solution
            #plot_field(xx, yy, vort, t, parms, 'Vorticity ')
            #plot_field(xx,yy, np.sqrt(uvh[0,:,:]**2+uvh[1,:,:]**2), t, parms, 'Velocity ')
            #plot_field(xx, yy, uvh[2,:,:], t, parms, 'Depth ')
    
            plot_quiver(xx[0::3],yy[0::3],uvh[0,:,:][0::3], uvh[1,:,:][0::3], t, parms, 'Velocity Quiver Plot')
            
        


            
main()