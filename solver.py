import numpy as np
from parameters import Parameters

class Solver:
    def __init__(self, params):
        self.params = params

    def flux_sw_spec(self, uvh):
        # Define parameters
        Ny, Nx = uvh[0, :, :].shape
        ikx, iky = self.params.ikx, self.params.iky

        # Compute algebraic fields
        u, v, h = uvh[0, :, :], uvh[1, :, :], self.params.H0 + uvh[2, :, :]
        U, V, B = h * u, h * v, self.params.gp * h + 0.5 * (u ** 2 + v ** 2)

        # single derivatives (1D transforms)
        vx = (np.fft.ifft(ikx[0:Ny, :] * np.fft.fft(v, axis=1), axis=1)).real
        Ux = (np.fft.ifft(ikx[0:Ny, :] * np.fft.fft(U, axis=1), axis=1)).real
        uy = (np.fft.ifft(iky * np.fft.fft(np.vstack((u, np.flipud(u))), axis=0), axis=0))[0:Ny, :].real
        Vy = (np.fft.ifft(iky * np.fft.fft(np.vstack((V, -np.flipud(V))), axis=0), axis=0))[0:Ny, :].real
        vort = vx - uy
        q = (vort + self.params.f0) / h

        # gradient of B (2D transform)
        Bk = np.fft.fftn(np.vstack((B, np.flipud(B))))
        Bx = (np.fft.ifftn(ikx * Bk))[0:Ny, :].real
        By = (np.fft.ifftn(iky * Bk))[0:Ny, :].real

        # fluxes
        flux = np.zeros((3, Ny, Nx))
        flux[0, :, :] = q * V - Bx
        flux[1, :, :] = -q * U - By
        flux[2, :, :] = -Ux - Vy

        # energy and enstrophy
        energy, enstrophy = 0.5 * np.mean(self.params.gp * h ** 2 + h * (u ** 2 + v ** 2)), 0.5 * np.mean(q ** 2 * h)

        return flux, vort, energy, enstrophy

    def run_simulation(self, viz, io):
        # Initial Conditions
        uvh = np.zeros((3, self.params.Ny, self.params.Nx))
        energy, enstr = np.zeros(self.params.Nt), np.zeros(self.params.Nt)

        uvh[0, :, :] = self.params.uB
        uvh[1, :, :] = self.params.vB
        uvh[2, :, :] = self.params.hB + 1e-3 * np.random.randn(self.params.Nx, self.params.Ny)

        # Compute vorticity numerically
        u, v = uvh[0, :, :], uvh[1, :, :]
        vx = (np.fft.ifft(self.params.ikx[0:self.params.Ny, :] * np.fft.fft(v, axis=1), axis=1)).real
        uy = (np.fft.ifft(self.params.iky * np.fft.fft(np.vstack((u, np.flipud(u))), axis=0), axis=0))[0:self.params.Ny, :].real
        vort = vx - uy

        # Create netCDF file
        io.create_netcdf_file(uvh, vort)

        # Begin Plotting
        t = self.params.t0
        viz.plot_quiver(self.params.xx[0::3], self.params.yy[0::3], u[0::3], v[0::3], t, 'Velocity Quiver Plot')

        # Euler step
        NLnm, vort, energy[0], enstr[0] = self.flux_sw_spec(uvh)
        uvh = uvh + self.params.dt * NLnm

        # AB2 step
        NLn, vort, energy[1], enstr[1] = self.flux_sw_spec(uvh)
        uvh = uvh + 0.5 * self.params.dt * (3 * NLn - NLnm)

        for ii in range(3, self.params.Nt + 1):
            # AB3 step
            t = ii * self.params.dt / self.params.days
            NL, vort, energy[ii - 1], enstr[ii - 1] = self.flux_sw_spec(uvh)
            uvh = uvh + self.params.dt / 12 * (23 * NL - 16 * NLn + 5 * NLnm)

            # Filter
            uvh[0, :, :] = (np.fft.ifftn(self.params.sfilt * np.fft.fftn(np.vstack((uvh[0, :, :], np.flipud(uvh[0, :, :]))))))[0:self.params.Ny, :].real
            uvh[1, :, :] = (np.fft.ifftn(self.params.sfilt * np.fft.fftn(np.vstack((uvh[1, :, :], -np.flipud(uvh[1, :, :]))))))[0:self.params.Ny, :].real
            uvh[2, :, :] = (np.fft.ifftn(self.params.sfilt * np.fft.fftn(np.vstack((uvh[2, :, :], np.flipud(uvh[2, :, :]))))))[0:self.params.Ny, :].real

            # Reset fluxes
            NLnm, NLn = NLn, NL

            if (ii - 0) % self.params.npt == 0:
                # Save data to a file
                io.save_to_netcdf(uvh, vort, ii)

                # Update solution
                viz.plot_quiver(self.params.xx[0::3], self.params.yy[0::3], uvh[0, :, :][0::3], uvh[1, :, :][0::3], t, 'Velocity Quiver Plot')
