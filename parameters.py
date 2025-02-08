import numpy as np
from dataclasses import dataclass

@dataclass
class Parameters:
    # Grid Parameters
    Lx: float = 200e3
    Ly: float = 200e3
    Nx: int = 128
    Ny: int = 128
    dx: float = Lx / Nx
    dy: float = Ly / Ny

    # Physical parameters
    f0: float = 2e-4
    gp: float = 23.1516
    H0: float = 500

    # Temporal Parameters
    mins: float = 60.
    hours: float = 3600.
    days: float = 0.414 * 24 * 3600.
    t0: float = 0.0
    tf: float = 0.4 * days
    dt: float = 3
    tp: float = 0.1 * hours
    Nt: int = int(tf / dt)
    npt: int = int(tp / dt)
    tt: np.ndarray = np.arange(0, tf + dt, dt)
    ttpl: np.ndarray = np.arange(0, tf + tp, tp)

    # Define Grid (staggered grid)
    x: np.ndarray = np.linspace(-Lx / 2 + dx / 2, Lx / 2 - dx / 2, Nx)
    y: np.ndarray = np.linspace(-Ly / 2 + dy / 2, Ly / 2 - dy / 2, Ny)
    xx, yy = np.meshgrid(x, y)

    # Define wavenumber (frequency)
    kx = 2 * np.pi / Lx * Nx * np.fft.fftfreq(Nx)
    ky = 2 * np.pi / Ly * Ny * np.fft.fftfreq(2 * Ny)
    kxx, kyy = np.meshgrid(kx, ky)

    # Filter Parameters
    kmax = max(max(kx), max(ky))
    ks = 0.8 * kmax
    km = 0.9 * kmax
    alpha = 0.69 * ks ** (-1.88 / np.log(km / ks))
    beta = 1.88 / np.log(km / ks)
    sfilt = np.exp(-alpha * (kxx ** 2 + kyy ** 2) ** (beta / 2.0))

    # Modify class
    ikx, iky = 1j * kxx, 1j * kyy
