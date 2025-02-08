## This repository contains a Python implementation of a 1-Layer Rotating Shallow Water (RSW) model to simulate vortices, with a focus on applications such as modeling Jupiter's atmosphere. The code uses a pseudospectral method for spatial discretization and Adams-Bashforth for time stepping.

#### I've also included two reports I've written about vortices on Jupiter.

# Features
Numerical Solver: Solves the 1-Layer RSW equations using a pseudospectral method.

Visualization: Includes tools for visualizing velocity fields, vorticity, and fluid depth.

NetCDF Output: Saves simulation results in NetCDF format for post-processing and analysis.

Modular Design: The code is organized into classes and modules for easy extension and maintenance.

# Installation
Prerequisites
Python 3.x

Required Python packages: numpy, matplotlib, netCDF4, pyfftw (optional but recommended for faster FFTs)

# Steps
Clone the repository:

bash
Copy
git clone https://github.com/your-username/Simulating-Vortices-on-Jupiter.git
cd Simulating-Vortices-on-Jupiter.git
Install the required Python packages:

bash
Copy
pip install numpy matplotlib netCDF4 pyfftw
Run the simulation:

bash
Copy
python main.py
Usage
Running the Simulation
The main script main.py initializes the simulation parameters, runs the solver, and visualizes the results. You can modify the parameters in parameters.py to customize the simulation.

# Output
NetCDF Files: Simulation results are saved in a NetCDF file (data_sw_Nx128_Ny128.nc by default).

Plots: Velocity quiver plots and other visualizations are displayed during the simulation.

# Customization
Grid Parameters: Adjust Lx, Ly, Nx, and Ny in parameters.py to change the domain size and resolution.

Physical Parameters: Modify f0, gp, and H0 to change the Coriolis parameter, gravitational acceleration, and mean fluid depth.

Initial Conditions: Customize the initial velocity and depth fields in solver.py.

# Code Structure
main.py: Main script to run the simulation.

parameters.py: Defines simulation parameters and grid setup.

solver.py: Implements the numerical solver and time-stepping methods.

visualization.py: Contains functions for plotting and visualization.

io.py: Handles input/output operations, including NetCDF file handling.
