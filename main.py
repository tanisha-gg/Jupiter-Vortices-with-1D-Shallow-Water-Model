#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from parameters import Parameters
from solver import Solver
from visualization import Visualization
from io import IO

def main():
    # Initialize parameters
    params = Parameters()

    # Initialize solver
    solver = Solver(params)

    # Initialize visualization
    viz = Visualization(params)

    # Initialize IO
    io = IO(params)

    # Run simulation
    solver.run_simulation(viz, io)

if __name__ == "__main__":
    main()
