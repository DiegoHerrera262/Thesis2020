################################################################################
##             PROGRAM FOR SIMULATING A SYSTEM OF 2 INTERACTING SPINS         ##
################################################################################
# AUTHOR: Diego Alejandro Herrera Rojas
# DATE: 06/08/21
# DESCRIPTION: In this program, I test the routines implemented in the class
# QSTsimulator by comparing exact diagonalitawion with numpy,
# and ST integration with a quantum algorithm. This is intended
# for usage with qasm_simulator rather than IBM Q devices.

################################################################################
##              IMPORTS NECESSARY TO PERFORM QUANTUM ALGORITHMS               ##
################################################################################
from QuantumSTsimulator import QSTsimulator, InitSpinProdState
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
plt.style.use('FigureStyle.mplstyle')

################################################################################
##              HERE I PERFORM TIME SIMULATION WITH MY ALGORITHM              ##
################################################################################
if __name__ == '__main__':
    # Total number of spins
    tot_spins = 2
    #  Create random initial state
    angs = 2*np.pi * np.random.rand(tot_spins, 2)
    ## angs = np.zeros((tot_spins, 2))
    # Instantiate a simulator class
    DemoSimulator = QSTsimulator(num_spins=tot_spins,
                                 ExchangeIntegrals=[2, 3, 5],
                                 ExternalField=[1, 3, 2],
                                 local_simul=True,
                                 init_state=InitSpinProdState(
                                     angs=angs, num_spins=tot_spins))
    # Diagonalize Hamiltonian
    DemoSimulator.DiagHamilt()
    # Simulation time
    tsimul = 1.8
    # Define time discretization
    tx = np.linspace(0, tsimul, 200)
    # Produce exact curves for time evolution
    PDFex = np.array([
        DemoSimulator.ExactTimeEvol(t=ts)
        for ts in tx
    ])
    # Produce simulated curves for time evolution
    TOTSTEPS = int(input('Enter number of ST steps: '))
    tsim = np.linspace(0, tsimul, TOTSTEPS)
    PDFsim = np.array([
        DemoSimulator.SimulTimeEvol(shots=1 << 15, STEPS=idx, t=tsim[idx])
        for idx in range(len(tsim))
    ])

################################################################################
##                      HERE I COMPARE THE DISTRIBUTIONS                      ##
################################################################################
    plt.xlabel(r'$t$ (u. a.)')
    plt.ylabel(r'$\langle \psi | q_n \rangle$')
    colors = cm.rainbow(np.linspace(5/8, 1, 2**DemoSimulator.num_spins))
    for num in range(2**DemoSimulator.num_spins):
        # Scatter plot for simulated data
        plt.scatter(tsim, PDFsim[:, num], color=colors[num])
        # Continous line plot for exact data
        plt.plot(tx, PDFex[:, num], color=colors[num])
    plt.show()
    # save plot
    # plt.savefig('../images/'+DemoSimulator.backend_name +
    #            'ComparisonTeorVSim'+str(DemoSimulator.num_spins) +
    #            'spins'+'Steps'+str(TOTSTEPS)+'.pdf')
