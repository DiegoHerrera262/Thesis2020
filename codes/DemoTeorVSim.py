################################################################################
##             PROGRAM FOR SIMULATING A SYSTEM OF 2 INTERACTING SPINS         ##
################################################################################
## AUTHOR: Diego Alejandro Herrera Rojas
## DATE: 06/08/21
## DESCRIPTION: In this program, I test the routines implemented in the class
##              QSTsimulator by considering the example of a 2 spin system.

################################################################################
##              IMPORTS NECESSARY TO PERFORM QUANTUM ALGORITHMS               ##
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
plt.style.use('FigureStyle.mplstyle')

################################################################################
##                      I HIGHLIGHT THE SIMULATION MODULE                     ##
################################################################################
from QuantumSTsimulator import QSTsimulator

################################################################################
##              HERE I PERFORM TIME SIMULATION WITH MY ALGORITHM              ##
################################################################################
if __name__ == '__main__':
    ## Instantiate a simulator class
    DemoSimulator = QSTsimulator(num_spins=2,\
                                ExchangeIntegrals=[1,1,1],\
                                ExternalField=[1,1,1],\
                                local_simul=True)
    ## Diagonalize Hamiltonian
    DemoSimulator.DiagHamilt()
    ## Define time discretization
    tx = np.linspace(0,1.75,200)
    ## Produce exact curves for time evolution
    initstate = np.zeros(2**DemoSimulator.num_spins)
    initstate[0] = 1
    PDFex = np.array([\
        DemoSimulator.ExactTimeEvol(initstate,t=ts) \
        for ts in tx\
    ])
    ## Produce simulated curves for time evolution
    TOTSTEPS = int(input('Enter number of ST steps: '))
    tsim = np.linspace(0,1.75,TOTSTEPS)
    PDFsim = np.array([\
        DemoSimulator.SimulTimeEvol(STEPS=idx,t=tsim[idx]) \
        for idx in range(len(tsim))
    ])

################################################################################
##                      HERE I COMPARE THE DISTRIBUTIONS                      ##
################################################################################
    plt.xlabel(r'$t$ (u. a.)')
    plt.ylabel(r'$\langle \psi | q_n \rangle$')
    colors = cm.rainbow(np.linspace(5/8,1,2**DemoSimulator.num_spins))
    for num in range(2**DemoSimulator.num_spins):
        ## Scatter plot for simulated data
        plt.scatter(tsim,PDFsim[:,num],color=colors[num])
        ## Continous line plot for exact data
        plt.plot(tx,PDFex[:,num],color=colors[num])
    ## Show plot
    plt.show()
