################################################################################
##             PROGRAM FOR SIMULATING A SYSTEM OF 2 INTERACTING SPINS         ##
################################################################################
## AUTHOR: Diego Alejandro Herrera Rojas
## DATE: 06/04/21
## DESCRIPTION: In this program, I test the routines implemented in the class
##              QSTsimulator by considering the example of a 2 spin system.

################################################################################
##              IMPORTS NECESSARY TO PERFORM QUANTUM ALGORITHMS               ##
################################################################################
import matplotlib.pyplot as plt
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
                                ExchangeIntegrals=[1.0,1.0,1.0],\
                                ExternalField=[1.0,1.0,1.0],\
                                local_simul=True)
    ## Diagonalize Hamiltonian
    DemoSimulator.DiagHamilt()
    ## Capture data of simulation with Qiskit
    TOTSTEPS = 10
    fidelities = [DemoSimulator.ExpFidelity(STEPS=numsteps,ts=0.875) \
                    for numsteps in range(1,TOTSTEPS+1)]

################################################################################
##                          HERE I PLOT THE FIDELITIES                        ##
################################################################################
    plt.xlabel(r'Number of iterations')
    plt.ylabel(r'$\langle \psi_{exc} | \psi_{sim} \rangle$')
    plt.scatter([i for i in range(1,TOTSTEPS+1)],fidelities)
    plt.savefig('../images/'+DemoSimulator.backend_name+\
                'Fidelity'+\
                str(DemoSimulator.num_spins)+\
                'Steps'+str(TOTSTEPS)+'.pdf')
