################################################################################
##             PROGRAM FOR SIMULATING A SYSTEM OF 2 INTERACTING SPINS         ##
################################################################################
## AUTHOR: Diego Alejandro Herrera Rojas
## DATE: 02/03/21
## DESCRIPTION: In this program, I test the routines implemented in the class
##              QSTsimulator by considering the example of a 2 spin system.

################################################################################
##              IMPORTS NECESSARY TO PERFORM QUANTUM ALGORITHMS               ##
################################################################################
import matplotlib.pyplot as plt
import pandas as pd

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
                                ExchangeIntegrals=[2,3,5],\
                                ExternalField=[1,3,2],\
                                local_simul=False)
    ## Capture data of simulation with Qiskit
    for n in range(1,16):
        PDF = DemoSimulator.EvolAlgorithm(NUMSTEPS=n,t=1.8)
        plt.clf()
