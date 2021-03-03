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
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
from QuantumSTsimulator import QSTsimulator
plt.style.use('FigureStyle.mplstyle')

if __name__ == '__main__':
    ## Instantiate a simulator class
    DemoSimulator = QSTsimulator(num_spins=2,\
                                ExchangeIntegrals=[1.0,1.0,1.0],\
                                ExternalField=[1.0,1.0,1.0])
    ## Capture data of simulation with Qiskit
    PDF = DemoSimulator.EvolAlgorithm(NUMSTEPS=400,t=5)
