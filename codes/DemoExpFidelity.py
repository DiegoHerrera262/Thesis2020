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
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, Aer
from qiskit import IBMQ
from qiskit.circuit import exceptions
from qiskit.tools.monitor import job_monitor
import numpy as np
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
                                ExchangeIntegrals=[2,3,5],\
                                ExternalField=[1,3,5],\
                                local_simul=False)
    ## Diagonalize Hamiltonian
    DemoSimulator.DiagHamilt()
    ## Parameters for fidelity evaluation
    TOTSTEPS = int(input('Enter total steps: '))
    ts       = 0.875
    shots    = 2048
    ## Create Job for execution
    Circuits = [DemoSimulator.PerformManySTsteps(\
                    STEPS=numsteps,dt=ts/numsteps) \
                    for numsteps in range(1,TOTSTEPS+1)]
    Job = execute(Circuits,DemoSimulator.backend,shots=shots)
    ## Monitor Job
    if DemoSimulator.local_simul == False:
        job_monitor(Job)
    ## Get counts
    simul_pdf = [Job.result().get_counts(circuit) for circuit in Circuits]
    ## Convert to array of data
    spdf = DemoSimulator.Counts2PDF(Job,Circuits)
    spdf = 1/shots * spdf
    ## Compute exact PDF
    initstate = np.zeros(2**DemoSimulator.num_spins)
    initstate[0] = 1
    ex_vec = DemoSimulator.ExactTimeEvol(initstate,t=ts)
    epdf = np.abs(ex_vec)**2
    ## Compute fidelities
    fidelities = np.array([
        sum(np.sqrt(spdf[i] * epdf))**2 for i in range(len(spdf))
    ])

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
