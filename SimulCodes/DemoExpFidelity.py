################################################################################
##             PROGRAM FOR SIMULATING A SYSTEM OF 2 INTERACTING SPINS         ##
################################################################################
## AUTHOR: Diego Alejandro Herrera Rojas
## DATE: 06/04/21
## DESCRIPTION: This program is designed for computing experimental fidelity
##              of my simulation algorithm both locally and on IBM Q devices.
##              The measured fidelity is with respect to probability densities
##              of measurement in computational basis.
##
##              This is an upper bound on the actual fidelity.

################################################################################
##              IMPORTS NECESSARY TO PERFORM QUANTUM ALGORITHMS               ##
################################################################################
import qiskit
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
##             FUNCTION FOR CREATING ARBITRARY PRODUCT STATE                  ##
################################################################################
def InitSpinProdState(angs,num_spins=2):
    '''
    Function for creating an
    arbitrary product state of
    a spin chain
    '''
    ## Initialize quantum circuit
    qc_prod = QuantumCircuit(num_spins,num_spins)
    ## Append U gate to each spin
    for idx in range(num_spins):
        qc_prod.u(*angs[idx],idx)
    ## Compute statevector
    mybackend = Aer.get_backend('statevector_simulator')
    stvec = execute(qc_prod,mybackend).result().get_statevector()
    ## Return quantum circuit
    ## and the initial statevec
    return stvec, qc_prod


################################################################################
##              HERE I PERFORM TIME SIMULATION WITH MY ALGORITHM              ##
################################################################################
if __name__ == '__main__':
    ## Total number of spins
    tot_spins = 2
    ## Instantiate a simulator class
    DemoSimulator = QSTsimulator(num_spins=tot_spins,\
                                ExchangeIntegrals=[2,3,5],\
                                ExternalField=[1,3,5],\
                                local_simul=True)
    ## Diagonalize Hamiltonian
    DemoSimulator.DiagHamilt()
    ## Parameters for fidelity evaluation
    TOTSTEPS = int(input('Enter total steps: '))
    ts       = 1.8
    shots    = 1<<13
    ## Create random initial step
    ## angs = 2*np.pi * np.random.rand(tot_spins,3)
    angs = np.zeros((tot_spins,3))
    initstate, qc_init = InitSpinProdState(angs,num_spins=tot_spins)
    ## Create Job for execution
    Circuits = [DemoSimulator.PerformManySTsteps(\
                    STEPS=numsteps,dt=ts/numsteps) \
                    for numsteps in range(1,TOTSTEPS+1)]
    InitCircuits = []
    simul_pdf = []
    ## Hardcoded initilization
    for circuit in Circuits:
        qc = qc_init.compose(circuit)
        InitCircuits.append(qc)

    Job = execute(InitCircuits,DemoSimulator.backend,shots=shots)
    ## Monitor Job
    if not DemoSimulator.local_simul:
        job_monitor(Job)
    ## Get counts
    simul_pdf = [Job.result().get_counts(circuit) for circuit in Circuits]
    ## Convert to array of data
    spdf = DemoSimulator.Counts2PDF(Job,Circuits)
    spdf = 1/shots * spdf
    ## Compute exact PDF
    epdf = DemoSimulator.ExactTimeEvol(initstate,t=ts)
    print(epdf)
    ## Compute fidelities
    fidelities = np.array([
        sum(np.sqrt(epdf * spdf[i]))**2 for i in range(len(spdf))
    ])

################################################################################
##                          HERE I PLOT THE FIDELITIES                        ##
################################################################################
    plt.xlabel(r'Number of iterations')
    plt.ylabel(r'$\langle \psi_{exc} | \psi_{sim} \rangle$')
    plt.scatter([i for i in range(1,TOTSTEPS+1)],fidelities)
    plt.show()
    ##plt.savefig('../images/'+DemoSimulator.backend_name+\
    ##            'Fidelity'+\
    ##            str(DemoSimulator.num_spins)+\
    ##            'Steps'+str(TOTSTEPS)+'.pdf')
