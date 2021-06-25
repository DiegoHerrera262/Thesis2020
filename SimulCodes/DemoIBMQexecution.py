################################################################################
##              PROGRAM FOR TESTING REMOTE CIRCUIT EXECUTION ON IBM Q         ##
################################################################################
## AUTHOR: Diego Alejandro Herrera Rojas
## DATE: 18/03/21
## DESCRIPTION: In this program I test the workings of my IBM Q account and the
##              results obtained using a real quantum device. For my purpouses,
##              perhaps ibmq_quito or ibmq_lima might suffice.

################################################################################
##              IMPORTS NECESSARY TO PERFORM QUANTUM ALGORITHMS               ##
################################################################################
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, Aer
from qiskit import IBMQ, assemble, transpile
from qiskit.tools.monitor import job_monitor
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('FigureStyle.mplstyle')

################################################################################
##                          SET UP IBM Q ACCOUNT                              ##
################################################################################
IBMQ.load_account()

################################################################################
##                          SET UP IBM Q PROVIDER                             ##
################################################################################
provider = IBMQ.get_provider('ibm-q')

################################################################################
##                           SET UP IBM Q DEVICE                              ##
################################################################################
qpu = provider.get_backend('ibmq_armonk')

################################################################################
##                         CREATE QUANTUM ALGORITHM                           ##
################################################################################
def SpinPrecession(tsimul):
    '''
    Funciton for simulating spin
    precession with Qiskit
    '''
    Simul = QuantumCircuit(1,1)
    Simul.h(0)
    Simul.ry(-2*np.pi*tsimul,0)
    Simul.measure(0,0)
    return Simul

################################################################################
##                      CREATE LIST OF QUANTUM CIRCUITS                       ##
################################################################################
tsimuls = np.linspace(0,1,20)
DemoCircuits = [SpinPrecession(ti) for ti in tsimuls]

################################################################################
##                        EXECUTE QUANTUM ALGORITHM                           ##
################################################################################
job = execute(DemoCircuits,backend=qpu,shots=2048)
job_monitor(job)
## Reindex data for plotting
results = np.array([[job.result().get_counts(circuit).get(val,0) \
            for val in ['0','1']] \
            for circuit in DemoCircuits])

################################################################################
##                      DISPLAY RESULTS OF EXECUTION                          ##
################################################################################
plt.xlabel(r'$t$ (u. a.)')
plt.ylabel(r'$|\langle \psi | q\rangle|^2$')
plt.plot(tsimuls, results[:,0] * 1/2048, label=r'$p(0)$')
plt.plot(tsimuls, results[:,1] * 1/2048, label=r'$p(1)$')
plt.legend()
plt.show()
