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
qpu = provider.get_backend('ibmq_quito')

################################################################################
##                         CREATE QUANTUM ALGORITHM                           ##
################################################################################
QC_ibm = QuantumCircuit(2,2)
QC_ibm.h(0)
## QC_ibm.cx(0,1)
QC_ibm.measure([0,1],[0,1])

################################################################################
##                        EXECUTE QUANTUM ALGORITHM                           ##
################################################################################
job = execute(QC_ibm,backend=qpu)
job_monitor(job)
results = job.result().get_counts(QC_ibm)
print(results)
