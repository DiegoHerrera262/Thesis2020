################################################################################
##      PROGRAM FOR SIMULATING A HEISENBERG CHAIN WITH A QUANTUM COMPUTER     ##
################################################################################
# AUTHOR: Diego Alejandro Herrera Rojas
# DATE: 02/03/21
# DESCRIPTION: In this program, I define a class that simulates a Heisenberg
# anisotropic chain in the presence of an external magnetic
# field, using a quantum algorithm based upon the first
# iteration of a Suzuki Trotter Scheme. I also include methods
# for solving the system exactly. As a first measure, I just
# compare PDFs for measurement on computational basis.

################################################################################
##              IMPORTS NECESSARY TO PERFORM QUANTUM ALGORITHMS               ##
################################################################################
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, Aer
from qiskit import IBMQ, assemble, transpile
from qiskit.circuit import exceptions
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('FigureStyle.mplstyle')

################################################################################
##                     AUXILIAR FUNCTIONS FOR THE PROJECT                     ##
################################################################################


def Dec2nbitBin(num, bits):
    return "{0:b}".format(num).zfill(bits)


def InitSpinProdState(angs, num_spins=2):
    '''
    Function for creating an
    arbitrary product state of
    a spin chain
    '''
    #  Define one-qubit statevecs
    statevecs = np.array([
        [np.cos(p[0]), np.exp(1j*p[1])*np.sin(p[0])] for p in angs
    ])
    # Compute kroenecker product
    statevec = statevecs[0]
    for onequbit in statevecs[1:]:
        statevec = np.kron(onequbit, statevec)
    return statevec


################################################################################
##                  GENERIC PARAMETERS FOR CHAIN SIMULATION                   ##
################################################################################
#  Magnetic field parameters
alpha1 = Parameter('α_1')  # Magnetic field on x axis
alpha2 = Parameter('α_2')  # Magnetic field on y axis
alpha3 = Parameter('α_3')  # Magnetic field on z axis
alphaH = Parameter('α_H')  # Magnetic field magnitude
# Spin interaction parameters
theta1 = Parameter('θ_1')  # Exchange Integral x interaction
theta2 = Parameter('θ_2')  # Exchange Integral y interaction
theta3 = Parameter('θ_3')  # Exchange Integral z interaction

################################################################################
##                     CLASS FOR HAMILTONIAN EVOLUTION                        ##
################################################################################


class QSTsimulator:

    '''
    Class for executing time evolution of Heisenberg chain on external
    magnetic field using quantum algorithm, with Qiskit
    '''

    # Attributes of the class
    num_spins = 2  # Number of spins in the chain
    ExchangeIntegrals = [1.0, 1.0, 1.0]  # See chain Hamiltonian
    ExternalField = [0.0, 0.0, 0.0]  # See chain Hamiltonian
    backend_name = 'qasm_simulator'  # For simulation with Qiskit
    local_simul = True  # For determining if local run
    HamMatEstates = None  # For storing Hamiltonian
    HamMatEnergies = None  # For storing Energy spectrum

    # Init method
    def __init__(self,
                 num_spins=2,
                 ExchangeIntegrals=[1.0, 1.0, 1.0],
                 ExternalField=[0.0, 0.0, 0.0],
                 local_simul=True,
                 **kwargs):
        '''
        Initialize generic parameters for future time evolution
        See chain Hamiltonian for more details
        '''
        self.num_spins = num_spins
        self.ExchangeIntegrals = ExchangeIntegrals
        self.ExternalField = ExternalField
        # Allow non local backend in case I want to run a program on
        # an IBMQ device
        self.local_simul = local_simul
        if self.local_simul:
            self.backend_name = input('Enter local simulator name: ')
            self.backend = Aer.get_backend(self.backend_name)
            # Loaded initial state for local simulations
            if 'init_state' in kwargs:
                self.init_state = kwargs['init_state']
            else:
                self.init_state = np.zeros(2 ** self.num_spins)
                self.init_state[0] = 1
        else:
            IBMQ.load_account()
            provider = IBMQ.get_provider('ibm-q')
            self.backend_name = input('Enter IBMQ device name: ')
            self.backend = provider.get_backend(self.backend_name)
        print('Instantiated QSTsimulator...')
        print('Backend: ', self.backend)

    # IMPORTANT: The detailed ST scheme is presented on the log of this repo
    # the programmer is advised to go to the file log/SimulationAlgorithms.pdf
    #  to fully understand this implementation

################################################################################
##                 AUXILIARY FUNCTIONS FOR PROCESSING RESULTS                 ##
################################################################################
    def Counts2PDF(self, Job, Circuits):
        '''
        Routine for extracting the
        PDF produced after execution
        of a list of circuits on a
        NumPy Array
        '''
        # Get Counts
        simul_pdf = [Job.result().get_counts(circuit) for circuit in Circuits]
        # Convert to array of data
        spdf = np.array([
            [res.get(Dec2nbitBin(num, self.num_spins), 0)
             for num in range(2**self.num_spins)]
            for res in simul_pdf
        ])
        return spdf

################################################################################
##                    EVOLUTION UNDER EXTERNAL FIELD UNITARY                  ##
################################################################################
    def MagFieldEvol(self, spinChain):
        '''
        Qiskit instruction for defining spin eigenstate under
        external field Hamiltonian (see log).
        '''
        Hx = self.ExternalField[0]
        Hy = self.ExternalField[1]
        Hz = self.ExternalField[2]
        H = np.sqrt(Hx**2 + Hy**2 + Hz**2)
        # Parameter values for Qiskit
        PHI = np.arctan2(Hy, Hx) + 2*np.pi
        THETA = np.arccos(Hz/H)
        LAMBDA = np.pi
        # Create demo quantum circuit
        QC_FieldEvol = QuantumCircuit(spinChain)
        # Change from computational to eigenbasis
        for spin in spinChain:
            QC_FieldEvol.u(-THETA, -LAMBDA, -PHI, spin)
            # Evolve with z rotation
            QC_FieldEvol.rz(alphaH, spin)
            # Change from eigenbasis to computational
            QC_FieldEvol.u(THETA, PHI, LAMBDA, spin)
        # Return quantum instruction
        return QC_FieldEvol.to_instruction()

################################################################################
##                    EVOLUTION UNDER TWO SPIN INTERACTION                    ##
################################################################################
    def TwoSpinEvolUnit(self):
        '''
        Qiskit instruction for evolving under
        spin-spin interaction (see log).
        '''
        # Initialization of 2 spin quantum register
        spinPair = QuantumRegister(2, name='s')
        #  Initialization of quantum circuit
        qc_Uij = QuantumCircuit(spinPair)
        #  Convert to computational basis
        qc_Uij.cx(*spinPair)
        qc_Uij.h(spinPair[0])
        # Compute J3 phase
        qc_Uij.rz(theta3, spinPair[1])
        # Compute J1 phase
        qc_Uij.rz(theta1, spinPair[0])
        # Compute J2 phase
        qc_Uij.cx(spinPair[1], spinPair[0])
        qc_Uij.rz(-theta2, spinPair[0])
        qc_Uij.cx(spinPair[1], spinPair[0])
        #  Return to computational basis
        qc_Uij.h(spinPair[0])
        qc_Uij.cx(*spinPair)
        #  Return instruction
        return qc_Uij.to_instruction()

################################################################################
##                        EVOLUTION STEP USING ST SCHEME                      ##
################################################################################
    # Suzuki - Trotter step for time simulation
    def SuzukiTrotter(self, spinChain):
        '''
        Qiskit instruction for performing a ST step. See chain Hamiltonian
        and ST scheme.
        '''
        # Create quantum circuit
        qc_H = QuantumCircuit(spinChain)
        #  Append TwoSpinEvol with field evolution for odd particles
        for idx in range(0, len(spinChain), 2):
            try:
                qc_H.append(
                    self.TwoSpinEvolUnit(), [spinChain[idx], spinChain[idx+1]])
            except:
                continue
        #  Append TwoSpinEvol with field evolution for even particles
        for idx in range(1, len(spinChain), 2):
            try:
                qc_H.append(
                    self.TwoSpinEvolUnit(), [spinChain[idx], spinChain[idx+1]])
            except:
                continue
        # Perform time evolution of mag. field
        qc_H.append(self.MagFieldEvol(spinChain), spinChain)
        # Return instruction
        return qc_H.to_instruction()

    # Circuit for performing several ST steps for future simulation
    def PerformManySTsteps(self, STEPS=200, dt=1.7/200):
        '''
        Quantum circuit that performs time evolution from t=0 to t=simul_time
        using STEPS.
        '''
        # Define parameter values for gates
        # Values for theta parameters
        th1 = 2*self.ExchangeIntegrals[0]*dt
        th2 = 2*self.ExchangeIntegrals[1]*dt
        th3 = 2*self.ExchangeIntegrals[2]*dt
        # Values for alpha parameters
        aH = 2*np.sqrt(sum(comps**2 for comps in self.ExternalField))*dt
        #  Define dictionary for parameter substitution
        params = {
            theta1: th1,
            theta2: th2,
            theta3: th3,
            alphaH: aH,
        }
        #  Create spin chain
        spinChain = QuantumRegister(self.num_spins, name='s')
        # Create measurement register
        measureReg = ClassicalRegister(self.num_spins, name='b')
        # Create quantum circuit
        qc_MST = QuantumCircuit(spinChain, measureReg)
        # Load initial state in local simul
        if self.local_simul:
            qc_MST.initialize(self.init_state, qc_MST.qubits)
        #  Append ST steps to circuit
        for _ in range(STEPS):
            qc_MST.append(self.SuzukiTrotter(spinChain), spinChain)
        #  Perform measurement for further simulation
        qc_MST.measure(spinChain, measureReg)
        #  Return circuit with binded parameters
        try:
            return qc_MST.bind_parameters(params)
        except exceptions.CircuitError:
            return qc_MST

    # Return only PDF after time simulation
    def SimulTimeEvol(self, shots=2048, STEPS=200, t=1.7):
        '''
        Function for computing final
        PDF after time simulation
        '''
        if STEPS > 0:
            # Define parameter dt for ST step
            dt = t/STEPS
            #  Create quantum circuit with many steps
            qc_Evol = self.PerformManySTsteps(STEPS=STEPS, dt=dt)
            # Execute circuit
            Job = execute(qc_Evol, self.backend, shots=shots)
            if not self.local_simul:
                job_monitor(Job)
            if self.backend_name == 'statevector_simulator':
                StateVecs = np.array(Job.result().get_statevector(qc_Evol))
                return np.abs(StateVecs)**2
            else:
                SimResults = Job.result().get_counts(qc_Evol)
                return np.array([
                    SimResults.get(Dec2nbitBin(num, self.num_spins), 0)/shots
                    for num in range(2**self.num_spins)
                ])
        else:
            initstate = np.zeros(2**self.num_spins)
            initstate[0] = 1
            return initstate

    # Consolidating time evolution results using ST scheme
    def EvolAlgorithm(self, NUMSTEPS=200, t=1.7, shots_=2048, save_PDF=True):
        '''
        Function for plotting and consolidating simulation results obtained by
        simulation over backend.
        '''
        #  Create sequence of circuits
        Circuits = [self.PerformManySTsteps(STEPS=idx, dt=t/NUMSTEPS)
                    for idx in range(1, NUMSTEPS+1)]
        # Simulate and store counts for each circuit simulation
        Job = execute(Circuits, self.backend, shots=shots_)
        if not self.local_simul:
            job_monitor(Job)
        if self.backend_name == 'statevector_simulator':
            StateVecs = np.array([Job.result().get_statevector(circuit)
                                  for circuit in Circuits])
        else:
            SimResults = [Job.result().get_counts(circuit)
                          for circuit in Circuits]
        # Accomodate simulation results to reflect PDF evolution over time
        trange = [(idx+1)*t/NUMSTEPS for idx in range(NUMSTEPS)]
        PDF = {'t': trange}
        if self.backend_name == 'statevector_simulator':
            PDFvals = StateVecs * StateVecs.conjugate()
            PDF.update({
                num: PDFvals[:, num] for num in range(2**self.num_spins)
            })
        else:
            PDF.update({num: [res.get(Dec2nbitBin(num, self.num_spins), 0)/shots_
                              for res in SimResults]
                        for num in range(2**self.num_spins)})
        if save_PDF:
            plt.xlabel(r'$t$ (u. a.)')
            plt.ylabel(r'$|\langle \psi_0 | q_n \rangle|^2$')
            for n in range(2**self.num_spins):
                plt.plot(trange, PDF[n], label=Dec2nbitBin(n, self.num_spins))
            plt.legend()
            plt.savefig('../images/'+self.backend_name +
                        'Simul'+str(self.num_spins)+'spins' +
                        'Steps'+str(NUMSTEPS)+'.pdf')
        # Store results in csv file
        SimulData = pd.DataFrame.from_dict(PDF)
        SimulData.to_csv('../datafiles/'+self.backend_name +
                         'SimulData'+str(self.num_spins) +
                         'Steps'+str(NUMSTEPS)+'.csv', index=False)
        # Return simulation results
        return SimulData

################################################################################
##                         STATE FIDELITY COMPUTATION                         ##
################################################################################
    # Diagonalize Hamiltonian
    def DiagHamilt(self):
        '''
        Diagonalize matrix Hamiltonian for
        future time evolution
        '''
        # Define Pauli operators
        PauliX = np.array([
            [0, 1],
            [1, 0]
        ])
        PauliY = np.array([
            [0, -1j],
            [1j, 0]
        ])
        PauliZ = np.array([
            [1, 0],
            [0, -1]
        ])
        PauliOps = [PauliX, PauliY, PauliZ]
        # Definition of two-qubit Hamiltonian
        Hij = np.sum(Jint * np.kron(Pauli, Pauli)
                     for Jint, Pauli in zip(self.ExchangeIntegrals, PauliOps))
        #  Definition of one-qubit Hamiltonian
        Hi = np.sum(hcomp * Pauli
                    for hcomp, Pauli in zip(self.ExternalField, PauliOps))
        # Definition of Chain Hamiltonian
        Hchain = np.sum(
            np.kron(np.identity(2**idx),
                    np.kron(Hij, np.identity(2**(self.num_spins-(idx + 2))))) +
            np.kron(np.identity(2**idx),
                    np.kron(Hi, np.identity(2**(self.num_spins-(idx + 1)))))
            for idx in range(self.num_spins-1)
        ) + np.kron(np.identity(2**(self.num_spins - 1)), Hi)
        #  Diagonalization of Hamiltonian
        self.HamMatEnergies, self.HamMatEstates = np.linalg.eig(Hchain)

    # Perform exact time evolution
    def ExactStateVecTimeEvol(self, t=1.7):
        '''
        Compute evolution operator and
        perform matrix multiplication
        '''
        # Compute unitary evolution operator
        Udiag = np.diag(np.exp(-1j*t*self.HamMatEnergies))
        try:
            Uev = np.matmul(np.matmul(self.HamMatEstates, Udiag),
                            self.HamMatEstates.transpose().conjugate())
        except Exception as e:
            print('Error: ', e.__class__)
            print('Forgot to diagonalize Hamiltonian')
        # Return application to initial state
        return Uev.dot(self.init_state)

    #  Compute exact PDF
    def ExactTimeEvol(self, t=1.7):
        '''
        Function for computing exact PDF
        after time evolution
        '''
        #  Perform time evolution
        psi = self.ExactStateVecTimeEvol(t=t)
        # Return norm squared profile
        return np.abs(psi)**2

    # Plot exact probability density
    def ExacEvolAlgorithm(self, NUMSTEPS=200, t=1.7, save_PDF=True):
        '''
        Perform exact time evolution by
        diagonalization and store data
        for future comparison
        '''
        trange = [(idx+1)*t/NUMSTEPS for idx in range(NUMSTEPS)]
        # Create array of evolved states
        sts = np.array([self.ExactStateVecTimeEvol(t=ts)
                        for ts in trange])
        # Transpose and compute norm for PDF
        PDFvals = sts * sts.conjugate()
        # Create dictionary for DataFrame
        PDF = {'t': trange}
        PDF.update({n: PDFvals[:, n] for n in range(2**self.num_spins)})
        if save_PDF:
            plt.xlabel(r'$t$ (u. a.)')
            plt.ylabel(r'$|\langle \psi_0 | q_n \rangle|^2$')
            for n in range(2**self.num_spins):
                plt.plot(trange, PDF[n], label=Dec2nbitBin(n, self.num_spins))
            plt.legend()
            plt.savefig('../images/'+self.backend_name +
                        'ExactEvol'+str(self.num_spins)+'spins' +
                        'Steps'+str(NUMSTEPS)+'.pdf')
        # Store results in csv file
        EvolData = pd.DataFrame.from_dict(PDF)
        EvolData.to_csv('../datafiles/' +
                        'ExactEvolData'+str(self.num_spins) +
                        'Steps'+str(NUMSTEPS)+'.csv', index=False)
        # Return simulation results
        return EvolData

    #  Compute statevector evolution of |00...0>
    def StatevecSuzukiTrotter(self, STEPS=200, dt=1.7/200):
        '''
        Function for computing
        evolution statevector
        for fidelity
        '''
        # Create circuit that performs evolution
        qc_SimulStv = self.PerformManySTsteps(STEPS, dt)
        # Execute circuit
        local_backend = Aer.get_backend('statevector_simulator')
        # Get statevector
        return execute(qc_SimulStv, local_backend).result().get_statevector()

    # Compute fidelity
    def TeorFidelity(self, STEPS=200, ts=1.7):
        '''
        Function for computing
        state fidelity from
        qiskit simulation
        '''
        # Define parameters for simulation
        dt = ts/STEPS
        initstate = np.zeros(2**(self.num_spins))
        initstate[0] = 1
        # Compute simulated final state
        finstate_sim = self.StatevecSuzukiTrotter(STEPS=STEPS, dt=dt)
        #  Compute exact final state
        finstate_exa = self.ExactTimeEvol(initstate, t=ts)
        # Compute dot product
        fidelity = finstate_exa.conjugate().dot(np.array(finstate_sim))
        #  return absolute value squared
        return np.abs(fidelity)**2

    def ExpFidelity(self, shots=4096, STEPS=200, ts=1.7):
        '''
        Function for computing
        pdf fidelity from simulation
        both local and on IBMQ
        '''
        # Define parameters for simulation
        dt = ts/STEPS
        initstate = np.zeros(2**(self.num_spins))
        initstate[0] = 1
        #  Compute simulated PDF
        qc_Sim = self.PerformManySTsteps(STEPS=STEPS, dt=dt)
        Job = execute(qc_Sim, self.backend, shots=shots)
        job_monitor(Job)
        simul_pdf = Job.result().get_counts()
        spdf = np.array([
            simul_pdf.get(Dec2nbitBin(num, self.num_spins), 0)/shots
            for num in range(2**self.num_spins)
        ])
        #  Compute exact PDF
        ex_vec = self.ExactTimeEvol(initstate, t=ts)
        epdf = np.abs(ex_vec)**2
        # Compute statistical fidelity
        return sum(np.sqrt(epdf * spdf))**2
