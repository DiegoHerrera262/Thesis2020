################################################################################
##      PROGRAM FOR SIMULATING A HEISENBERG CHAIN WITH A QUANTUM COMPUTER     ##
################################################################################
# AUTHOR: Diego Alejandro Herrera Rojas
# DATE: 14/05/21
# DESCRIPTION: This program builds upon my previous insights on time simulation
# of a Heisenberg Spin chain with a quantum computer. Here, I try
# to implement a variational algorithm for learning the thermal
# states of a Heisenberg chain, based on a Layer with corresopnds
# to a Heisenberg Hamiltonian on itself

################################################################################
##              IMPORTS NECESSARY TO PERFORM QUANTUM ALGORITHMS               ##
################################################################################
import pennylane as qml
from pennylane import numpy as np

################################################################################
##                     AUXILIAR FUNCTIONS FOR THE PROJECT                     ##
################################################################################


def Dec2nbitBin(num, bits):
    return [int(d) for d in "{0:b}".format(num).zfill(bits)]


class VQThermalizer:

    '''
    Class for thermalizing a spin system using a Variation Quantum 
    thermalizer according to Quantum Hamiltonian-Based Models &
    the Variational Quantum Thermalizer Algorithm 
    [arXiv:1910.02071v1 [quant-ph] 4 Oct 2019]
    '''

    # Attribures of the class
    num_spins = 2  # Number of spins in the chain
    ExchangeIntegrals = [1.0, 1.0, 1.0]  # See chain Hamiltonian
    ExternalField = [0.0, 0.0, 0.0]  # See chain Hamiltonian
    backend_name = 'default.qubit'  # For simulation with PennyLane
    SysHamiltonian = None  # PennyLane Hamiltonian of chain
    HamMatEstates = None  # For storing Ham. eigenstates
    HamMatEnergies = None  # For storing Eigenenergies
    ThermalQNode = None  # For storing qnode for emasuring Hamiltonian

    def __init__(self,
                 num_spins=2,
                 ExchangeIntegrals=[1.0, 1.0, 1.0],
                 ExternalField=[0.0, 0.0, 0.0],):
        '''
        Initialize thermalizer ina similar fashion as
        QSTSimulator
        '''
        self.num_spins = num_spins
        self.ExchangeIntegrals = ExchangeIntegrals
        self.ExternalField = ExternalField
        self.backend_name = input('Enter local simulator name: ')
        self.device = qml.device(self.backend_name, wires=self.num_spins)
        # Communicate initialization details
        print('Instantiated VQThermalizer...')
        print('Backend: ', self.backend_name)

        # IMPORTANT: The detailed ST scheme is presented on the log of this repo
        # the programmer is advised to go to the file log/SimulationAlgorithms.pdf
        # to fully understand this implementation

################################################################################
##                  INITIALIZE AND DIAGONALIZE HAMILTONIAN                    ##
################################################################################
    def GenHamiltonian(self):
        '''
        Define and diagonalize chains hamiltonian
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
        # Definition of one-qubit Hamiltonian
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
        # Diagonalization of Hamiltonian
        self.HamMatEnergies, self.HamMatEstates = np.linalg.eig(Hchain)
        # Storing system hamiltonian
        self.SysHamiltonian = qml.Hermitian(
            Hchain, wires=range(self.num_spins))

################################################################################
##             DEFINITION OF QNN LAYER BASED ON SPIN SIMULATION               ##
################################################################################
    def QNNLayer(self, params):
        '''
        Definition of a single ST step for
        layering QNN
        '''
        ExcInts = params[0:3]
        ExtField = params[3:6]
        #  Parameters for external
        # field evol
        Hx = ExtField[0]
        Hy = ExtField[1]
        Hz = ExtField[2]
        H = np.sqrt(Hx**2 + Hy**2 + Hz**2)
        # Parameter values for Qiskit
        PHI = np.arctan2(Hy, Hx) + 2*np.pi
        THETA = np.arccos(Hz/H)
        LAMBDA = np.pi
        # Cascade Spin pair interaction
        for idx in range(self.num_spins-1):
            # Convert to computational basis
            qml.CNOT(wires=[idx, idx+1])
            qml.Hadamard(wires=idx)
            #  Compute J3 phase
            qml.RZ(ExcInts[2], wires=idx+1)
            # Compute J1 phase
            qml.RZ(ExcInts[0], wires=idx)
            # Compute J2 Phase
            qml.CNOT(wires=[idx, idx+1])
            qml.RZ(-ExcInts[1], wires=idx+1)
            qml.CNOT(wires=[idx, idx+1])
            # Return to computational basis
            qml.Hadamard(wires=idx)
            qml.CNOT(wires=[idx, idx+1])
            # Include external field
            qml.U3(-THETA, -LAMBDA, -PHI, wires=idx)
            qml.RZ(H, wires=idx)
            qml.U3(THETA, PHI, LAMBDA, wires=idx)
        #  Include external field for last spin
        qml.U3(-THETA, -LAMBDA, -PHI, wires=self.num_spins-1)
        qml.RZ(H, wires=self.num_spins-1)
        qml.U3(THETA, PHI, LAMBDA, wires=self.num_spins-1)

################################################################################
##                    DEFINITION OF QNN FOR THERMALIZATION                    ##
################################################################################
    def QNN(self, params):
        '''
        QNN for learning thermal states
        '''
        for idx in range(0, len(params), 6):
            self.QNNLayer(params[idx:idx+6])

################################################################################
##                    DEFINITION OF QNN FOR THERMALIZATION                    ##
################################################################################
    def BasisQNN(self, params, i=None):
        '''
        Including Initial basis state for
        Energy computation
        '''
        # Prepare initial state
        qml.templates.BasisStatePreparation(i, wires=range(self.num_spins))
        # Include QNN
        self.QNN(params)
        # Return expected value of Hmailtonian
        return qml.expval(self.SysHamiltonian)

    def SetThermalQNode(self):
        '''
        Set thermal QNode for computation
        of cost function
        '''
        self.ThermalQNode = qml.QNode(self.BasisQNN, self.device)
