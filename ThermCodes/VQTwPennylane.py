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
from scipy.optimize import minimize

################################################################################
##                     AUXILIAR FUNCTIONS FOR THE PROJECT                     ##
################################################################################


def Dec2nbitBin(num, bits):
    return [int(d) for d in "{0:b}".format(num).zfill(bits)]


def sigmoid(x):
    return np.exp(x)/(np.exp(x) + 1)


def TraceDistance(A, B):
    return 0.5 * np.trace(np.absolute(np.add(A, -1*B)))


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
                 ExternalField=[0.0, 0.0, 0.0],
                 Beta=1):
        '''
        Initialize thermalizer ina similar fashion as
        QSTSimulator
        '''
        self.num_spins = num_spins
        self.ExchangeIntegrals = ExchangeIntegrals
        self.ExternalField = ExternalField
        self.backend_name = input('Enter local simulator name: ')
        self.device = qml.device(self.backend_name, wires=self.num_spins)
        self.beta = Beta
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

################################################################################
##                    COMPUTATION IF TRIAL ENSEMBLE ENTROPY                   ##
################################################################################
    def GenProbDist(self, params):
        '''
        For generating a prob dits corres-
        ponding to product mized state.
        Dist of ith is accessed by Dits[i]
        '''
        return np.vstack([sigmoid(params), 1-sigmoid(params)]).T

    def EnsembleEntropy(self, ProbDist):
        '''
        Compute ensemble entropy
        from prob dist. 
        '''
        ent = 0.0
        # E = sum of entropies since
        # ansatz is prod state
        for dist in ProbDist:
            ent += -1 * np.sum(dist * np.log(dist))
        return ent

################################################################################
##                         DEFINITION OF COST FUNCTION                        ##
################################################################################
    def MapParams(self, params):
        # Set ensemble params
        dist_params = params[0:self.num_spins]
        # Set QNN params
        qnn_params = params[self.num_spins:]
        return dist_params, qnn_params

    def BasisStateProb(self, probDist, i=0):
        '''
        Probability of basis state
        in ensemble
        '''
        # Convert number to array
        state = Dec2nbitBin(i, self.num_spins)
        # Iterate to find probs
        prob = 1.0
        for idx, ent in enumerate(state):
            prob = prob * probDist[idx, int(ent)]
        return prob

    def CostFunc(self, params):
        '''
        Cost function is ensemble
        free energy
        '''
        dist_params, qnn_params = self.MapParams(params)
        # Compute prob distribution
        dist = self.GenProbDist(dist_params)
        # Compute Hamiltonian exp val
        HamExpval = 0
        for num in range(2 ** self.num_spins):
            HamExpval += self.BasisStateProb(dist, i=num) * \
                self.ThermalQNode(
                    qnn_params, i=Dec2nbitBin(num, self.num_spins
                                              ))
        # Return free energy
        e = HamExpval * self.beta - self.EnsembleEntropy(dist)
        print('Cost Func: {}'.format(e))
        return e

################################################################################
##                   OPTIMIZATION OF VARIATIONAL PARAMETERS                   ##
################################################################################
    def InitOptimizer(self):
        '''
        Initialize Hamiltonian
        and Qnode
        '''
        self.GenHamiltonian()
        self.SetThermalQNode()

    def GetOptimalParams(self, layers=3, optimizer='COBYLA', maxiter=1600):
        '''
        Use layers QNNLayer constructs
        for optimizationl algorithms
        '''
        # Create random seed for algorithm
        params = 300 * np.random.rand(self.num_spins + layers * 6) - 150
        # Default COBYLA is thought to be
        # more effcient
        out = minimize(self.CostFunc,
                       x0=params,
                       method=optimizer,
                       options={'maxiter': maxiter})
        return out['x']

################################################################################
##                          BUILD DENSITY MATRIX                              ##
################################################################################
    def BuildDensityBasisState(self, params, num=0):
        '''
        Build density matrix associated
        to params and initial basis state
        '''
        # Get device state
        self.ThermalQNode(params, i=Dec2nbitBin(num, self.num_spins))
        state = self.device.state
        # Return density matrix
        return np.outer(state, np.conj(state))

    def ThermalDensityMatrix(self, params):
        '''
        Build thermal density matrix
        '''
        dist_params, qnn_params = self.MapParams(params)
        # Compute prob distribution
        dist = self.GenProbDist(dist_params)
        # Add succesive density mats with weights
        density = np.zeros((2**self.num_spins, 2**self.num_spins))
        for num in range(2**self.num_spins):
            density = np.add(
                density,
                self.BasisStateProb(dist, i=num) *
                self.BuildDensityBasisState(qnn_params, num=num)
            )
        return density

    def TeorThermDensity(self):
        '''
        Build theoretical thermal density
        '''
        # Exponentiate with eigenvectors
        d = np.exp(-self.beta * self.HamMatEnergies)
        return np.matmul(
            np.matmul(self.HamMatEstates, d),
            self.HamMatEstates.conj().T
        )
