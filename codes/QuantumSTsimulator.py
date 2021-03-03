################################################################################
##      PROGRAM FOR SIMULATING A HEISENBERG CHAIN WITH A QUANTUM COMPUTER     ##
################################################################################
## AUTHOR: Diego Alejandro Herrera Rojas
## DATE: 02/03/21
## DESCRIPTION: In this program, I define a class that simulates a Heisenberg
##              anisotropic chain in the presence of an external magnetic
##              field, using a quantum algorithm based upon the first
##              iteration of a Suzuki Trotter Scheme. I also include methods
##              for solving the system exactly. As a first measure, I just
##              compare PDFs for measurement on computational basis.

################################################################################
##              IMPORTS NECESSARY TO PERFORM QUANTUM ALGORITHMS               ##
################################################################################
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
plt.style.use('FigureStyle.mplstyle')

################################################################################
##               FORMATTING DECIMAL NUMBER INTO BINARY NUMBER                 ##
################################################################################
def Dec2nbitBin(num,bits):
    return "{0:b}".format(num).zfill(bits)

################################################################################
##                  GENERIC PARAMETERS FOR CHAIN SIMULATION                   ##
################################################################################
## Magnetic field parameters
alpha1 = Parameter('α_1')                   ## Magnetic field on x axis
alpha2 = Parameter('α_2')                   ## Magnetic field on y axis
alpha3 = Parameter('α_3')                   ## Magnetic field on z axis
## Spin interaction parameters
theta1 = Parameter('θ_1')                   ## Exchange Integral x interaction
theta2 = Parameter('θ_2')                   ## Exchange Integral y interaction
theta3 = Parameter('θ_3')                   ## Exchange Integral z interaction

################################################################################
##                     CLASS FOR HAMILTONIAN EVOLUTION                        ##
################################################################################
class QSTsimulator:

    '''
    Class for executing time evolution of Heisenberg chain on external
    magnetic field using quantum algorithm, with Qiskit
    '''

    ## Attributes of the class
    num_spins = 2                           ## Number of spins in the chain
    ExchangeIntegrals = [1.0,1.0,1.0]       ## See chain Hamiltonian
    ExternalField = [0.0,0.0,0.0]           ## See chain Hamiltonian
    backend = 'qasm_simulator'              ## For simulation with Qiskit

    ## Init method
    def __init__(self,\
                num_spins=2,\
                ExchangeIntegrals=[1.0,1.0,1.0],\
                ExternalField=[0.0,0.0,0.0],\
                local_simul=True):
                '''
                Initialize generic parameters for future time evolution
                See chain Hamiltonian for more details
                '''
                self.num_spins = num_spins
                self.ExchangeIntegrals = ExchangeIntegrals
                self.ExternalField = ExternalField
                ## Allow non local backend in case I want to run a program on
                ## an IBMQ device
                if local_simul:
                    self.backend = Aer.get_backend('qasm_simulator')
                else:
                    self.backend = Aer.get_backend(input('Enter IBM backend: '))

    ## IMPORTANT: The detailed ST scheme is presented on the log of this repo
    ## the programmer is advised to go to the file log/SimulationAlgorithms.pdf
    ## to fully understand this implementation

################################################################################
##              EVOLUTION UNDER Z-Y PROJECTION OF MAGNETIC FIELD              ##
################################################################################
    ## Instruction for evolution under y-z magnetic field interaction
    def U_23(self,spinChain):
        '''
        Qiskit instruction for ST step corresponding to x-y magnetic field
        interaction. See chain Hamiltonian
        '''
        ## Create a placeholder quantum circuit
        qc_U23 = QuantumCircuit(spinChain)
        ## Perform y-z rotations
        for spin in spinChain:
            ## For left of ST
            qc_U23.ry(alpha2,spin)
            ## For middle of ST
            qc_U23.rz(alpha3,spin)
            ## For inverted left (right) of ST
            qc_U23.ry(alpha2,spin)
        ## Return instruction
        return qc_U23.to_instruction()

################################################################################
##  EVOLUTION UNDER SPIN-SPIN INTERACTION AND X PROJECTION OF MAGNETIC FIELD  ##
################################################################################
    ## Left unitary evolution: After z-mag. field evolution
    def U_ijLeft(self):
        '''
        Qiskit instruction for evolution under spin-spin interaction on the
        right of first iteration of ST scheme. See ST scheme.
        '''
        ## Initialization of 2 spin quantum register
        spinPair = QuantumRegister(2,name='s')
        ## Initialization of quantum circuit
        qc_Uij = QuantumCircuit(spinPair)
        ## Convert to computational basis
        qc_Uij.cx(*spinPair)
        qc_Uij.h(spinPair[0])
        ## Include x-magnetic field evolution for s_1
        qc_Uij.rx(alpha1,spinPair[1])
        ## Compute J3 phase
        qc_Uij.rz(theta3,spinPair[1])
        ## Compute J1 phase
        qc_Uij.rz(theta1,spinPair[0])
        ## Compute J2 phase
        qc_Uij.cx(spinPair[1],spinPair[0])
        qc_Uij.rz(-theta2,spinPair[0])
        qc_Uij.cx(spinPair[1],spinPair[0])
        ## Return to computational basis
        qc_Uij.h(spinPair[0])
        qc_Uij.cx(*spinPair)
        ## Return instruction
        return qc_Uij.to_instruction()

    ## Right unitary evolution: Before z-mag. field evolution
    def U_ijRight(self):
        '''
        Qiskit instruction for evolution under spin-spin interaction on the
        left of first iteration of ST scheme. See ST scheme.
        '''
        ## Initialization of 2 spin quantum register
        spinPair = QuantumRegister(2,name='s')
        ## Initialization of quantum circuit
        qc_Uij = QuantumCircuit(spinPair)
        ## Convert to computational basis
        qc_Uij.cx(*spinPair)
        qc_Uij.h(spinPair[0])
        ## Compute J3 phase
        qc_Uij.rz(theta3,spinPair[1])
        ## Compute J1 phase
        qc_Uij.rz(theta1,spinPair[0])
        ## Compute J2 phase
        qc_Uij.cx(spinPair[1],spinPair[0])
        qc_Uij.rz(-theta2,spinPair[0])
        qc_Uij.cx(spinPair[1],spinPair[0])
        ## Include x-magnetic field evolution for s_1
        qc_Uij.rx(alpha1,spinPair[1])
        ## Return to computational basis
        qc_Uij.h(spinPair[0])
        qc_Uij.cx(*spinPair)
        ## Return instruction
        return qc_Uij.to_instruction()

################################################################################
##                        EVOLUTION STEP USING ST SCHEME                      ##
################################################################################
    ## Suzuki - Trotter step for time simulation
    def SuzukiTrotter(self,spinChain):
        '''
        Qiskit instruction for performing a ST step. See chain Hamiltonian
        and ST scheme.
        '''
        ## Create quantum circuit
        qc_H = QuantumCircuit(spinChain)
        ## Append UijRight with field evolution for odd particles
        for idx in range(0,len(spinChain),2):
            try:
                qc_H.append(self.U_ijRight(),[spinChain[idx],spinChain[idx+1]])
            except:
                continue
        ## Append UijRight with field evolution for even particles
        for idx in range(1,len(spinChain),2):
            try:
                qc_H.append(self.U_ijRight(),[spinChain[idx],spinChain[idx+1]])
            except:
                continue
        ## Perform time evolution for x-mag. field on s_0
        qc_H.rx(alpha1,spinChain[0])
        ## Perform time evolution of x-y mag. field
        qc_H.append(self.U_23(spinChain),spinChain)
        ## Perform time evolution for x-mag. field on s_0
        qc_H.rx(alpha1,spinChain[0])
        ## Append UijRight with field evolution for even particles
        for idx in range(1,len(spinChain),2):
            try:
                qc_H.append(self.U_ijRight(),[spinChain[idx],spinChain[idx+1]])
            except:
                continue
        ## Append UijRight with field evolution for odd particles
        for idx in range(0,len(spinChain),2):
            try:
                qc_H.append(self.U_ijRight(),[spinChain[idx],spinChain[idx+1]])
            except:
                continue
        ## Return instruction
        return qc_H.to_instruction()

    ## Circuit for performing several ST steps for future simulation
    def PerformManySTsteps(self,STEPS=200,dt=1.7/200):
        '''
        Quantum circuit that performs time evolution from t=0 to t=simul_time
        using STEPS
        '''
        ## Define parameter values for gates
        ## Values for theta parameters
        th1 = self.ExchangeIntegrals[0]*dt/2
        th2 = self.ExchangeIntegrals[1]*dt/2
        th3 = self.ExchangeIntegrals[2]*dt/2
        ## Values for alpha parameters
        a1 = self.ExternalField[0]*dt/2
        a2 = self.ExternalField[1]*dt/2
        a3 = self.ExternalField[2]*dt
        ## Define dictionary for parameter substitution
        params = {
            theta1:th1,
            theta2:th2,
            theta3:th3,
            alpha1:a1,
            alpha2:a2,
            alpha3:a3,
        }
        ## Create spin chain
        spinChain = QuantumRegister(self.num_spins,name='s')
        ## Create measurement register
        measureReg = ClassicalRegister(self.num_spins,name='b')
        ## Create quantum circuit
        qc_MST = QuantumCircuit(spinChain,measureReg)
        ## Append ST steps to circuit
        for _ in range(STEPS):
            qc_MST.append(self.SuzukiTrotter(spinChain),spinChain)
        ## Perform measurement for further simulation
        qc_MST.measure(spinChain,measureReg)
        ## Return circuit with binded parameters
        return qc_MST.bind_parameters(params)

    ## Consolidating time evolution results using ST scheme
    def EvolAlgorithm(self,NUMSTEPS=200,t=1.7,shots_=2048,save_PDF=True):
        '''
        Function for plotting and consolidating simulation results obtained by
        simulation over backend.
        '''
        ## Create sequence of circuits
        Circuits = [self.PerformManySTsteps(STEPS=idx,dt=t/NUMSTEPS) \
                    for idx in range(1,NUMSTEPS+1)]
        ## Simulate and store counts for each circuit simulation
        SimResults = \
        [execute(circuit,self.backend,shots=shots_).result().get_counts() \
            for circuit in Circuits]
        ## Accomodate simulation results to reflect PDF evolution over time
        PDF = {num:[res.get(Dec2nbitBin(num,self.num_spins),0)/shots_ \
            for res in SimResults] \
            for num in range(2**self.num_spins)}
        if save_PDF:
            trange = [(idx+1)*t/NUMSTEPS for idx in range(NUMSTEPS)]
            plt.xlabel(r'$t$ (u. a.)')
            plt.ylabel(r'$|\langle \psi_0 | q_n \rangle|^2$')
            for n in range(2**self.num_spins):
                plt.plot(trange,PDF[n],label=Dec2nbitBin(n,self.num_spins))
            plt.legend()
            plt.savefig('Simul'+str(self.num_spins)+'spins.pdf')
        ## Return simulation results
        return PDF
