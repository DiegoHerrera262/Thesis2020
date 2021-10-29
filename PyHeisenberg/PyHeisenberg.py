################################################################################
##         CORE ROUTINES FOR SIMULATION OF GRAPH HEISENBERG HAMILTONIAN       ##
################################################################################
# AUTHOR: Diego Alejandro Herrera Rojas
# DATE: 11/18/21
# DESCRIPTION: In this program, I define subroutines that will help on defining # Suzuki-Trotter schemes for simulation. They are based upon the graph
# constructs designed on file graphRoutines.py

import graphRoutines as gr
import numpy as np
from scipy.linalg import expm
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm
import auxiliaryRoutines as aux
from qiskit.circuit import exceptions
from qiskit import IBMQ, execute, Aer
from qiskit.tools.monitor import job_monitor
from qiskit.utils import QuantumInstance
from qiskit.test.mock import FakeSantiago
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.ignis.mitigation.measurement import complete_meas_cal
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter

plt.style.use('FigureStyle.mplstyle')

defaultSpinInteractions = {
    (0, 1): [0.5, 0.5, 0.5],
    (0, 3): [0.5, 0.5, 0.5],
    (0, 5): [0.5, 0.5, 0.5],
    (1, 2): [0.5, 0.5, 0.5],
    (1, 4): [0.5, 0.5, 0.5],
    (4, 3): [0.5, 0.5, 0.5],
    (2, 5): [0.5, 0.5, 0.5],
    (3, 2): [0.5, 0.5, 0.5],
    (4, 5): [0.5, 0.5, 0.5],
}
defaultExternalField = {
    0: [0, 0, 0],
    1: [0, 0, 0],
    2: [0, 0, 0],
    3: [0, 0, 0],
    4: [0, 0, 0],
    5: [0, 0, 0],
}


class HeisenbergGraph:

    '''
    Class for simulating a generic Heisenberg hamiltonian
    defined on a graph
    '''

    spinInteractions = defaultSpinInteractions
    externalField = defaultExternalField

    def __init__(self,
                 spinInteractions=defaultSpinInteractions,
                 externalField=defaultExternalField,
                 withColoring=True,
                 **kwargs):
        self.spinInteractions = spinInteractions
        self.externalField = externalField
        self.graph = gr.generateGraph(spinInteractions, externalField)
        # Color the graph for parallelisation
        if withColoring:
            self.matching, self.graphColors = gr.colorMatching(self.graph)
        else:
            self.graph.es['color'] = [-1 for _ in range(len(self.graph.es))]
            self.graphColors = [-1]
            self.matching = {
                -1: self.graph
            }
        # Setting up simulation environment
        try:
            self.localSimulation = kwargs['localSimulation']
            if not self.localSimulation:
                IBMQ.load_account()
                provider = IBMQ.get_provider('ibm-q')
                try:
                    self.backendName = kwargs['backendName']
                except KeyError:
                    self.backendName = 'ibmq_santiago'
                self.backend = provider.get_backend(self.backendName)
            if self.localSimulation:
                try:
                    self.backendName = kwargs['backendName']
                except KeyError:
                    self.backendName = 'qasm_simulator'
                self.backend = Aer.get_backend(self.backendName)
                # Include error model
                try:
                    self.noisySimulation = kwargs['noisySimulation']
                    if self.noisySimulation:
                        self.backend = self.backend.from_backend(
                            FakeSantiago())
                except KeyError:
                    pass
        except KeyError:
            self.localSimulation = True
            self.backendName = 'qasm_simulator'
            self.backend = Aer.get_backend(self.backendName)
        # Load initial state
        try:
            self.initialState = kwargs['initialState']
            self.withInitialState = True
        except KeyError:
            initState = np.zeros(2**len(self.graph.vs))
            initState[0] = 1
            self.initialState = initState
            self.withInitialState = False

################################################################################
##                ANALYTIC ROUTINES FOR COMPUTING HAMILTONIAN                 ##
################################################################################

    def edgeHamiltonian(self, edge):
        '''
        Function for computing spin
        interaction on an edge
        '''
        numSpins = len(self.graph.vs)
        return sum(edge['exchangeIntegrals'][i] * aux.pauliProductMatrix(
            numSpins,
            edge.tuple,
            (i, i)
        ) for i in range(3))

    def vertexHamiltonian(self, vertex):
        '''
        Function for computing field
        interaction on a vertex
        '''
        numSpins = len(self.graph.vs)
        return sum(vertex['externalField'][i] * aux.pauliMatrix(
            numSpins,
            vertex.index,
            i
        ) for i in range(3))

    def SpinSpinHamiltonian(self):
        '''
        Function that computes spin-spin 
        Hamiltonian matrix from graph structure
        '''
        Hij = sum(
            self.edgeHamiltonian(edge)
            for edge in self.graph.es
        )
        return Hij

    def FieldHamiltonian(self):
        '''
        Function that computes field
        Hamiltonian matrix from graph
        '''
        Hf = sum(
            self.vertexHamiltonian(vertex)
            for vertex in self.graph.vs
        )
        return Hf

    def HamiltonianMatrix(self):
        '''
        Function that computes Hamiltonian matrix
        from graph structure
        '''
        return self.FieldHamiltonian() + self.SpinSpinHamiltonian()

################################################################################
##         DEFINITION OF FUNDAMENTAL QUANTUM CIRCUIT FOR SIMULATION           ##
################################################################################

    def vertexCircuit(self, vertex, spinChain):
        '''
        Function for building circuit that 
        performs vertex Hamiltonian evolution
        '''
        Hx = vertex['externalField'][0]
        Hy = vertex['externalField'][1]
        Hz = vertex['externalField'][2]
        H = np.sqrt(Hx**2 + Hy**2 + Hz**2)
        # Parameter values for Qiskit
        PHI = np.arctan2(Hy, Hx) + 2*np.pi if H > 0 else 0
        THETA = np.arccos(Hz/H) if H > 0 else 0
        LAMBDA = np.pi if H > 0 else 0
        # Align to field main axis
        qcVertex = QuantumCircuit(spinChain)
        qcVertex.u(-THETA, -LAMBDA, -PHI, spinChain[vertex.index])
        qcVertex.rz(vertex['paramExternalField'], spinChain[vertex.index])
        qcVertex.u(THETA, PHI, LAMBDA, spinChain[vertex.index])
        return qcVertex.to_instruction()

    def fieldCircuit(self, spinChain):
        '''
        Function for building circuit that
        performs field Hamiltonian evolution
        '''
        qcField = QuantumCircuit(spinChain)
        for vertex in self.graph.vs:
            qcField.append(
                self.vertexCircuit(vertex, spinChain),
                spinChain
            )
        return qcField.decompose().to_instruction()

    def edgeCircuit(self, edge, spinChain):
        '''
        Function for building circuit that
        performs edge Hamiltonian evolution
        '''
        start, end = edge.tuple
        J = edge['paramExchangeIntegrals']
        qcEdge = QuantumCircuit(spinChain)
        # Convert to computational basis
        qcEdge.cx(spinChain[start], spinChain[end])
        qcEdge.h(spinChain[start])
        # Compute J3 phase
        qcEdge.rz(J[2], spinChain[end])
        # Compute J1 phase
        qcEdge.rz(J[0], spinChain[start])
        # Copmute J2 phase
        qcEdge.cx(spinChain[end], spinChain[start])
        qcEdge.rz(-J[1], spinChain[start])
        qcEdge.cx(spinChain[end], spinChain[start])
        # Return to computational basis
        qcEdge.h(spinChain[start])
        qcEdge.cx(spinChain[start], spinChain[end])
        return qcEdge.to_instruction()

    def spinSpinCircuit(self, spinChain):
        '''
        Function for building circuit that
        performs spin-spin Ham. evolution
        '''
        spinChain = QuantumRegister(len(self.graph.vs))
        qcSpin = QuantumCircuit(spinChain)
        for color in self.graphColors:
            for edge in self.matching[color]:
                qcSpin.append(
                    self.edgeCircuit(edge, spinChain),
                    spinChain
                )
        return qcSpin.decompose().to_instruction()

    def HamiltonianQNN(self, spinChain):
        '''
        Function for building circuit that
        implements parametric QNN
        '''
        qnn = QuantumCircuit(spinChain)
        qnn.append(self.spinSpinCircuit(spinChain), spinChain)
        qnn.append(self.fieldCircuit(spinChain), spinChain)
        return qnn.decompose().to_instruction()

    def evolutionStep(self, dt, spinChain):
        '''
        Function for binding parameters and
        producing time evolution step
        '''
        qcEvolution = QuantumCircuit(spinChain)
        bindingDict = aux.BindParameters(self.graph, dt)
        qcEvolution.append(self.HamiltonianQNN(spinChain), spinChain)
        try:
            return qcEvolution.bind_parameters(bindingDict).to_instruction()
        except exceptions.CircuitError:
            print('An error occurred')
            return qcEvolution.decompose().to_instruction()

    def rawEvolutionCircuit(self, STEPS=200, t=1.7):
        ''''
        Function for retrieving evolution
        circuit without measurement
        '''
        dt = t/STEPS
        spinChain = QuantumRegister(len(self.graph.vs), name='s')
        measureReg = ClassicalRegister(len(self.graph.vs), name='r')
        qcEvolution = QuantumCircuit(spinChain, measureReg)
        qcStep = self.evolutionStep(dt, spinChain)
        for _ in range(STEPS):
            qcEvolution.append(qcStep, spinChain)
        return qcEvolution.decompose()

    def evolutionCircuit(self, STEPS=200, t=1.7):
        '''
        Function for retrieving a simulation 
        circuit with several evolution steps
        and given target time
        '''
        dt = t/STEPS
        spinChain = QuantumRegister(len(self.graph.vs), name='s')
        measureReg = ClassicalRegister(len(self.graph.vs), name='r')
        qcEvolution = QuantumCircuit(spinChain, measureReg)
        qcStep = self.evolutionStep(dt, spinChain)
        if self.withInitialState:
            qcEvolution.initialize(self.initialState)
        for _ in range(STEPS):
            qcEvolution.append(qcStep, spinChain)
        qcEvolution.measure(spinChain, measureReg)
        return qcEvolution.decompose()

################################################################################
##                          GRAPH EVOLUTION ROUTINES                          ##
################################################################################

    # MEASUREMENT ERROR MITIGATION ROUTINES

    def getCalibrationFitter(self):
        '''
        Function for computing error mitigation
        calibration filter for runs on IBM Q
        '''
        print('Generating calibration circuit...')
        spinRegister = QuantumRegister(len(self.graph.vs))
        calibrationCircuits, stateLabels = complete_meas_cal(
            qr=spinRegister,
            circlabel='meas_cal'
        )
        calibrationJob = execute(
            calibrationCircuits,
            backend=self.backend,
            shots=2048
        )
        print('Calibrating measurement with ignis...')
        job_monitor(calibrationJob)
        calibrationResults = calibrationJob.result()
        fitter = CompleteMeasFitter(
            calibrationResults,
            stateLabels,
            circlabel='meas_cal'
        )
        print('Computed measurement correction matrix:\n', fitter.cal_matrix)
        return fitter

    # QUANTUM EVOLUTION ROUTINES

    def quantumTimeEvolution(
            self,
            STEPS=200,
            t=1.7,
            shots=2048,
            **kwargs):
        '''
        Function for QTE. Produces probability
        density at time t
        '''
        timeEvolution = [self.evolutionCircuit(STEPS=STEPS, t=t)]
        job = execute(timeEvolution, self.backend, shots=shots)
        if not self.localSimulation:
            job_monitor(job)
        try:
            measurementFilter = kwargs['measurementFitter'].filter
            return 1/shots * aux.Counts2Pdf(
                len(self.graph.vs),
                job,
                timeEvolution,
                measurementFilter=measurementFilter
            )[0]
        except KeyError:
            job = execute(timeEvolution, self.backend, shots=shots)
            return 1/shots * aux.Counts2Pdf(
                len(self.graph.vs),
                job,
                timeEvolution
            )[0]

    def quantumEvolutionUnitary(self, STEPS=200, t=1.7, shots=2048):
        '''
        Function for QTE. Produces evolution
        unitary at time t
        '''
        timeEvolution = self.rawEvolutionCircuit(STEPS=STEPS, t=t)
        unitarySimulator = Aer.get_backend('unitary_simulator')
        job = execute(timeEvolution, unitarySimulator, shots=shots)
        return job.result().get_unitary()

    # IMPORTANT: Do not use quantumTimeEvolution as a subroutine on
    # evolutionSeries, since this would cause poor performance when executing
    # jobs on IBM Q devices.

    def evolutionSeries(self,
                        STEPS=200,
                        t=1.7,
                        shots=2048,
                        saveToFile=False,
                        **kwargs):
        '''
        Function for evaluation PDF evolution
        under graph Hamiltonian
        '''
        dt = t/STEPS
        evolutionSteps = [self.evolutionCircuit(STEPS=idx, t=dt*idx)
                          for idx in range(1, STEPS+1)]
        job = execute(evolutionSteps, self.backend, shots=shots)
        if not self.localSimulation:
            job_monitor(job)
        numSpins = len(self.graph.vs)
        simulationResults = None
        try:
            measurementFilter = kwargs['measurementFitter'].filter
            simulationResults = 1/shots * aux.Counts2Pdf(
                numSpins,
                job,
                evolutionSteps,
                measurementFilter=measurementFilter
            )
        except KeyError:
            print('In evolutionSeries: Not using measurement error mitigation...')
            simulationResults = 1/shots * aux.Counts2Pdf(
                numSpins,
                job,
                evolutionSteps
            )
        simulationData = np.append(
            np.array(
                [[
                    (idx+1)*dt for idx in range(STEPS)
                ]]
            ).T,
            simulationResults,
            axis=1
        )
        if saveToFile:
            filename = 'evolutionSeries_{}_{}spins.csv'.format(
                self.backendName,
                numSpins
            )
            np.savetxt(
                filename,
                simulationData,
                delimiter=',',
                header='t ' +
                ' '.join([str(idx) for idx in range(2**numSpins)]),
                encoding='utf8'
            )
        return simulationData

    # IMPORTANT: This routine is included here and not in the analyzer for
    # simplicity. It might not be needed on applications.

    def stepsSeries(self,
                    MAX_STEPS=200,
                    t=1.7,
                    shots=2048,
                    saveToFile=False,
                    **kwargs):
        '''
        Function for evaluation PDF v Steps
        under graph Hamiltonian evolution
        '''
        evolutionSteps = [self.evolutionCircuit(STEPS=idx, t=t)
                          for idx in range(1, MAX_STEPS+1)]
        job = execute(evolutionSteps, self.backend, shots=shots)
        if not self.localSimulation:
            job_monitor(job)
        numSpins = len(self.graph.vs)
        simulationResults = None
        try:
            measurementFilter = kwargs['measurementFitter'].filter
            simulationResults = 1/shots * aux.Counts2Pdf(
                numSpins,
                job,
                evolutionSteps,
                measurementFilter=measurementFilter
            )
        except KeyError:
            simulationResults = 1/shots * aux.Counts2Pdf(
                numSpins,
                job,
                evolutionSteps
            )
        simulationData = np.append(
            np.array(
                [[
                    idx for idx in range(1, MAX_STEPS+1)
                ]]
            ).T,
            simulationResults,
            axis=1
        )
        if saveToFile:
            filename = 'stepsSeries_{}_{}spins.csv'.format(
                self.backendName,
                numSpins
            )
            np.savetxt(
                filename,
                simulationData,
                delimiter=',',
                header='t ' +
                ' '.join([str(idx) for idx in range(2**numSpins)]),
                encoding='utf8'
            )
        return simulationData

    # DIAGONALIZATION EVOLUTION ROUTINES

    def exactEvolutionUnitary(self, t=1.7):
        '''
        Function for computing evolution unitary
        by exponentiating Hamiltonian
        '''
        return expm(-1j * t * self.HamiltonianMatrix())

    def exactTimeEvolution(self, t=1.7):
        '''
        Function for performing exact time evolution
        by numpy diagonalization
        '''
        finalState = np.matmul(
            self.exactEvolutionUnitary(t=t),
            self.initialState
        )
        return np.abs(finalState)**2

    def exactEvolutionSeries(self, STEPS=200, t=1.7, saveToFile=False):
        '''
        Function for exact evaluation of PDF
        evolution under graph Hamiltonian
        '''
        dt = t/STEPS
        # Iterate this way to avoid repetitive diagonalization
        evolutionSteps = []
        basicStep = self.exactEvolutionUnitary(t=dt)
        currentStep = basicStep
        for _ in range(STEPS):
            evolutionSteps.append(np.matmul(currentStep, self.initialState))
            currentStep = np.matmul(basicStep, currentStep)
        simulationData = np.append(
            np.array(
                [[
                    (idx+1)*dt for idx in range(STEPS)
                ]]
            ).T,
            np.abs(evolutionSteps)**2,
            axis=1
        )
        numSpins = len(self.graph.vs)
        if saveToFile:
            filename = 'exactEvolutionSeries_{}spins.csv'.format(
                numSpins
            )
            np.savetxt(
                filename,
                simulationData,
                delimiter=',',
                header='t ' +
                ' '.join([str(idx) for idx in range(2**numSpins)]),
                encoding='utf8'
            )
        return simulationData

################################################################################
##                     HAMILTONIAN EXPECTED VALUE ROUTINES                    ##
################################################################################

    def exactHamiltonianExpVal(self):
        '''
        Function for computing the exact expected
        value of the graph Hamiltonian.
        '''
        H = self.HamiltonianMatrix()
        return np.abs(np.inner(
            np.conj(self.initialState).T,
            np.matmul(H, self.initialState)
        ))

    def computePauliProductExpVal(self, QNN, PauliString):
        '''
        Function for computing expected value
        of XX operator on state produced by QNN
        '''
        # create a spinchain
        numSpins = len(self.graph.vs)
        spinChain = QuantumRegister(numSpins)
        # extend QNN
        extendedQNN = QuantumCircuit(spinChain)
        extendedQNN.append(QNN.to_instruction(), spinChain)
        # set up 0th operator
        for idx in range(len(PauliString)):
            if PauliString[idx] == 'X':
                extendedQNN.h(spinChain[idx])
            if PauliString[idx] == 'Y':
                extendedQNN.sdg(spinChain[idx])
                extendedQNN.h(spinChain[idx])
        extendedQNN.draw(output='text')
        try:
            # execute quantum circuit
            totCounts = 2048
            numSpins = len(self.graph.vs)
            stateCircuit = QuantumCircuit(numSpins)
            stateCircuit.append(extendedQNN.to_instruction(), range(numSpins))
            stateCircuit.measure_all()
            job = execute(
                stateCircuit,
                backend=self.backend,
                shots=totCounts
            )
            if not self.localSimulation:
                job_monitor(job)
            # get counts from job
            counts = job.result().get_counts(stateCircuit)
            # compute expected value
            return sum(
                aux.numberOperatorEigenvalue(numSpins, state, PauliString) *
                counts.get(aux.dec2nBitBinaryChain(state, numSpins), 0)
                * 1/totCounts
                for state in range(2**numSpins)
            )
        except exceptions.CircuitError:
            print('Circuit exception ocurred. Check your QNN.')
            return float('NaN')

    def edgeHamiltonianExpVal(self, QNN, edge):
        '''
        Function that computes the expected value
        of the graph Hamiltonian upon application
        of a QNN
        '''
        numSpins = len(self.graph.vs)
        return sum(
            J * self.computePauliProductExpVal(
                QNN,
                aux.twoSpinPauliProductString(
                    numSpins, edge, PauliDouble
                )
            )
            for J, PauliDouble in zip(
                edge['exchangeIntegrals'],
                ['XX', 'YY', 'ZZ']
            )
        )

    def spinSpinHamiltonianExpVal(self, QNN):
        '''
        Function that computes the expected value
        of the graph Hamiltonian upon application
        of a QNN
        '''
        return sum(
            self.edgeHamiltonianExpVal(QNN, edge)
            for edge in self.graph.es
        )

    def vertexHamiltonianExpVal(self, QNN, vertex):
        '''
        Function for computing the expected value
        of the graph Hamiltonian upon application
        of a QNN
        '''
        numSpins = len(self.graph.vs)
        return sum(
            H * self.computePauliProductExpVal(
                QNN,
                aux.spinPauliString(
                    numSpins, vertex, PauliChar
                )
            )
            for H, PauliChar in zip(
                vertex['externalField'],
                ['X', 'Y', 'Z']
            )
        )

    def spinHamiltonianExpVal(self, QNN):
        '''
        Function that computes the expected value
        of the graph Hamiltonian upon application
        of a QNN
        '''
        return sum(
            self.vertexHamiltonianExpVal(QNN, vertex)
            for vertex in self.graph.vs
        )

    def quantumHamiltonianExpVal(self, QNN):
        '''
        Function that computes the expected value
        of the graph Hamiltonian upon application
        of a QNN
        '''
        return self.spinHamiltonianExpVal(QNN) + \
            self.spinSpinHamiltonianExpVal(QNN)


class NaiveSpinGraph(HeisenbergGraph):

    '''
    Class for simulating a generic Heisenberg hamiltonian
    defined on a graph, with Las Heras algorithm
    '''

################################################################################
#           OVERLOAD OF SPIN SPIN CIRCUIT TO MATCH LAS HERAS ET. AL.           #
################################################################################

    def edgeCircuit(self, edge, spinChain):
        '''
        Function for building circuit that
        performs edge Hamiltonian evolution
        '''
        start, end = edge.tuple
        J = edge['paramExchangeIntegrals']
        qcEdge = QuantumCircuit(spinChain)
        # Compute J0 phase
        qcEdge.h(spinChain[start])
        qcEdge.h(spinChain[end])
        qcEdge.cx(spinChain[start], spinChain[end])
        qcEdge.rz(J[0], spinChain[end])
        qcEdge.cx(spinChain[start], spinChain[end])
        qcEdge.h(spinChain[start])
        qcEdge.h(spinChain[end])
        # Compute J1 phase
        qcEdge.sdg(spinChain[start])
        qcEdge.h(spinChain[start])
        qcEdge.sdg(spinChain[end])
        qcEdge.h(spinChain[end])
        qcEdge.cx(spinChain[start], spinChain[end])
        qcEdge.rz(J[1], spinChain[end])
        qcEdge.cx(spinChain[start], spinChain[end])
        qcEdge.h(spinChain[start])
        qcEdge.s(spinChain[start])
        qcEdge.h(spinChain[end])
        qcEdge.s(spinChain[end])
        # Compute J2 phase
        qcEdge.cx(spinChain[start], spinChain[end])
        qcEdge.rz(J[2], spinChain[end])
        qcEdge.cx(spinChain[start], spinChain[end])
        return qcEdge.to_instruction()


class DataAnalyzer:

    '''
    Class for data processing
    '''

    def __init__(self, spinGraph=HeisenbergGraph()):
        self.spinGraph = spinGraph

################################################################################
##                      COMPARATIVE EVOLUTION PLOTS                           ##
################################################################################

    def circuitDepthVSteps(self, MIN_STEPS=1, MAX_STEPS=200):
        '''
        Function for plotting circuit depth
        as function of number of integration
        steps in ST scheme
        '''
        evolutionSteps = [self.spinGraph.evolutionCircuit(STEPS=idx, t=5.456)
                          for idx in range(MIN_STEPS, MAX_STEPS+1)]
        circuitsDepth = np.array(
            [circuit.depth() for circuit in evolutionSteps]
        )
        stepsArray = np.array(
            [idx for idx in range(MIN_STEPS, MAX_STEPS+1)]
        )
        logSteps = np.log(stepsArray)
        logDepth = np.log(circuitsDepth)
        plt.xlabel(r'$ln(N)$')
        plt.ylabel(r'$ln(C_D)$')
        plt.scatter(
            logSteps,
            logDepth
        )
        plt.show()
        # return linear regression data
        # slope, intercept, r-value, p-value, stderr
        return linregress(logSteps, logDepth)


################################################################################
##                      COMPARATIVE EVOLUTION PLOTS                           ##
################################################################################

    def comparativeEvolution(
            self,
            STEPS=200,
            t=3.4,
            saveToFiles=False,
            **kwargs):
        '''
        Function for comparative plotting
        '''
        simulatedData = None
        try:
            simulatedData = self.spinGraph.evolutionSeries(
                STEPS=STEPS,
                t=t,
                saveToFile=saveToFiles,
                measurementFitter=kwargs['measurementFitter']
            )
        except KeyError:
            print('In comparativeEvolution: Not using measurement error mitigation...')
            simulatedData = self.spinGraph.evolutionSeries(
                STEPS=STEPS,
                t=t,
                saveToFile=saveToFiles
            )
        exactData = self.spinGraph.exactEvolutionSeries(
            STEPS=120,
            t=t,
            saveToFile=saveToFiles
        )
        plt.xlabel(r'$t$ (u. a.)')
        plt.ylabel(r'$|\langle \psi_0 | k \rangle|^2$')
        shape = simulatedData.shape
        # plot exactData using solid curves (expected to have more points)
        # and simulatedData with scatter plot
        for idx in range(1, shape[1]):
            plt.plot(
                exactData[:, 0],
                exactData[:, idx],
                '--'
            )
            plt.scatter(
                simulatedData[:, 0],
                simulatedData[:, idx],
                label=str(idx-1)
            )
        plt.legend()
        plt.show()


################################################################################
##                     EVOLUTION OPERATOR ERROR PLOTS                         ##
################################################################################

    def unitaryEvolutionError(self, STEPS=200, t=3.4):
        '''
        Function for computing trace distance between 
        exact and quantum evolution operators 
        '''
        quantumUnitary = self.spinGraph.quantumEvolutionUnitary(
            STEPS=STEPS, t=t
        )
        exactUnitary = self.spinGraph.exactEvolutionUnitary(t=t)
        return np.linalg.norm(quantumUnitary-exactUnitary, ord='fro')

    def unitaryEvolutionErrorVectorized(self,
                                        STEPS=np.array([200]),
                                        times=np.array([3.4])
                                        ):
        '''
        Numpy wrapper for direct vectorization of
        error computation
        '''
        return np.array([
            [
                self.unitaryEvolutionError(
                    STEPS=step,
                    t=time
                )
                for step in STEPS
            ] for time in times
        ])

    def unitaryErrorStepsPlot(self, MAX_STEPS=200, t=3.4):
        '''
        Function for plotting unitary
        error as a function of steps
        for given time
        '''
        steps = np.array([
            step for step in range(1, MAX_STEPS+1)
        ])
        errors = np.array([
            self.unitaryEvolutionError(STEPS=step, t=t)
            for step in steps
        ])
        logErrors = np.log(errors)
        logSteps = np.log(steps)
        # Plot logarithmic scale data
        plt.xlabel(r'$\ln(N)$')
        plt.ylabel(r'$\ln(E)$')
        plt.scatter(logSteps, logErrors)
        plt.show()
        # return linear regression data
        # slope, intercept, r-value, p-value, stderr
        return linregress(logSteps, logErrors)

    def unitaryErrorTimePlot(self, STEPS=200, times=np.array([3.4])):
        '''
        Function for plotting unitary
        error as a function of time
        for given number of steps
        '''
        errors = np.array([
            self.unitaryEvolutionError(STEPS=STEPS, t=time)
            for time in times
        ])
        logErrors = np.log(errors)
        logTimes = np.log(times)
        # Plot logarithmic scale data
        plt.xlabel(r'$\ln(t)$')
        plt.ylabel(r'$\ln(E)$')
        plt.scatter(logErrors, logErrors)
        plt.show()
        # return linear regression data
        # slope, intercept, r-value, p-value, stderr
        return linregress(logTimes, logErrors)

    def unitaryErrorMixedPlot(self,
                              STEPS=np.array([200]),
                              times=np.array([3.4])
                              ):
        '''
        Function for plotting error as function of
        number of steps for given times and viceversa
        '''
        errors = self.unitaryEvolutionErrorVectorized(
            STEPS=STEPS,
            times=times
        )
        numSteps = len(STEPS)
        numTimes = len(times)
        fig, (axSteps, axTimes) = plt.subplots(1, 2)
        # Plot of error vs steps
        axSteps.set_title(r'Log Plot of Error v. No. Steps')
        axSteps.set_ylabel(r'$\ln(E)$')
        axSteps.set_xlabel(r'$\ln(N)$')
        for idx in range(numTimes):
            axSteps.scatter(
                np.log(STEPS),
                np.log(errors[idx, :]),
                label='$t={0:.2f}$'.format(times[idx])
            )
        # axSteps.legend(loc='lower left')
        # Plot of time vs steps
        axTimes.set_title(r'Log Plot of Error v. Time')
        axTimes.set_ylabel(r'$\ln(E)$')
        axTimes.set_xlabel(r'$\ln(t)$')
        for idx in range(numSteps):
            axTimes.scatter(
                np.log(times),
                np.log(errors[:, idx]),
                label='$N={}$'.format(STEPS[idx])
            )
        # axTimes.legend(loc='lower left')
        plt.show()
        return errors

    def unitaryErrorExponents(self,
                              STEPS=np.array([200]),
                              times=np.array([3.4])
                              ):
        '''
        Function for plotting error power law exponents
        '''
        errors = self.unitaryEvolutionErrorVectorized(
            STEPS=STEPS,
            times=times
        )
        numSteps = len(STEPS)
        numTimes = len(times)
        avStepsExponent = []
        for idx in range(numTimes):
            exponent, _, _, _, _ = linregress(
                np.log(STEPS),
                np.log(np.transpose(errors[idx, :]))
            )
            avStepsExponent.append(exponent)
        avTimesExponent = []
        for idx in range(numSteps):
            exponent, _, _, _, _ = linregress(
                np.log(times),
                np.log(errors[:, idx])
            )
            avTimesExponent.append(exponent)
        return avStepsExponent, avTimesExponent

################################################################################
##                     EVOLUTION FIDELITY ERROR PLOTS                         ##
################################################################################

    def pdfFidelities(self, pdf1, pdf2):
        '''
        Function for computing pdf
        fidelities
        '''
        return sum(np.sqrt(pdf1 * pdf2))**2

    def pdfErrorStepsPlot(
            self,
            MAX_STEPS=200,
            t=3.4,
            saveToFile=False,
            **kwargs):
        '''
        Function for plotting pdf error
        as function of steps for given time
        '''
        evolutionSteps = None
        try:
            evolutionSteps = self.spinGraph.stepsSeries(
                MAX_STEPS=MAX_STEPS,
                t=t,
                saveToFile=saveToFile,
                measurementFitter=kwargs['measurementFitter']
            )
        except KeyError:
            evolutionSteps = self.spinGraph.stepsSeries(
                MAX_STEPS=MAX_STEPS,
                t=t,
                saveToFile=saveToFile
            )
        exactPdf = self.spinGraph.exactTimeEvolution(t=t)
        errorArray = np.array([
            1-self.pdfFidelities(evolutionSteps[idx, 1:], exactPdf)
            for idx in range(MAX_STEPS)
        ])
        logSteps = np.log(evolutionSteps[:, 0])
        logErrors = np.log(errorArray)
        plt.xlabel(r'$\ln(N)$')
        plt.ylabel(r'$\ln(1-F_N)$')
        plt.scatter(
            logSteps,
            logErrors
        )
        plt.show()
        return linregress(logSteps, logErrors)

    def pdfErrorStepsData(
            self,
            MAX_STEPS=200,
            t=3.4,
            saveToFile=False,
            **kwargs):
        '''
        Function for plotting pdf error
        as function of steps for given time
        '''
        evolutionSteps = None
        try:
            evolutionSteps = self.spinGraph.stepsSeries(
                MAX_STEPS=MAX_STEPS,
                t=t,
                saveToFile=saveToFile,
                measurementFitter=kwargs['measurementFitter']
            )
        except KeyError:
            evolutionSteps = self.spinGraph.stepsSeries(
                MAX_STEPS=MAX_STEPS,
                t=t,
                saveToFile=saveToFile
            )
        exactPdf = self.spinGraph.exactTimeEvolution(t=t)
        errorArray = np.array([
            1-self.pdfFidelities(evolutionSteps[idx, 1:], exactPdf)
            for idx in range(MAX_STEPS)
        ])
        return evolutionSteps[:, 0], errorArray

    def pdfErrorStepsPlotFit(
            self,
            trialFunction,
            MAX_STEPS=200,
            t=3.4,
            saveToFile=False,
            **kwargs):
        '''
        Function for plotting pdf error
        as function of steps for given time
        and fit to trial function
        '''
        evolutionSteps = None
        try:
            evolutionSteps = self.spinGraph.stepsSeries(
                MAX_STEPS=MAX_STEPS,
                t=t,
                saveToFile=saveToFile,
                measurementFitter=kwargs['measurementFitter']
            )
        except KeyError:
            evolutionSteps = self.spinGraph.stepsSeries(
                MAX_STEPS=MAX_STEPS,
                t=t,
                saveToFile=saveToFile
            )
        exactPdf = self.spinGraph.exactTimeEvolution(t=t)
        errorArray = np.array([
            1-self.pdfFidelities(evolutionSteps[idx, 1:], exactPdf)
            for idx in range(MAX_STEPS)
        ])
        steps = evolutionSteps[:, 0]
        errors = errorArray
        plt.xlabel(r'$N$')
        plt.ylabel(r'$1-F_N$')
        plt.scatter(
            steps,
            errors
        )
        # Fit to trial funcition with scipy
        optParams, cov = curve_fit(
            trialFunction,
            steps,
            errors,
            bounds=(0, 1e5)
        )
        contSteps = np.linspace(1, MAX_STEPS, 1000)
        contErrors = trialFunction(contSteps, *optParams)
        plt.plot(
            contSteps,
            contErrors,
            '--'
        )
        plt.show()
        return optParams, cov
