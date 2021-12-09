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
from scipy.optimize import curve_fit, minimize
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import cm
import auxiliaryRoutines as aux
import qiskit.quantum_info as qi
from qiskit.circuit import exceptions
from qiskit import IBMQ, execute, Aer
from qiskit.tools.monitor import job_monitor
from qiskit.ignis.verification.tomography import process_tomography_circuits, ProcessTomographyFitter
from qiskit.test.mock import FakeSantiago
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.ignis.mitigation.measurement import complete_meas_cal
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.circuit.library import PhaseEstimation

plt.style.use('FigureStyle.mplstyle')

PauliMatrices = [
    np.array([
        [0, 1],
        [1, 0]
    ]),
    np.array([
        [0, -1j],
        [1j, 0]
    ]),
    np.array([
        [1, 0],
        [0, -1]
    ])
]
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
                provider = IBMQ.get_provider(
                    hub='ibm-q-community',
                    group='ibmquantumawards',
                    project='open-science-22'
                )
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
        return qcVertex

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
        return qcField.decompose()

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
        return qcEdge

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
        return qcSpin.decompose()

    def HamiltonianQNN(self, spinChain):
        '''
        Function for building circuit that
        implements parametric QNN
        '''
        qnn = QuantumCircuit(spinChain)
        qnn.append(self.spinSpinCircuit(spinChain), spinChain)
        qnn.append(self.fieldCircuit(spinChain), spinChain)
        return qnn.decompose()

    def evolutionStep(self, dt, spinChain):
        '''
        Function for binding parameters and
        producing time evolution step
        '''
        qcEvolution = QuantumCircuit(spinChain)
        bindingDict = aux.BindParameters(self.graph, dt)
        qcEvolution.append(self.HamiltonianQNN(spinChain), spinChain)
        try:
            return qcEvolution.bind_parameters(bindingDict)
        except exceptions.CircuitError:
            # print('An error occurred')
            return qcEvolution.decompose()

    def rawEvolutionCircuit(self, STEPS=200, t=1.7):
        ''''
        Function for retrieving evolution
        circuit without measurement
        '''
        dt = t/STEPS
        spinChain = QuantumRegister(len(self.graph.vs), name='s')
        qcEvolution = QuantumCircuit(spinChain)
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
##                            TIME SERIES SCHEDULES                           ##
################################################################################

    def timeSeriesCircuits(self, STEPS=10, t=3.4):
        '''
        Function that returns a list of circuits
        that allow computation of observables
        time series
        '''
        dt = t/STEPS
        spinChain = QuantumRegister(len(self.graph.vs))
        twoSpins = False
        if len(spinChain) == 2:
            sum = 0
            for field in self.graph.vs['externalField']:
                for h in field:
                    sum += np.abs(h)**2
            twoSpins = sum < 1e-3
        circ = []
        for idx in range(1, STEPS+1):
            qc = QuantumCircuit(spinChain)
            qc.initialize(self.initialState)
            qc.append(
                self.rawEvolutionCircuit(
                    STEPS=idx if not twoSpins else 1,
                    t=idx * dt
                ),
                spinChain
            )
            circ.append(qc)
        return circ

    def stepsSeriesCircuits(self, STEPS=10, t=4):
        '''
        Function that returns a list of circuits
        that allow computation of observables
        steps series
        '''
        spinChain = QuantumRegister(len(self.graph.vs))
        circ = []
        for idx in range(1, STEPS+1):
            qc = QuantumCircuit(spinChain)
            qc.initialize(self.initialState)
            qc.append(
                self.rawEvolutionCircuit(
                    STEPS=idx,
                    t=t
                ),
                spinChain
            )
            circ.append(qc)
        return circ

    def rotateToMeasurePauliString(self, QNN, PauliString):
        '''
        Function for appending measurement
        rotation for Pauli string ops.
        '''
        # create a spinchain
        numSpins = len(self.graph.vs)
        spinChain = QuantumRegister(numSpins)
        # extend QNN
        extendedQNN = QuantumCircuit(spinChain)
        extendedQNN.append(QNN, spinChain)
        # set up 0th operator
        for idx in range(len(PauliString)):
            if PauliString[idx] == 'X':
                extendedQNN.h(spinChain[idx])
            if PauliString[idx] == 'Y':
                extendedQNN.sdg(spinChain[idx])
                extendedQNN.h(spinChain[idx])
        return extendedQNN

    def pauliStringTimeSeriesCircuits(
            self,
            PauliString,
            STEPS=10,
            t=3.4):
        '''
        Function for computing time series
        efficiently on current IBM Cloud
        '''
        pauliDict = {}
        pauliSchedule = []
        for pauliString in PauliString:
            stringSchedule = []
            for circuit in self.timeSeriesCircuits(STEPS=STEPS, t=t):
                rotatedCircuit = self.rotateToMeasurePauliString(
                    circuit, pauliString
                )
                rotatedCircuit.measure_all()
                stringSchedule.append(rotatedCircuit)
            pauliDict[pauliString] = stringSchedule
            pauliSchedule.extend(pauliDict[pauliString])
        return pauliDict, pauliSchedule

################################################################################
##                         FLOQUET EVOLUTION ROUTINES                         ##
################################################################################

    def floquetUnitary(self, dt):
        '''
        Function for retrieving Floquet
        unitary under Trotterization scheme
        '''
        spinChain = QuantumRegister(len(self.graph.vs))
        qcFloquet = QuantumCircuit(spinChain)
        qcFloquet.append(
            self.evolutionStep(dt, spinChain),
            spinChain
        )
        job = execute(
            qcFloquet,
            Aer.get_backend('unitary_simulator'),
            optimization_level=0
        )
        return job.result().get_unitary()

    def floquetEigenbasis(self, dt):
        '''
        Function for diagonalizing Floquet
        operator for analysis
        '''
        F = self.floquetUnitary(dt)
        return np.linalg.eig(F)

    def floquetEvolution(self, dt, t):
        '''
        Function for computing time evolution
        with Floquet operator exponentiation
        '''
        F = self.floquetUnitary(dt)
        steps = int(t/dt)
        evOp = np.linalg.matrix_power(F, steps)
        return np.matmul(evOp, self.initialState)

    def floquetTimeAverageFidelity(self, dt, reps=100, offset=20):
        '''
        Function for computing time average
        of fidelity for numTimes til tmax
        for Floquet evolution under dt
        '''
        F = self.floquetUnitary(dt)
        average = sum(
            np.abs(
                np.matmul(
                    np.matmul(
                        self.exactEvolutionUnitary(t=steps*dt),
                        self.initialState
                    ).conj().T,
                    np.matmul(
                        np.linalg.matrix_power(F, steps),
                        self.initialState
                    )
                )
            )**2
            for steps in range(offset, reps+1)
        )
        return 1/(reps + 1 - offset) * average

    def floquetInterestingQuantities(self, dt, reps=100, offset=20):
        '''
        Function for computing interesting
        Floquet dynamics indicators
        - IPR
        - Long time average fidelity
        '''
        F = self.floquetUnitary(dt)
        average = sum(
            np.array([
                np.abs(
                    np.matmul(
                        np.matmul(
                            self.exactEvolutionUnitary(t=steps*dt),
                            self.initialState
                        ).conj().T,
                        np.matmul(
                            np.linalg.matrix_power(F, steps),
                            self.initialState
                        )
                    )
                )**2,
                np.abs(
                    np.matmul(
                        self.initialState.conj().T,
                        np.matmul(
                            np.linalg.matrix_power(F, steps),
                            self.initialState
                        )
                    )
                )**2
            ])
            for steps in range(offset, reps+1)
        )
        return 1/(reps + 1 - offset) * average


################################################################################
##                          GRAPH EVOLUTION ROUTINES                          ##
################################################################################

    def execute(self, circuits, backend, shots=2048):
        '''
        Function for executing the
        experiments according to
        backend
        '''
        return execute(circuits, backend, shots=shots)

    # MEASUREMENT ERROR MITIGATION ROUTINES

    def getCalibrationFitter(self):
        '''
        Function for computing error mitigation
        calibration filter for runs on IBM Q
        '''
        # print('Generating calibration circuit...')
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
        # print('Calibrating measurement with ignis...')
        job_monitor(calibrationJob)
        calibrationResults = calibrationJob.result()
        fitter = CompleteMeasFitter(
            calibrationResults,
            stateLabels,
            circlabel='meas_cal'
        )
        # print('Computed measurement correction matrix:\n', fitter.cal_matrix)
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
        job = self.execute(timeEvolution, self.backend, shots=shots)
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
        job = self.execute(evolutionSteps, self.backend, shots=shots)
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
            # print('In evolutionSeries: Not using measurement error mitigation...')
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
        job = self.execute(evolutionSteps, self.backend, shots=shots)
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

    def pauliExpValSeries(self,
                          PauliString,
                          MAX_STEPS=200,
                          t=1.7,
                          shots=2048,
                          **kwargs):
        '''
        Function for computing expected value
        of Pauli strings operators
        '''
        dt = t/MAX_STEPS
        times = [dt * idx for idx in range(1, MAX_STEPS+1)]
        pauliDict, pauliSchedule = self.pauliStringTimeSeriesCircuits(
            PauliString,
            STEPS=MAX_STEPS,
            t=t
        )
        job = self.execute(pauliSchedule, self.backend, shots=shots)
        if not self.localSimulation:
            job_monitor(job)
        # Filter result if error correction is true
        try:
            measurementFilter = kwargs['measurementFitter'].filter
            result = measurementFilter.apply(
                job.result()
            )
        except KeyError:
            result = job.result()
        resultSeries = {}
        numSpins = len(self.graph.vs)
        for pauliString in PauliString:
            resultSeries[pauliString] = np.array([
                aux.pauliExpValFromCounts(
                    numSpins,
                    pauliString,
                    result.get_counts(circuit),
                    shots=shots
                )
                for circuit in pauliDict[pauliString]
            ])
        return times, resultSeries

    def exactPauliExpValSeries(self, PauliStrings, STEPS=200, t=3.4):
        '''
        Function for computing the exact
        expected value of a series of Pauli strings
        '''
        dt = t/STEPS
        times = np.array([dt * idx for idx in range(1, STEPS+1)])
        resultSeries = {
            pauliString: np.array([
                self.exactPauliExpectedValue(
                    pauliString,
                    t=time
                )
                for time in times
            ])
            for pauliString in PauliStrings
        }
        return times, resultSeries

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

    def exactPauliExpectedValue(self, PauliString, t=1.7):
        '''
        Function for computing the 
        exact expected value of
        Pauli operator
        '''
        PauliOp = aux.multiSpinPauliOpMatrix(PauliString)
        finalState = np.matmul(
            self.exactEvolutionUnitary(t=t),
            self.initialState
        )
        return np.inner(
            np.conj(finalState).T,
            np.matmul(PauliOp, finalState)
        ).real

    def exactHamiltonianExpVal(self):
        '''
        Function for computing the exact expected
        value of the graph Hamiltonian.
        '''
        H = self.HamiltonianMatrix()
        return np.inner(
            np.conj(self.initialState).T,
            np.matmul(H, self.initialState)
        ).real

    def computePauliProductExpVal(self, QNN, PauliString, **kwargs):
        '''
        Function for computing expected value
        of XX operator on state produced by QNN
        '''
        # create a spinchain
        numSpins = len(self.graph.vs)
        spinChain = QuantumRegister(numSpins)
        # extend QNN
        extendedQNN = QuantumCircuit(spinChain)
        extendedQNN.append(QNN, spinChain)
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
            job = self.execute(
                stateCircuit,
                backend=self.backend,
                shots=totCounts
            )
            if not self.localSimulation:
                job_monitor(job)
            # get counts from job
            try:
                measurementFilter = kwargs['measurementFitter'].filter
                counts = measurementFilter.apply(
                    job.result()
                ).get_counts(stateCircuit)
            except KeyError:
                counts = job.result().get_counts(stateCircuit)
            # compute expected value
            return sum(
                aux.numberOperatorEigenvalue(numSpins, state, PauliString) *
                counts.get(aux.dec2nBitBinaryChain(state, numSpins), 0)
                * 1/totCounts
                for state in range(2**numSpins)
            )
        except exceptions.CircuitError:
            # print('Circuit exception ocurred. Check your QNN.')
            return float('NaN')

    def edgeHamiltonianExpVal(self, QNN, edge, **kwargs):
        '''
        Function that computes the expected value
        of the graph Hamiltonian upon application
        of a QNN
        '''
        numSpins = len(self.graph.vs)
        try:
            return sum(
                J * self.computePauliProductExpVal(
                    QNN,
                    aux.twoSpinPauliProductString(
                        numSpins, edge, PauliDouble
                    ),
                    measurementFitter=kwargs['measurementFitter']
                )
                for J, PauliDouble in zip(
                    edge['exchangeIntegrals'],
                    ['XX', 'YY', 'ZZ']
                )
            )
        except KeyError:
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

################################################################################
##                           ANNIHILATION ROUTINES                            ##
################################################################################

    def groundStateEnergy(self):
        '''
        Function for getting
        the GS energy
        '''
        H = self.HamiltonianMatrix()
        eigs, _ = np.linalg.eig(H)
        return np.min(eigs)

    def rawAnnealingCircuit(self, Period=1, STEPS=13):
        '''
        Function that simulates annealing
        using a shallow network
        '''
        spinChain = QuantumRegister(len(self.graph.vs), name='s')
        qcAnnealing = QuantumCircuit(spinChain)
        # Initializing GS of mixer Hamiltonian: sum(X)
        dt = Period / STEPS
        times = np.arange(dt, Period, step=dt)
        # qcAnnealing.h(spinChain)
        for time in times:
            beta = 1 - time / Period
            gamma = 1 - beta
            # get hamiltonian evol. instruction
            qcAnnealing.append(
                self.evolutionStep(gamma*dt, spinChain),
                spinChain
            )
            # get mixer evolution
            qcAnnealing.rz(-2*beta*dt, spinChain)
        return qcAnnealing

    def annihilationEnergy(self, Period=1, STEPS=13):
        '''
        Function that returns the
        energy of graph Hamiltonian GS
        '''
        return self.quantumHamiltonianExpVal(
            self.rawAnnealingCircuit(Period=Period, STEPS=STEPS)
        )

################################################################################
##                           QPE ENERGY ESTIMATION                            ##
################################################################################

    def energySpectrum(self, evalQubits=1):
        '''
        FuNction for computing the
        energy spectrum of the
        Hamiltonian using QPE
        '''
        pointerRegister = QuantumRegister(evalQubits, name="eval")
        stateRegister = QuantumRegister(len(self.graph.vs), name="s")
        energyRegister = ClassicalRegister(evalQubits, name="e")
        qcQpe = QuantumCircuit(pointerRegister, stateRegister, energyRegister)
        # unitary for qpe
        qcUnitary = QuantumCircuit(stateRegister)
        STEPS = 21
        t = 2*np.pi
        dt = t/STEPS
        for _ in range(STEPS):
            qcUnitary.append(
                self.evolutionStep(dt, stateRegister),
                stateRegister
            )
        qpe = PhaseEstimation(
            evalQubits,
            qcUnitary.decompose()
        )
        qcQpe.append(qpe, [*pointerRegister, *stateRegister])
        qcQpe.measure(pointerRegister, energyRegister)
        job = self.execute(qcQpe, self.backend, shots=2048)
        return job


class PulseSpinGraph(HeisenbergGraph):

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


################################################################################
##         DEFINITION OF FUNDAMENTAL QUANTUM CIRCUIT FOR SIMULATION           ##
################################################################################


    def edgeCircuit(self, edge, spinChain):
        '''
        Function for building circuit that
        performs edge Hamiltonian evolution
        '''
        start, end = edge.tuple
        gamma = edge['variationalParams']
        qcEdge = QuantumCircuit(spinChain)
        # append first two u3 gates
        qcEdge.u3(*gamma[0:3], spinChain[start])
        qcEdge.u3(*gamma[3:6], spinChain[end])
        # Include CNOT
        qcEdge.cx(spinChain[start], spinChain[end])
        # append middle two u3 gates
        qcEdge.u3(*gamma[6:9], spinChain[start])
        qcEdge.u3(*gamma[9:12], spinChain[end])
        # Include CNOT
        qcEdge.cx(spinChain[start], spinChain[end])
        # append last two u3 gates
        qcEdge.u3(*gamma[12:15], spinChain[start])
        qcEdge.u3(*gamma[15:18], spinChain[end])
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
            # print('In comparativeEvolution: Not using measurement error mitigation...')
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
            ax = plt.gca()
            ax.set_aspect(2)
            ax.set_ylim([-0.1, 1.1])
        plt.legend()
        plt.show()

    def comparativeExactPauliExpEvolution(
            self,
            PauliString,
            STEPS=200,
            t=3.4):
        '''
        Function for plotting comarative evolution
        of pauli string op with direc diagonalization
        '''
        dt = t/STEPS
        times = np.array([dt*idx for idx in range(1, STEPS+1)])
        expVals = np.array([self.spinGraph.exactPauliExpectedValue(
            PauliString,
            t=time
        ) for time in times])
        return times, expVals

    def comparativePauliExpEvolution(
            self,
            PauliString,
            STEPS=200,
            t=3.4,
            **kwargs):
        '''
        Function for plotting comparative evolution of
        pauli string expected value
        '''
        dt = t/STEPS
        times = np.array([dt*idx for idx in range(1, STEPS+1)])
        twoSpins = len(self.spinGraph.graph.vs) == 2
        # Evaluate if there is no external field
        sum = 0
        for field in self.spinGraph.graph.vs['externalField']:
            for h in field:
                sum += np.abs(h)**2
        evolutionSteps = []
        for idx, time in enumerate(times):
            qc = QuantumCircuit(len(self.spinGraph.graph.vs))
            qc.initialize(self.spinGraph.initialState)
            qc.append(
                self.spinGraph.rawEvolutionCircuit(
                    STEPS=idx+1 if sum > 1e-3 and twoSpins else 1,
                    t=time
                ),
                range(len(self.spinGraph.graph.vs))
            )
            evolutionSteps.append(qc)
        expVals = np.zeros(len(evolutionSteps))
        for idx, circuit in enumerate(evolutionSteps):
            try:
                expVals[idx] = self.spinGraph.computePauliProductExpVal(
                    circuit,
                    PauliString,
                    measurementFitter=kwargs['measurementFitter']
                )
            except KeyError:
                expVals[idx] = self.spinGraph.computePauliProductExpVal(
                    circuit,
                    PauliString
                )
        return times, expVals


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
                              times=np.array([3.4]),
                              saveToFile=False
                              ):
        '''
        Function for plotting error as function of
        number of steps for given times and viceversa
        '''
        print("Started data generation...")
        errors = self.unitaryEvolutionErrorVectorized(
            STEPS=STEPS,
            times=times
        )
        print("Finished data generation...")
        numSteps = len(STEPS)
        numTimes = len(times)
        print("Generating plot")
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
        # axTimes.set_ylabel(r'$\ln(E)$')
        axTimes.set_xlabel(r'$\ln(t)$')
        for idx in range(numSteps):
            axTimes.scatter(
                np.log(times),
                np.log(errors[:, idx]),
                label='$N={}$'.format(STEPS[idx])
            )
        # axTimes.legend(loc='lower left')
        fig.savefig(
            f"../images/TrotterErrorPlots/unitaryErrorPlot_{numSteps}N{numTimes}ts.pdf"
        )
        if saveToFile:
            np.savetxt(
                f"../datafiles/TrotterData/unitaryErrorData_{numSteps}N{numTimes}ts.csv",
                errors,
                delimiter=','
            )
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
##                   PROCESS TOMOGRAPHY FIDELITY PLOTS                        ##
################################################################################

    def processTomographyInfidelity(self, STEPS=200, t=3.4):
        '''
        Function for computing process fidelity
        of trotterization using SQPT
        '''
        # Set up QPT
        spinChain = QuantumRegister(len(self.spinGraph.graph.vs))
        qcFidelity = QuantumCircuit(spinChain)
        qcFidelity.append(
            self.spinGraph.rawEvolutionCircuit(STEPS=STEPS, t=t),
            spinChain
        )
        qpt_circs = process_tomography_circuits(qcFidelity, spinChain)
        job = execute(qpt_circs, self.spinGraph.backend, shots=2048)
        tomography = ProcessTomographyFitter(
            job.result(),
            qpt_circs
        )
        # Fit process Chi matrix
        chi_fit = tomography.fit(method='lstsq')
        # Get target process
        targetProcess = self.spinGraph.exactEvolutionUnitary(t=t)
        return 1 - qi.average_gate_fidelity(
            chi_fit,
            target=targetProcess,
        )

    def processTomographyData(self, STEPS=[200], times=[3.4]):
        '''
        Function for generating average
        fidelity data
        '''
        return np.array([
            [
                self.processTomographyInfidelity(STEPS=N, t=t)
                for N in STEPS
            ]
            for t in times
        ])

################################################################################
##                     EVOLUTION FIDELITY ERROR PLOTS                         ##
################################################################################

    def pdfErrorCircuits(
            self,
            MAX_STEPS=200,
            times=[3.4],
            reps=50,
            randomInitState=False):
        '''
        Function for retrieving an experiment
        scheme and circuits for assessing fidelity
        of state evolution
        '''
        experimentScheme = []
        experimentCircuits = []
        numSpins = len(self.spinGraph.graph.vs)
        for _ in range(reps):
            repetition = []
            for time in times:
                if randomInitState:
                    rState = (0.5 - np.random.rand(2**numSpins)) + \
                        1j * (0.5 - np.random.rand(2**numSpins))
                    norm = np.linalg.norm(rState)
                    self.spinGraph.initialState = 1/norm * rState
                scheme = self.spinGraph.stepsSeriesCircuits(
                    STEPS=MAX_STEPS,
                    t=time
                )
                for idx, circuit in enumerate(scheme):
                    circuit.measure_all()
                    scheme[idx] = circuit
                experimentScheme.extend(scheme)
                repetition.append(scheme)
            experimentCircuits.append(repetition)
        return experimentCircuits, experimentScheme

    def pdfFidelities(self, pdf1, pdf2):
        '''
        Function for computing pdf
        fidelities
        '''
        return sum(np.sqrt(pdf1 * pdf2))**2

    def countFidelity(self, counts, targetPdf, shots=2048):
        '''
        Function for computing pdf
        fidelity from counts
        '''
        numSpins = len(self.spinGraph.graph.vs)
        data = 1/shots * np.array([
            counts.get(aux.dec2nBitBinaryChain(num, numSpins), 0)
            for num in range(2**numSpins)
        ])
        return self.pdfFidelities(data, targetPdf)

    def pdfErrorStepsPlot(
            self,
            MAX_STEPS=200,
            times=[3.4],
            reps=50,
            saveToFile=False,
            **kwargs):
        '''
        Function for plotting pdf error
        as function of steps for given time
        '''
        repetitions, experiment = self.pdfErrorCircuits(
            MAX_STEPS=MAX_STEPS,
            times=times,
            reps=reps
        )
        job = execute(experiment, self.spinGraph.backend, shots=2048)
        if not self.spinGraph.localSimulation:
            job_monitor(job)
        try:
            measurementFilter = kwargs['measurementFitter'].filter
            result = measurementFilter.apply(job.result())
        except KeyError:
            result = job.result()
        data = np.array([
            [
                [
                    self.countFidelity(
                        result.get_counts(circuit),
                        self.spinGraph.exactTimeEvolution(t=times[idx])
                    ) for circuit in timeSchedule
                ]
                for idx, timeSchedule in enumerate(repetition)
            ]
            for repetition in repetitions
        ])
        return data

        # evolutionSteps = None
        # try:
        #     evolutionSteps = self.spinGraph.stepsSeries(
        #         MAX_STEPS=MAX_STEPS,
        #         t=t,
        #         saveToFile=saveToFile,
        #         measurementFitter=kwargs['measurementFitter']
        #     )
        # except KeyError:
        #     evolutionSteps = self.spinGraph.stepsSeries(
        #         MAX_STEPS=MAX_STEPS,
        #         t=t,
        #         saveToFile=saveToFile
        #     )
        # exactPdf = self.spinGraph.exactTimeEvolution(t=t)
        # errorArray = np.array([
        #     1-self.pdfFidelities(evolutionSteps[idx, 1:], exactPdf)
        #     for idx in range(MAX_STEPS)
        # ])
        # logSteps = np.log(evolutionSteps[:, 0])
        # logErrors = np.log(errorArray)
        # plt.xlabel(r'$\ln(N)$')
        # plt.ylabel(r'$\ln(1-F_N)$')
        # plt.scatter(
        #     logSteps,
        #     logErrors
        # )
        # plt.show()
        # return linregress(logSteps, logErrors)

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

################################################################################
##                        ANNIHILATION ENERGY PLOTS                           ##
################################################################################

    def annealingProcessGraph(self, maxPeriod=13, periodStep=1, STEPS=250):
        '''
        Function for plotting 
        annihilation convergence
        '''
        periods = np.array([s for s in range(1, maxPeriod+1, periodStep)])
        energies = np.array([self.spinGraph.annihilationEnergy(
            Period=s,
            STEPS=STEPS
        ) for s in periods])
        # get exact minimum energy
        H = self.spinGraph.HamiltonianMatrix()
        eigs, _ = np.linalg.eig(H)
        Emin = round(np.min(eigs).real, 2)
        plt.xlabel(r'$T$')
        plt.ylabel(r'$E_0$')
        plt.scatter(periods, energies)
        plt.hlines(Emin, 0, maxPeriod+1,
                   colors=['r'], label=f"$E = {Emin} u. a.$")
        plt.legend()
        plt.show()
        return periods, energies
