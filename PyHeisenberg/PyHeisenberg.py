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
import matplotlib.pyplot as plt
from matplotlib import cm
import auxiliaryRoutines as aux
from qiskit.circuit import exceptions
from qiskit import IBMQ, execute, Aer
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

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
        PHI = np.arctan2(Hy, Hx) + 2*np.pi
        THETA = np.arccos(Hz/H)
        LAMBDA = np.pi
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

    # QUANTUM EVOLUTION ROUTINES

    def quantumTimeEvolution(self, STEPS=200, t=1.7, shots=2048):
        '''
        Function for QTE. Produces probability
        density at time t
        '''
        timeEvolution = [self.evolutionCircuit(STEPS=STEPS, t=t)]
        job = execute(timeEvolution, self.backend, shots=shots)
        if not self.localSimulation:
            job_monitor(job)
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

    def evolutionSeries(self, STEPS=200, t=1.7, shots=2048, saveToFile=False):
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
            np.savetext(
                filename,
                simulationData,
                delimiter=',',
                header='t ' +
                ' '.join([str(idx) for idx in range(2**numSpins)]),
                encoding='utf8'
            )
        return simulationData


class DataAnalyzer:

    '''
    Class for data processing
    '''

    def __init__(self, spinGraph=HeisenbergGraph()):
        self.spinGraph = spinGraph

################################################################################
##                      COMPARATIVE EVOLUTION PLOTS                           ##
################################################################################

    def comparativeEvolution(self, STEPS=200, t=3.4, saveToFiles=False):
        '''
        Function for comparative plotting
        '''
        simulatedData = self.spinGraph.evolutionSeries(
            STEPS=STEPS,
            t=t,
            saveToFile=saveToFiles
        )
        exactData = self.spinGraph.exactEvolutionSeries(
            STEPS=STEPS,
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
                exactData[:, idx]
            )
            plt.scatter(
                simulatedData[:, 0],
                simulatedData[:, idx],
                label=str(idx-1)
            )
        plt.legend()
        plt.show()


################################################################################
##                           EVOLUTION ERROR PLOTS                            ##
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
