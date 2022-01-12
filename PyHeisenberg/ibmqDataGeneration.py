from PyHeisenberg import HeisenbergGraph
from PyHeisenberg import PulseSpinGraph, DirectSpinGraph
from PyHeisenberg import DataAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')
plt.style.use('FigureStyle.mplstyle')

if __name__ == '__main__':

################################################################################
#                             AUXILIARY FUNCTIONS                              #
################################################################################

    def plotSingleQubitObservables(
        pauliStringsDict,
        spinGraph,
        figName,
        STEPS=8,
        t=np.pi/2,
        **kwargs
    ):
        '''
        Function for plotting X, Y, Z exp vals
        for each qubit in the benchmark
        Hamiltonian
        '''
        pauliStrings = [
            pauliStringsDict['x'],
            pauliStringsDict['y'],
            pauliStringsDict['z'],
        ]
        fig, ax = plt.subplots(1,3)
        colors = ['ro', 'bo', 'ko']
        lines = ['red', 'blue','black']
        titles = [
            r"$\langle\hat{X}_i\rangle$",
            r"$\langle\hat{Y}_i\rangle$",
            r"$\langle\hat{Z}_i\rangle$",
        ]
        flatStrings = [
            pauli
            for obs in pauliStrings
            for pauli in obs
        ]

        timesTheor, pauliTheor = spinGraph.exactPauliExpValSeries(
            flatStrings,
            t=t
        )

        MAX_STEPS = STEPS
        try:
            timesEx, pauliExp = spinGraph.pauliExpValSeries(
                flatStrings,
                MAX_STEPS=MAX_STEPS,
                t=t,
                measurementFitter=kwargs['measurementFitter']
            )
        except KeyError:
            timesEx, pauliExp = spinGraph.pauliExpValSeries(
                flatStrings,
                MAX_STEPS=MAX_STEPS,
                t=t,
            )

        fig.tight_layout(pad=3.0)
        for pauliObs, axis, title in zip(pauliStrings, ax, titles):

            axis.set_aspect(aspect=0.5)

            for pauli, color, line in zip(pauliObs, colors, lines):
                axis.plot(
                    timesTheor, 
                    pauliTheor[pauli], 
                    color=line,
                    linewidth=2,
                    linestyle='dashed')
                axis.plot(timesEx, pauliExp[pauli], color)
            
            axis.set_title(title)
            axis.set_xlabel(r"$t$ (u. a.)")
            axis.set_aspect('equal')
            axis.set_ylim([-1.1, 1.1])

        fig.savefig(figName)

    def plotFidelityStepsSeries(
        analyzer,
        figName,
        STEPS=8,
        times=[np.pi/4, np.pi/2, np.pi],
        reps=5,
        **kwargs
    ):
        '''
        Function for plotting average pdf fidelity
        over steps for fixed times
        '''
        try:
            rawExps = analyzer.pdfErrorStepsSeries(
                MAX_STEPS=STEPS,
                times=times,
                reps=reps,
                measurementFitter=kwargs['measurementFitter']
            )
        except KeyError:
            rawExps = analyzer.pdfErrorStepsSeries(
                MAX_STEPS=STEPS,
                times=times,
                reps=reps
            )
        avFidelity = 1/len(rawExps) * sum(res for res in rawExps)
        stdFidelity = 1/len(rawExps) * sum(
            (res - avFidelity)**2 for res in rawExps
        )
        fig, ax = plt.subplots(1,1)
        ax.set_ylabel(r"$(\sum_{i}\sqrt{\bar{p}_i}p_i)^2$")
        ax.set_xlabel(r"$N$")
        N = [n + 1 for n in range(STEPS)]
        for time, series, std in zip(times, avFidelity, stdFidelity):
            ax.errorbar(
                N, series, 
                yerr = std,
                elinewidth = 2, 
                capsize = 2,
                fmt='o--',
                label = f"$t={time:.2f}$ u. a."
            )
        ax.legend()
        fig.savefig(figName)

################################################################################
#                                GLOBAL PARAMS                                 #
################################################################################

    localSimulation = False
    backendName = 'ibmq_jakarta'
    noisySimulation = False

    STEPS = 8
    t = 1
    times = [1, 2, 3]

################################################################################
#                         DECLARE SPIN GRAPH OBJECTS                           #
################################################################################
    
    qasmGraph = DirectSpinGraph(
        spinInteractions = {
            (0, 1): [1, 1, 1],
            (1, 2): [1, 1, 1],
        },
        externalField = {
            0: [0, 0, 0],
            1: [0, 0, 0],
        },
        localSimulation = True,
        backendName = 'qasm_simulator',
        noisySimulation = noisySimulation,
        initializeList = [1 , 2],
    )
    hGraph = HeisenbergGraph(
        spinInteractions = {
            (0, 1): [1, 1, 1],
            (1, 2): [1, 1, 1],
        },
        externalField = {
            0: [0, 0, 0],
            1: [0, 0, 0],
        },
        localSimulation = localSimulation,
        backendName = backendName,
        noisySimulation = noisySimulation,
        initializeList = [1 , 2],
    )
    dGraph = DirectSpinGraph(
        spinInteractions = {
            (0, 1): [1, 1, 1],
            (1, 2): [1, 1, 1],
        },
        externalField = {
            0: [0, 0, 0],
            1: [0, 0, 0],
        },
        localSimulation = localSimulation,
        backendName = backendName,
        noisySimulation = noisySimulation,
        initializeList = [1 , 2],
    )
    pGraph = PulseSpinGraph(
        spinInteractions={
            (0, 1): [0, 0, 0],
            (1, 2): [0, 0, 0],
            (1, 3): [1, 1, 1],
            (3, 5): [1, 1, 1],
            (4, 5): [0, 0, 0],
            (5, 6): [0, 0, 0],
        },
        externalField={
            0: [0.0, 0.0, 0.0],
            1: [0.0, 0.0, 0.0],
            2: [0.0, 0.0, 0.0],
            3: [0.0, 0.0, 0.0],
            4: [0.0, 0.0, 0.0],
            5: [0.0, 0.0, 0.0],
            6: [0.0, 0.0, 0.0],
        },
        localSimulation = localSimulation,
        backendName = backendName,
        noisySimulation = noisySimulation,
        initializeList = [3 , 5],
    )

################################################################################
#                        DECLARE DATA ANALYZING OBJECTS                        #
################################################################################

    qasmAnalyzer = DataAnalyzer(spinGraph = qasmGraph)
    hAnalyzer = DataAnalyzer(spinGraph = hGraph)
    dAnalyzer = DataAnalyzer(spinGraph = dGraph)
    pAnalyzer = DataAnalyzer(spinGraph = pGraph)

################################################################################
#                           GET MEASUREMENT FITTER                             #
################################################################################

    print("Capture measurement error fitter")
    print("=========================================")
    measurementFitter = hGraph.getCalibrationFitter()
    measurementFitterPulse = pGraph.getCalibrationFitter()

################################################################################
#                          GENERATE PAULI OBS PLOTS                            #
################################################################################

    pauliDict = {
        'x': ['XII', 'IXI', 'IIX'],
        'y': ['YII', 'IYI', 'IIY'],
        'z': ['ZII', 'IZI', 'IIZ'],
    }
    pulseDict = {
        'x': ['IXIIIII', 'IIIXIII', 'IIIIIXI'],
        'y': ['IYIIIII','IIIYIII', 'IIIIIYI'], 
        'z': ['IZIIIII', 'IIIZIII', 'IIIIIZI']
    }
    tstart = time.time()
    print("Starting Pauli Observable Plot Generation")
    print("=========================================")
    plotSingleQubitObservables(
        pauliDict,
        qasmGraph,
        '../images/Benchmark/qasm/qasm_control_pauli_exps.pdf',
        STEPS = STEPS,
        t = t
    )
    print("Finished qasm control plot")
    plotSingleQubitObservables(
        pauliDict,
        hGraph,
        '../images/Benchmark/basis/basis_efficient_pauli_exps.pdf',
        STEPS = STEPS,
        t = t,
        measurementFitter=measurementFitter
    )
    print("Finished basis efficient plot")
    plotSingleQubitObservables(
        pauliDict,
        dGraph,
        '../images/Benchmark/direct/direct_pauli_exps.pdf',
        STEPS = STEPS,
        t = t,
        measurementFitter=measurementFitter 
    )
    print("Finished direct transpilation plot")
    plotSingleQubitObservables(
        pulseDict,
        pGraph,
        '../images/Benchmark/pulse/pulse_efficient_pauli_exps.pdf',
        STEPS = STEPS,
        t = t,
        measurementFitter=measurementFitterPulse   
    )
    print(f"Finished pulse efficient plot")
    print(f"Ellapsed time: {(time.time()-tstart)/60} min")
    print("=========================================")

################################################################################
#                          GENERATE PAULI OBS PLOTS                            #
################################################################################

    tstart = time.time()
    print("Starting Pdf Evolution Plot Generation")
    print("=========================================")
    qasmAnalyzer.comparativeEvolution(
        STEPS=STEPS,
        t=t,
        measurementFitter=measurementFitter,
        figureFile='../images/Benchmark/qasm/qasm_control_pdf_evol.pdf',
        showLegend=True
    )
    print(f"Finished qasm control plot")
    hAnalyzer.comparativeEvolution(
        STEPS=STEPS,
        t=t,
        measurementFitter=measurementFitter,
        figureFile='../images/Benchmark/basis/basis_efficient_pdf_evol.pdf',
        showLegend=True
    )
    print(f"Finished basis efficient plot")
    dAnalyzer.comparativeEvolution(
        STEPS=STEPS,
        t=t,
        measurementFitter=measurementFitter,
        figureFile='../images/Benchmark/direct/direct_pdf_evol.pdf',
        showLegend=True
    )
    print(f"Finished direct transpilation plot")
    pAnalyzer.comparativeEvolution(
        STEPS=STEPS,
        t=t,
        measurementFitter=measurementFitterPulse,
        figureFile='../images/Benchmark/pulse/pulse_efficient_pdf_evol.pdf',
        showLegend=False
    )
    print(f"Finished pulse efficient plot")
    print(f"Ellapsed time: {(time.time()-tstart)/60} min")
    print("=========================================")

################################################################################
#                           GENERATE FIDELITY PLOTS                            #
################################################################################
    
    tstart = time.time()
    print("Starting Average Fidelity Plot Generation")
    print("=========================================")
    plotFidelityStepsSeries(
        qasmAnalyzer,
        '../images/Benchmark/qasm/qasm_control_pdf_fidelity.pdf',
        STEPS = STEPS,
        times = times,
        measurementFitter=measurementFitter
    )
    print("Finished qasm control plot")
    plotFidelityStepsSeries(
        hAnalyzer,
        '../images/Benchmark/basis/basis_efficient_pdf_fidelity.pdf',
        STEPS = STEPS,
        times = times,
        measurementFitter=measurementFitter
    )
    print("Finished basis efficient plot")
    plotFidelityStepsSeries(
        dAnalyzer,
        '../images/Benchmark/direct/direct_pdf_fidelity.pdf',
        STEPS = STEPS,
        times = times,
        measurementFitter=measurementFitter 
    )
    print("Finished direct transpilation plot")
    plotFidelityStepsSeries(
        pAnalyzer,
        '../images/Benchmark/pulse/pulse_efficient_pdf_fidelity.pdf',
        STEPS = STEPS,
        times = times,
        measurementFitter=measurementFitterPulse   
    )
    print(f"Finished pulse efficient plot")
    print(f"Ellapsed time: {(time.time()-tstart)/60} min")
    print("=========================================")
