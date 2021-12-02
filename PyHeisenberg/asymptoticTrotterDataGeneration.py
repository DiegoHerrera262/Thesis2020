from PyHeisenberg import HeisenbergGraph, DataAnalyzer
import numpy as np
from scipy.stats import linregress
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    STEPS = np.array([idx for idx in range(200, 232, 2)])
    times = [1, 2, 4, 8, 16, 32]
    for numSpins in [2, 4, 6, 8]:
        print(f'Using {numSpins} spins...')
        benchmarkGraph = HeisenbergGraph(
            spinInteractions={
                (idx, idx+1): [0.5, 1.0, 0.25]
                for idx in range(numSpins-1)
            },
            externalField={
                idx: [1.0, 0.25, 0.5]
                for idx in range(numSpins)
            },
            localSimulation=True,
            backendName='qasm_simulator',
            noisySimulation=False,
            initialState=np.array(
                [1 if idx == 1 else 0 for idx in range(2**numSpins)])
        )
        benchmarkAnalyzer = DataAnalyzer(spinGraph=benchmarkGraph)
        # Computation of asymptotic errors
        errorData = benchmarkAnalyzer.unitaryErrorMixedPlot(
            STEPS=STEPS,
            times=times
        )
        # Data analysis
        # IMP: Rows vary STEPS for fixed t
        # IMP: Cols vary t for fixed STEPS
        stepsLinregress = [
            [
                item
                for item in linregress(
                    np.log(STEPS),
                    np.log(errorData[idx, :])
                )
            ]
            for idx in range(len(times))
        ]
        timesLinregress = [
            [
                item
                for item in linregress(
                    np.log(times),
                    np.log(errorData[:, idx])
                )
            ]
            for idx in range(len(STEPS))
        ]
        # Saving data
        columns = ["slope", "intercept", "r-value", "p-value", "std-err"]
        stepRows = [f"t={t}" for t in times]
        timeRows = [f"N={n}" for n in STEPS]
        # steps dataframe
        stepDf = pd.DataFrame(
            data=stepsLinregress,
            index=stepRows,
            columns=columns
        )
        # times dataframe
        timesDf = pd.DataFrame(
            data=timesLinregress,
            index=timeRows,
            columns=columns
        )
        # saving to csv
        stepDf.to_csv(
            f"../datafiles/TrotterData/unitaryStepsErrorRegression_{numSpins}spins{len(STEPS)}N{len(times)}ts.csv"
        )
        timesDf.to_csv(
            f"../datafiles/TrotterData/unitaryTimesErrorRegression_{numSpins}spins{len(STEPS)}N{len(times)}ts.csv"
        )
