from PyHeisenberg import HeisenbergGraph
import numpy as np
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
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
        print('Starting data generation...')
        dts = np.linspace(0.01, 1.7, num=100)
        timeFloquetQuantites = np.array([
            benchmarkGraph.floquetInterestingQuantities(
                dt,
                reps=1000,
                offset=700
            )
            for dt in dts
        ])
        print('Finished data generation...')
        data = np.array([
            dts, timeFloquetQuantites[:, 0], timeFloquetQuantites[:, 1]
        ])
        data = data.T
        np.savetxt(f"../datafiles/sampleData_{numSpins}spins.csv", data, delimiter=',')
        print('Saved data to csv file')
        print('=============================')
