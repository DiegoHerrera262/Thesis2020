from PyHeisenberg import HeisenbergGraph, DataAnalyzer, NaiveSpinGraph
import numpy as np
import matplotlib.pyplot as plt
import warnings
from operator import itemgetter
warnings.filterwarnings('ignore')

numSpins = 10
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
    initialState=np.array([1 if idx == 1 else 0 for idx in range(2**numSpins)])
)
benchmarkAnalyzer = DataAnalyzer(spinGraph=benchmarkGraph)
print('Starting data generation...')
dts = np.linspace(0.01, 0.35, num=50)
timeAverageFidelities = np.array([
    benchmarkGraph.floquetTimeAverageFidelity(dt, reps=9000, offset=4500)
    for dt in dts
])
print('Finished data generation...')
data = np.array([
    dts, timeAverageFidelities
])
data = data.T
np.savetxt("sampleData_12spins.csv", data, delimiter=',')
print('Saved data to csv file')
