################################################################################
##              PROGRAM FOR TESTING VQT ALGORITHM ON PENNYLANE                ##
################################################################################
# AUTHOR: Diego Alejandro Herrera Rojas
# DATE: 14/05/21
# DESCRIPTION: The aim is to test the VQT class defined on VQTwPennyLane

import pennylane as qml
from pennylane import numpy as np
from VQTwPennylane import VQThermalizer, Dec2nbitBin

if __name__ == '__main__':
    np.random.seed(42)
    # initialize a thermalizer
    DemoTherm = VQThermalizer(
        num_spins=2,
        ExchangeIntegrals=[1.0, 7.0, 5.0],
        ExternalField=[4.0, 3.0, 9.0],
        Beta=3.56
    )
    # Initialize Hamiltonian
    DemoTherm.GenHamiltonian()
    # Test QNode
    testnum = 2
    DemoTherm.SetThermalQNode()
    params = 2 * np.pi * np.random.rand(8*6)
    x = DemoTherm.ThermalQNode(params, i=Dec2nbitBin(testnum, 2))
    print(x)
    # Test entropy
    dist = DemoTherm.GenProbDist(10*np.random.rand(DemoTherm.num_spins))
    print(DemoTherm.EnsembleEntropy(dist))
    # Test cost functions
    layers = 5
    params = 2*np.pi * np.random.rand(DemoTherm.num_spins + 6*layers)
    # a, b = DemoTherm.MapParams(params)
    print(DemoTherm.CostFunc(params))
    #  Test Optimizer
    params = DemoTherm.GetOptimalParams()
    print(params)
