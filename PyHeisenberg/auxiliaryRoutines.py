################################################################################
##                  AUXILIARY ROUTINES FOR TIME SIMULATION                    ##
################################################################################
# AUTHOR: Diego Alejandro Herrera Rojas
# DATE: 11/18/21
# DESCRIPTION: In this program, I define subroutines that will help on
# processing data from simulations to generate plots and so on

import numpy as np
from itertools import zip_longest
from itertools import chain

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


def pauliMatrix(numSpins, vertexId, pauliId):
    '''
    Function for computing pauli
    operator in graph
    '''
    realIdx = numSpins-1-vertexId
    return np.kron(np.identity(2**realIdx),
                   np.kron(PauliMatrices[pauliId],
                           np.identity(2**(numSpins-1-realIdx))
                           )
                   )


def pauliProductMatrix(numSpins, edgeTuple, pauliTuple):
    '''
    Function for computing pauli products
    on spin graph
    '''
    first, last = edgeTuple
    realFirst, realLast = numSpins-1-last, numSpins-1-first
    firstPauli, lastPauli = pauliTuple
    return np.kron(np.identity(2**realFirst),
                   np.kron(PauliMatrices[lastPauli],
                           np.kron(np.identity(2**(realLast-realFirst-1)),
                                   np.kron(PauliMatrices[firstPauli],
                                           np.identity(
                                               2**(numSpins-1-realLast))
                                           )
                                   )
                           )
                   )


def dec2nBitBinaryChain(num, bits):
    '''
    Function that converts an
    n bit number to a binary
    chain of n bits
    '''
    return "{0:b}".format(num).zfill(bits)


def Counts2Pdf(numSpins, Job, Circuits, **kwargs):
    '''
    Routine for extracting the
    PDF produced after execution
    of a list of circuits on a
    NumPy Array
    '''
    # Get Counts
    try:
        measurementFilter = kwargs['measurementFilter']
        filteredJob = measurementFilter.apply(Job.result())
        # print('Using measurement error mitigation...')
        simulationPdf = [
            filteredJob.get_counts(circuit)
            for circuit in Circuits
        ]
        # print('Implemented measurement error mitigation.')
    except KeyError:
        # print('In Counts2Pdf: Not using measurement error mitigation...')
        simulationPdf = [Job.result().get_counts(circuit)
                         for circuit in Circuits]
    # Convert to array of data
    data = np.array([
        [res.get(dec2nBitBinaryChain(num, numSpins), 0)
         for num in range(2**numSpins)]
        for res in simulationPdf
    ])
    return data


def BindParameters(heisenbergGraph, t):
    '''
    Function for binding parameters of
    model to substitute on quantum
    circuit
    '''
    return {
        **dict(zip_longest(
            list(chain.from_iterable(
                heisenbergGraph.es['paramExchangeIntegrals']
            )),
            list(chain.from_iterable(
                [[2*t*Ji for Ji in J]
                 for J in heisenbergGraph.es['exchangeIntegrals']]
            ))
        )),
        **dict(zip_longest(
            heisenbergGraph.vs['paramExternalField'],
            [
                2*t*np.sqrt(sum(Hi**2 for Hi in H))
                for H in heisenbergGraph.vs['externalField']
            ]
        ))
    }


def BindVarParameters(heisenbergGraph, t):
    '''
    Function for binding variational
    parameters for evolution
    '''
    return {
        **dict(zip_longest(
            list(chain.from_iterable(
                heisenbergGraph.es['variationalParams']
            )),
            list(chain.from_iterable(
                heisenbergGraph.es['optimalParams']
            ))
        )),
        **dict(zip_longest(
            heisenbergGraph.vs['paramExternalField'],
            [
                2*t*np.sqrt(sum(Hi**2 for Hi in H))
                for H in heisenbergGraph.vs['externalField']
            ]
        ))
    }


def numberOperatorEigenvalue(bits, state, PauliString):
    '''
    Function for determining eigenvalue
    associated to ZZ operator with respect
    to a given computational basis state
    '''
    value = 1
    stateBinary = dec2nBitBinaryChain(state, bits)
    for (bit, PauliOp) in zip(stateBinary[::-1], PauliString):
        if bit == '1' and PauliOp in ['X', 'Y', 'Z']:
            value = value * (-1)
    return value


def twoSpinPauliProductString(bits, edge, subsystemString):
    '''
    Function for determining the Pauli string
    that corresponds to a particular two
    qubit Pauli product operator
    '''
    pauliString = list('1' * bits)
    i = edge.tuple[0]
    pauliString[i] = subsystemString[0]
    j = edge.tuple[1]
    pauliString[j] = subsystemString[1]
    return ''.join(pauliString)


def spinPauliString(bits, vertex, PauliOpString):
    '''
    Function for determining the Pauli string
    that corresponds to a particular two
    qubit Pauli operator
    '''
    pauliString = list('1' * bits)
    i = vertex.index
    pauliString[i] = PauliOpString
    return ''.join(pauliString)


def multiSpinPauliOpMatrix(PauliOpString):
    '''
    Function for converting a pauli string
    to a matrix with qiskit numeration
    convention
    '''
    P = np.array([1])
    I = np.array([
        [1, 0],
        [0, 1]
    ])
    for op in PauliOpString[::-1]:
        if op == 'X':
            P = np.kron(P, PauliMatrices[0])
            continue
        elif op == 'Y':
            P = np.kron(P, PauliMatrices[1])
            continue
        elif op == 'Z':
            P = np.kron(P, PauliMatrices[2])
            continue
        else:
            P = np.kron(P, I)
    return P


def pauliExpValFromCounts(numSpins, PauliString, counts, shots=2048):
    '''
    Function for computing expected value from
    counts result
    '''
    return sum(
        numberOperatorEigenvalue(numSpins, state, PauliString) *
        counts.get(dec2nBitBinaryChain(state, numSpins), 0)
        * 1/shots
        for state in range(2**numSpins)
    )
