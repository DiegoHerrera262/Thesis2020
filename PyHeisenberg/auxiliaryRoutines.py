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
    return np.kron(np.identity(2**vertexId),
                   np.kron(PauliMatrices[pauliId],
                           np.identity(2**(numSpins-1-vertexId))
                           )
                   )


def pauliProductMatrix(numSpins, edgeTuple, pauliTuple):
    '''
    Function for computing pauli products
    on spin graph
    '''
    first, last = edgeTuple
    firstPauli, lastPauli = pauliTuple
    return np.kron(np.identity(2**first),
                   np.kron(PauliMatrices[firstPauli],
                           np.kron(np.identity(2**(last-first-1)),
                                   np.kron(PauliMatrices[lastPauli],
                                           np.identity(2**(numSpins-1-last))
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


def Counts2Pdf(numSpins, Job, Circuits):
    '''
    Routine for extracting the
    PDF produced after execution
    of a list of circuits on a
    NumPy Array
    '''
    # Get Counts
    simulationPdf = [Job.result().get_counts(circuit) for circuit in Circuits]
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
