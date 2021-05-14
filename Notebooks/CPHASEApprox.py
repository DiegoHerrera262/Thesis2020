################################################################################
##           PROGRAM FOR NUMERICAL ESTIMATION OF CPHASE WITH ONE CNOT         ##
################################################################################
## AUTHOR: Diego Alejandro Herrera Rojas
## DATE: 23/04/21
## DESCRIPTION: This program is intended for possible numerical optimization of
##              a CPHASE gate with only one qubit rotations and a single CNOT.

################################################################################
##              IMPORTS NECESSARY TO PERFORM QUANTUM ALGORITHMS               ##
################################################################################
import numpy as np
from scipy.optimize import minimize

################################################################################
##                      DEFINITION OF PAULI MATRICES                          ##
################################################################################
Pauli0 = np.array([
    [1,0],
    [0,1]
])
PauliX = np.array([
    [0,1],
    [1,0]
])
PauliY = np.array([
    [0,-1j],
    [1j,0]
])
PauliZ = np.array([
    [1,0],
    [0,-1]
])

################################################################################
##                    DEFINITION OF SPANING MATRICES                          ##
################################################################################
CNOT = 1/2 * (np.kron(Pauli0,Pauli0) + np.kron(Pauli0,PauliX) + \
              np.kron(PauliZ,Pauli0) - np.kron(PauliZ,PauliX))

def RotMat(p):
    '''
    Function for defining rotation
    matrices of one qubit
    '''
    u = np.array([
        [np.cos(p[0]/2), -np.exp(1j * p[2])*np.sin(p[0]/2)],
        [np.exp(1j*p[1])*np.sin(p[0]/2), np.exp(1j*(p[1]+p[2]))*np.cos(p[0]/2)]
    ])
    return u

def ApproxGate(p):
    '''
    Function for defining approximating
    operator
    '''
    U = [RotMat(p[idx:idx+3]) for idx in range(4)]
    ## Variational operators
    Left = np.kron(U[0],U[1])
    Right = np.kron(U[2],U[3])
    ## Sandwich with CNOT
    return np.matmul(np.matmul(Left,CNOT),Right)

def CPHASE(phi):
    '''
    Function for generating CPHASE GATE
    '''
    return np.cos(phi/2) * np.kron(Pauli0,Pauli0) - \
            1j * np.sin(phi/2) * np.kron(PauliZ,PauliZ)

################################################################################
##                             OPTIMIZATION                                   ##
################################################################################
def TraceNorm(A,B):
    '''
    Function for computing Hilbert
    Schmidt distance
    '''
    C = A - B
    return np.abs(np.trace(np.matmul(np.transpose(C).conjugate(),C)))

def Loss_fn_brute(p,phi):
    '''
    Function for computing loss
    funciton for optimization with
    Scipy
    '''
    return TraceNorm(ApproxGate(p),CPHASE(phi))

def OptimalParams(phi):
    '''
    Optimization of fitting params
    for given phase value
    '''
    loss_fn = lambda x: Loss_fn_brute(x,phi)
    ## Optimize
    x0 = 2*np.pi * np.random.random_sample(size=12)
    res = minimize(loss_fn, x0, method='nelder-mead',\
                  options = {'xatol': 1e-8, 'disp': True})
    return res.x, ApproxGate(res.x)

################################################################################
##                          SIMPLE DEMONSTRATION                              ##
################################################################################
if __name__=='__main__':
    phi = np.pi/3
    ## print(np.abs(CPHASE(phi)-ApproxPhase))
    for _ in range(10):
        p, ApproxPhase = OptimalParams(phi)
        print(TraceNorm(CPHASE(phi),ApproxPhase))
