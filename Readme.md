# Time Simulation of Heisenberg Model in a Quantum Computer

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DiegoHerrera262/Thesis2020/HEAD)

## Abstract

A digital quantum algorithm for simulating Heisenberg-like Hamiltonians is proposed and tested on IBM Quantum Systems. This algorithm is based on a direct trotterization that exploits the commutation properties of the spin-spin interaction. This leads to a reduction in the amount of two qubit gates necessary to perform a single Trotter step. The algorithm is also implemented efficiently on hardware by using a cross-resonance based transpilation. Models with up to three interacting spins are simulated on actual quantum processors, probability density fidelities and some interesting observables are measured and discussed. Applications to solid state physics and universal quantum computation are briefly introduced as possible perspectives of the work.

## Introduction

One of the most promising applications of quantum computing is the simulation of quantum systems. It has been shown that quantum computing models have a natural advantage for this type of problems [[4]](#4). As a matter of fact, previous work has been done on simulation of Heisenberg models and transverse field Ising models with superconducting qubits and microwave pulses [[1]](#1). By means of an elementary fermion-to-qubit mapping of Hilbert spaces, the Hubbard model on a chain has been successfully simulated using digital quantum algorithms [[2]](#2). Other works on quantum time simulation showcase the potential of digital time evolution algorithms for studying physical phenomena ranging from solid state and material science, to high energy physics.

The key of most implementations is to profit hardware-optimized two-qubit gates for simulating a second order trotterization scheme. Albeit useful when a direct control of the microwave pulse schedule is available, this technique is restricted by the current cloud quantum computing backends available to the public by IBM Quantum. Due to the level of specialization and skills required to control a device directly, most publicly available quantum processors only offer a universal set of calibrated gates. The aim of this work is to present a trotterization scheme that is suited for implementation on IBM Quantum backends, that also uses an efficient pulse schedule. This work also aims to generalize the scheme for describing systems modeled by Heisenberg-like Hamiltonians of the shape

$$
\hat{H} = \sum_{\langle i,j \rangle} J_{ij}^{(X)} \hat{X}_i \hat{X}_j + J_{ij}^{(Y)} \hat{Y}_i \hat{Y}_j + J_{ij}^{(Z)} \hat{Z}_i \hat{Z}_j + \sum_i h_i^{(X)} \hat{X}_i + h_i^{(Y)} \hat{Y}_i + h_i^{(Z)} \hat{Z}_i
$$

In general, there is a tradeoff between time evolution discretization and hardware noise. The smaller the time discretization, the larger the number of integration steps, and thus, given the finite coherence time of quantum states, the larger the noise error. As a result, a second order Trotter scheme is used. Such evolution algorithm is benchmarked using both QASM simulators and real quantum backends. The first benchmark is intended for studying the errors associated exclusively to the discretization, and determine the minimum time step required to control this source of error. The former benchmark is used for actual simulation and studying the limitations imposed by the finite coherence time of current quantum backends available through IBM Quantum.

## Justification

Quantum Computing has become a reality now that the field is entering the so called NISQ (Noisy Intermediate Quantum) era. Among many models of quantum computations, the digital quantum circuit model is one of the most extended and discussed. It is based upon the concept of **qubit**, which is an abstract two-level quantum system. By profiting linear superposition and entanglement of may-qubit systems, quantum algorithms are claimed to be more resource-efficient than standard classical algorithms in areas like machine learning and mathematical finance.

Although there is general excitement regarding the possibility of enhancing AI and data science with quantum computing, the present project studies a more fundamental, yet quite versatile, application of quantum computation: simulation of quantum physical systems. This perspective was introduced in 1982 by Richard Feynman. He suggested that using a quantum system to simulate another can reduce the exponential overhead that occurs by incrementing the number of components. In particular, simulation of a Heisenberg Hamiltonian on an arbitrary graph is considered in the context of the quantum circuit model.

Even though this Hamiltonian is far simpler than arbitrary N-qubit systems Hamiltonians, it can be readily generalized to produce interesting behavior. For instance, it is well known that a Heisenberg chain exhibits a quantum phase transition at zero temperature. Moreover, by allowing next-to-nearest-neighbor interactions, a Heisenber Hamiltonian can be used in applications such as Machine Learning based upon Boltzman Machines. In the present work, a parametric Heisenberg Model is simulated, thus opening the possibility of extending the algorithm to the field of Quantum Machine Learning.

In particular, the recently proposed Variational Quantum Thermalizer (VQT) is considered for producing thermal states of a magnetic system. A Quantum Neural Network (QNN) based upon time evolution of a parametric Heisenberg Hamiltonian is used to learn the thermal states of a magnetic system in a 2D system with non-square topology. In consequence, time evolution of an elementary Hamiltonian is used to study thermal properties of a quantum system, thus illustrating the potential of digital quantum computation in material science.

## Objectives

### General Objective

To propose a digital quantum algorithm for simulating a parametric Heisenberg Hamiltonian, and use it to measure quantum observables of a spin magnetic system.

### Specific objectives

- To propose a quantum circuit for simulating a parametric Heisenberg Hamiltonian and fit it for superconducting quantum devices on IBM Quantum.

- To estimate numerically the expected probability density fidelity of time evolution using a Suzuki Trotter scheme of second order.

- To evaluate experimentally the expected probability density fidelity of time evolution using a Suzuki Trotter scheme of second order, on a IBM's superconducting devices.

- To compare expected theoretical and experimental probability density fidelities, so as to characterize the optimal number of integration steps required for obtaining a given fidelity.

## Contents of the Repo

This repository is used to log all my work in the completion of my undergraduate thesis at Universidad Nacional de Colombia - Sede Bogotá. It contains:

- A log of my weekly work in this respect.
- A document with the thesis proposal.
- A folder with the Jupyter notebooks used to simulate algorithms.

## Resources

To reproduce the results of this repository, the system requirements are:

- Python 3.x (Conda/miniconda Distribution)
- Qiskit 0.11.0 (Or newer)

I ran the notebooks locally using `qasm_simulator` from Qiskit.

## References

<a id="1">[1]</a> Slathé Y. et al. (2015). Digital Quantum Simulation of Spin Models with Circuit Quantum Electrodynamics. Physical Review X.
<a id="2">[2]</a> Barends R. et al. (2015). Digital quantum simulation of fermionic models with a superconducting circuit. Nature Communications.
<a id="3">[3]</a> Las Heras U. et al. (2015). Fermionic Models with Superconducting Circuits. (2015). EPJ Quantum Technology
<a id="4">[4]</a> Lloyd S. (1996). Universal Quantum Simulators. Nature.
<a id="5">[5]</a> Feynman R. (1982). Simulating Physics with Computers. International Journal of Theoretical Physics.
<a id="6">[6]</a> Benentia G., Casati G., Strini G. (2004). Principles of Quantum Computation and Information. Volume I: Basic Concepts. World Scientific.
<a id="7">[7]</a> Nielsen M., Chuang I. (2010). Quantum Computation and Information. Cambridge University Press.
