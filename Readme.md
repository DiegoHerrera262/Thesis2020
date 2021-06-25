# Time Simulation of Heisenberg Model in a Quantum Computer

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DiegoHerrera262/Thesis2020/HEAD)

## Summary

This project proposes a digital quantum algorithm for simulating time evolution of a completely anisotropic Heisenberg Hamiltonian on an arbitrary lattice. Experiments were performed on IBM Q devices, and fidelities grater than 80% were obtained for spin chains with up to 4 lattice sites. Moreover, the gate count is significantly improved over similar algorithms, and by the nature of the gates used, transpilation is quite cost-effective. As an interesting application of quantum time evolution, the quantum algorithm is used as a quantum node for producing thermal states of a magnetic lattice describe by a Heisenberg-like model.

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

- To implement a VQT algorithm, with a QNN based upon time evolution of a parametric Heisenberg Hamiltonian.

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
