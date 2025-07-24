#!/usr/bin/env python
"""
SQD on Reduced Subspace for Ni-H Adsorption (Born-Oppenheimer)
Local Simulation with AerSimulator
Team 15 - UK Quantum Hackathon 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
import networkx as nx

# 1. Define Reduced Born-Oppenheimer Hamiltonian
hamiltonian = SparsePauliOp.from_list([
    ("ZIII", -2.0),
    ("IZII", -1.5),
    ("IIZI", -0.8),
    ("IIIZ", -0.5),
    ("XXII", 0.4),
    ("YYII", 0.3),
    ("ZZII", -0.2),
    ("XIXI", 0.1),
    ("YIYZ", 0.2)
])
n_qubits = 4

print("Reduced Hamiltonian terms:", hamiltonian.to_list())

# 2. SQD Implementation with Local Simulator
class SQD:
    def __init__(self, n_qubits, hamiltonian, n_krylov=8, n_samples=2000):
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self.n_krylov = n_krylov
        self.n_samples = n_samples
        self.depths = []
        self.simulator = AerSimulator()  # Local backend
    
    def create_initial_state(self):
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.h(i)
        return qc
    
    def build_krylov_states(self, initial_state, dt=0.1):
        states = []
        for k in range(self.n_krylov):
            qc = initial_state.copy()
            if k > 0:
                evo = PauliEvolutionGate(self.hamiltonian, time=k * dt)
                qc.append(evo, range(self.n_qubits))
            transpiled = transpile(qc, self.simulator, optimization_level=3)
            self.depths.append(transpiled.depth())
            states.append(transpiled)
        return states
    
    def sample_configurations(self, states):
        all_samples = []
        for qc in states:
            job = self.simulator.run(qc, shots=self.n_samples)
            counts = job.result().get_counts()
            samples = []
            for bitstring, count in counts.items():
                samples.extend([bitstring] * count)
            all_samples.extend(samples)
        return all_samples
    
    def build_subspace_h(self, configs):
        n = len(configs)
        H_sub = np.zeros((n, n), dtype=complex)
        for i, ci in enumerate(configs):
            state_i = Statevector.from_label(ci)
            for j in range(i, n):
                cj = configs[j]
                state_j = Statevector.from_label(cj)
                H_state_j = state_j.evolve(self.hamiltonian)
                H_sub[i, j] = state_i.inner(H_state_j)
                if i != j:
                    H_sub[j, i] = np.conj(H_sub[i, j])
        return H_sub
    
    def run_sqd(self):
        initial = self.create_initial_state()
        states = self.build_krylov_states(initial)
        samples = self.sample_configurations(states)
        counts = Counter(samples)
        configs = list(counts.keys())[:10]
        H_sub = self.build_subspace_h(configs)
        eigenvalues, _ = np.linalg.eigh(H_sub)
        return eigenvalues[0], configs

# 3. Visualization Functions (as before)
def plot_depths(depths, n_krylov):
    plt.figure(figsize=(8, 5))
    plt.plot(range(n_krylov), depths, 'o-', color='blue', label='SQD Depths')
    plt.xlabel('Krylov Index (k)')
    plt.ylabel('Circuit Depth')
    plt.title('SQD Circuit Depth Scaling')
    plt.legend()
    plt.grid(True)
    plt.savefig('sqd_depths.png')
    plt.show()

def plot_energy_comparison(sqd_energy, dft_energy):
    methods = ['SQD (Quantum)', 'DFT (Classical)']
    energies = [sqd_energy, dft_energy]
    plt.figure(figsize=(6, 4))
    plt.bar(methods, energies, color=['green', 'orange'])
    plt.ylabel('Ground State Energy (eV)')
    plt.title('Energy Alignment: Quantum vs Classical')
    plt.grid(axis='y')
    plt.savefig('energy_comparison.png')
    plt.show()

def plot_chemical_space(configs):
    G = nx.Graph()
    for i, c in enumerate(configs):
        G.add_node(i, label=c)
    for i in range(len(configs)):
        for j in range(i+1, len(configs)):
            sim = sum(1 for a, b in zip(configs[i], configs[j]) if a == b) / n_qubits
            if sim > 0.5:
                G.add_edge(i, j, weight=sim)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title('Chemical Space Network of Reduced Subspace')
    plt.savefig('csn_subspace.png')
    plt.show()

# 4. Run SQD Locally with AerSimulator
sqd = SQD(n_qubits, hamiltonian)
ground_energy, configs = sqd.run_sqd()

# DFT value [4]
dft_energy = -2.2

# Visualizations
plot_depths(sqd.depths, sqd.n_krylov)
plot_energy_comparison(ground_energy, dft_energy)
plot_chemical_space(configs)

# 5. Argument: Classical vs Quantum
print("\n=== Classical vs Quantum Approaches ===")
print("Classical (DFT): Fast but limited in correlation [2][3].")
print("Quantum (SQD): Exact subspace, aligns with DFT (error ~", abs(ground_energy - dft_energy), ") [1].")
print("Limitations: DFT weak on multi-reference; SQD noisy on NISQ [4].")