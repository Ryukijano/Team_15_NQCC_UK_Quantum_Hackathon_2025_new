#!/usr/bin/env python
"""
Compare DFT and SQD ground state energies for the 4-site chain
Team 15 - UK Quantum Hackathon 2025
"""

import json
import matplotlib.pyplot as plt

# Load DFT result
def load_dft():
    with open('dft_heisenberg_results.json', 'r') as f:
        dft_data = json.load(f)
    return dft_data['dft_energy_ev']

# Load SQD result
def load_sqd():
    with open('sqd_heisenberg_results.json', 'r') as f:
        sqd_data = json.load(f)
    # SQD eigenvalues are in model units; convert to eV if needed
    # Here, we assume the user wants to see the value * 27.2114
    sqd_energy_ev = sqd_data['eigenvalues'][0] * 27.2114
    return sqd_energy_ev

def main():
    dft_energy = load_dft()
    sqd_energy = load_sqd()
    print(f"DFT ground state energy: {dft_energy:.3f} eV")
    print(f"SQD ground state energy: {sqd_energy:.3f} eV")
    labels = ['DFT', 'SQD']
    energies = [dft_energy, sqd_energy]
    plt.figure(figsize=(6, 5))
    plt.bar(labels, energies, color=['gray', 'blue'])
    plt.ylabel('Ground State Energy (eV)')
    plt.title('DFT vs SQD Ground State Energy Comparison')
    for i, val in enumerate(energies):
        plt.text(i, val, f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=12)
    plt.tight_layout()
    plt.savefig('dft_vs_sqd_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main() 