import numpy as np
import matplotlib.pyplot as plt
import time

import qiskit

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import execute, Aer
from qiskit.visualization import plot_state_city
import qiskit.quantum_info as qi

backend = Aer.get_backend('statevector_simulator')
def qft_qiskit(N):
    psi = QuantumRegister(N)
    circ = QuantumCircuit(psi)


    for j in range(N):
        for k in range(j):
            circ.cp(np.pi/float(2**(j-k)), psi[j], psi[k])
        circ.h(psi[j])


    start = time.time()

    job_sim = execute(circ, backend)
    sim_result = job_sim.result()

    return time.time() - start


if __name__ == '__main__':
    qft_qiskit(25)
