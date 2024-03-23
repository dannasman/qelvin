import numpy as np
import matplotlib.pyplot as plt
import time

import qiskit

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import execute, Aer
from qiskit.visualization import plot_state_city
import qiskit.quantum_info as qi

from qelvin import QRegister, QCircuit

from memory_profiler import memory_usage

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

def qft_qelvin(N):
    psi = QRegister(N)
    circ = QCircuit(psi)

    for j in range(N):
        for k in range(j):
            circ.cp(np.pi/float(2**(j-k)), j, k)
        circ.h(j)

    start = time.time()

    circ.run()

    return time.time() - start

if __name__ == '__main__':
    t_qiskit = np.array([])
    t_qelvin = np.array([])

    mem_qiskit = np.array([])
    mem_qelvin = np.array([])

    print("Starting execution time measurements...")

    for N in range(1, 26):
        t_qiskit = np.append(t_qiskit, qft_qiskit(N))
        t_qelvin = np.append(t_qelvin, qft_qelvin(N))

    print("Starting memory usage measurements...")

    for N in range(1, 26):
        mqiskit = memory_usage((qft_qiskit, (N, )))
        mqelvin = memory_usage((qft_qelvin, (N, )))
        mem_qiskit = np.append(mem_qiskit, np.average(mqiskit))
        mem_qelvin = np.append(mem_qelvin, np.average(mqelvin))

    plt.figure()

    plt.plot(np.arange(1, 26), t_qiskit, label="qiskit")
    plt.plot(np.arange(1, 26), t_qelvin, label="qelvin")

    plt.xlabel("number of qubits", fontsize=25)
    plt.ylabel("execution time (s)", fontsize=25)
    plt.tick_params(labelsize=25)
    plt.tick_params(labelsize=25)

    plt.yscale("log")
    plt.legend(fontsize=25)
    plt.grid()
    plt.axis("tight")

    plt.figure()

    plt.plot(np.arange(1, 26), mem_qiskit, label="qiskit")
    plt.plot(np.arange(1, 26), mem_qelvin, label="qelvin")
    
    plt.xlabel("number of qubits", fontsize=25)
    plt.ylabel("memory usage (MB)", fontsize=25)
    plt.tick_params(labelsize=25)
    plt.tick_params(labelsize=25)

    plt.legend(fontsize=25)
    plt.grid()
    plt.axis("tight")
    
    plt.show()
