import numpy as np
import matplotlib.pyplot as plt
import time
import random

import qiskit

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import execute, Aer
from qiskit.visualization import plot_state_city
import qiskit.quantum_info as qi

from qelvin import QRegister, QCircuit

backend = Aer.get_backend('statevector_simulator')

N = 11
s = random.randint(0, 2**(N+1) - 1)


qelvin_psi = QRegister(N);
qelvin_circ = QCircuit(qelvin_psi)

start = time.time()

for j in range(N):
    for k in range(j):
        qelvin_circ.cp(np.pi/float(2**(j-k)), j, k)
    qelvin_circ.h(j)

qelvin_circ.run()

outputstate_qelvin = qelvin_circ.state();

print("Elapsed qelvin:\t\t\t{0:.2f}ms".format((time.time()-start)*1000))

qiskit_psi = QuantumRegister(N)
qiskit_circ = QuantumCircuit(qiskit_psi)

for j in range(N):
    for k in range(j):
        qiskit_circ.cp(np.pi/float(2**(j-k)), qiskit_psi[j], qiskit_psi[k])
    qiskit_circ.h(qiskit_psi[j])

start = time.time()
job_sim = execute(qiskit_circ, backend)
sim_result = job_sim.result()

print("Elapsed Qiskit:\t\t\t{0:.2f}ms".format((time.time()-start)*1000))

outputstate_qiskit = sim_result.get_statevector(qiskit_circ, decimals=3)

if N <= 5:
    print("Output state QSim: {}".format(outputstate_qelvin))
    print("Output state Qiskit: {}".format(outputstate_qiskit))
