import time
import numpy as np

from qelvin import QRegister, QCircuit

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
    qft_qelvin(25)
