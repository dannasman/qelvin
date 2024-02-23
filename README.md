# Qelvin

## Set up the environment

Retrieve the code and enter the project directory by running:
```
git clone https://github.com/dannasman/qelvin.git && cd qelvin
```
Create and activate a new python `virtualenv`:
```
python -m venv .env
source .env/bin/activate
```
Install all the requirements in `requirements.txt`:
```
pip install --requirement requirements.txt
```
To compile the code use `maturin develop`. At the moment, if you are not using a `x86`/`x86_64` machine with `FMA` support the code won't run properly.
```
RUSTFLAGS="-C target-feature=+avx2,+fma -C target-cpu=native" maturin develop --release
```
You should now be able use `import qelvin` in Python files inside the directory.

## Example code

Below is a Quantum Fourier transform of a 4-qubit system.

```python
import numpy as np
from qelvin import QRegister, QCircuit

N = 4

psi_qelvin = QRegister(N)
circ_qelvin = QCircuit(psi_qelvin)

print("State before QFT:")
print(circ_qelvin.state())

for j in range(N):
    for k in range(j):
        circ_qelvin.cp(np.pi/float(2**(j-k)), j, k)
    circ_qelvin.h(j)

print(circ_qelvin)

circ_qelvin.run()

outputstate_qelvin = circ_qelvin.state()

print("State after QFT:")
print(outputstate_qelvin)
```
Output:
```
State before QFT:
[ 1.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j  0.000+0.000*j ]
      -------  -------           -------                    -------                             
     |       ||       |         |       |                  |       |                            
q0---|   H   ||P(1.57)|---------|P(0.79)|------------------|P(0.39)|-------------------------------
     |       ||       |         |       |                  |       |                            
      -------  -------           -------                    -------                             
                  |     -------     |     -------              |     -------                    
                  |    |       |    |    |       |             |    |       |                   
q1----------------o----|   H   |----|----|P(1.57)|-------------|----|P(0.79)|----------------------
                       |       |    |    |       |             |    |       |                   
                        -------     |     -------              |     -------                    
                                    |        |     -------     |        |     -------           
                                    |        |    |       |    |        |    |       |          
q2----------------------------------o--------o----|   H   |----|--------|----|P(1.57)|-------------
                                                  |       |    |        |    |       |          
                                                   -------     |        |     -------           
                                                               |        |        |     -------  
                                                               |        |        |    |       | 
q3-------------------------------------------------------------o--------o--------o----|   H   |----
                                                                                      |       | 
                                                                                       -------  

State after QFT:
[ 0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j  0.250+0.000*j ]
```
