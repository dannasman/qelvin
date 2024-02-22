use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod circuit;
mod visual;

use crate::circuit::*;
use crate::visual::*;

#[pyclass]
struct QRegisterIter {
    inner: std::vec::IntoIter<(f64, f64)>,
}

#[pymethods]
impl QRegisterIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(f64, f64)> {
        slf.inner.next()
    }
}

#[pyclass(name = "QRegister")]
struct PyQRegister {
    inner: QRegister,
    nqubits: usize,
}

#[pymethods]
impl PyQRegister {
    #[new]
    #[pyo3(signature = (nqubits, init_state = 0))]
    fn new(nqubits: usize, init_state: usize) -> PyResult<Self> {
        let n: usize = 1 << nqubits;
        if init_state >= n {
            return Err(PyValueError::new_err("state binary given as initial sate greater than any possible binary state of the register"));
        }

        let mut states: Vec<Complex> = vec![Complex::new(0.0, 0.0); n];
        states[init_state].real = 1.0;
        Ok(Self {
            inner: QRegister::new(states),
            nqubits,
        })
    }

    fn __getitem__(slf: PyRef<'_, Self>, index: usize) -> PyResult<(f64, f64)> {
        if index >= (1 << slf.nqubits) {
            return Err(PyValueError::new_err("index out of bounds"));
        }
        let state: Complex = slf.inner[index];
        Ok((state.real, state.imag))
    }

    fn __setitem__(mut slf: PyRefMut<'_, Self>, index: usize, value: (f64, f64)) -> PyResult<()> {
        if index >= (1 << slf.nqubits) {
            return Err(PyValueError::new_err("index out of bounds"));
        }
        let (real, imag): (f64, f64) = value;
        slf.inner[index].real = real;
        slf.inner[index].imag = imag;
        Ok(())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<QRegisterIter>> {
        let iter = QRegisterIter {
            inner: slf
                .inner
                .iter()
                .map(|c| (c.real, c.imag))
                .collect::<Vec<(f64, f64)>>()
                .into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    fn __repr__(slf: PyRef<'_, Self>) -> String {
        slf.inner.to_string()
    }
}

#[pyclass(name = "QCircuit")]
struct PyQCircuit {
    inner: QCircuit,
    nqubits: usize,
}

#[pymethods]
impl PyQCircuit {
    #[new]
    fn new(register: &PyQRegister) -> PyResult<Self> {
        Ok(Self {
            inner: QCircuit::new(register.inner.to_owned(), std::collections::VecDeque::new()),
            nqubits: register.nqubits,
        })
    }

    fn h(mut slf: PyRefMut<'_, Self>, target: usize) {
        slf.inner.hadamard(target);
    }

    fn x(mut slf: PyRefMut<'_, Self>, target: usize) {
        slf.inner.pauli_x(target);
    }

    fn y(mut slf: PyRefMut<'_, Self>, target: usize) {
        slf.inner.pauli_y(target);
    }

    fn z(mut slf: PyRefMut<'_, Self>, target: usize) {
        slf.inner.pauli_z(target);
    }

    fn p(mut slf: PyRefMut<'_, Self>, theta: f64, target: usize) {
        slf.inner.pshift(theta, target);
    }

    fn u(
        mut slf: PyRefMut<'_, Self>,
        a: (f64, f64),
        b: (f64, f64),
        c: (f64, f64),
        d: (f64, f64),
        target: usize,
    ) {
        let g00: Complex = Complex::new(a.0, a.1);
        let g01: Complex = Complex::new(b.0, b.1);
        let g10: Complex = Complex::new(c.0, c.1);
        let g11: Complex = Complex::new(d.0, d.1);
        slf.inner.unitary(g00, g01, g10, g11, target);
    }

    fn cx(mut slf: PyRefMut<'_, Self>, control: usize, target: usize) {
        slf.inner.controlled_pauli_x(control, target);
    }

    fn cy(mut slf: PyRefMut<'_, Self>, control: usize, target: usize) {
        slf.inner.controlled_pauli_y(control, target);
    }

    fn cz(mut slf: PyRefMut<'_, Self>, control: usize, target: usize) {
        slf.inner.controlled_pauli_z(control, target);
    }

    fn cp(mut slf: PyRefMut<'_, Self>, theta: f64, control: usize, target: usize) {
        slf.inner.controlled_pshift(theta, control, target);
    }

    fn cu(
        mut slf: PyRefMut<'_, Self>,
        a: (f64, f64),
        b: (f64, f64),
        c: (f64, f64),
        d: (f64, f64),
        control: usize,
        target: usize,
    ) {
        let g00: Complex = Complex::new(a.0, a.1);
        let g01: Complex = Complex::new(b.0, b.1);
        let g10: Complex = Complex::new(c.0, c.1);
        let g11: Complex = Complex::new(d.0, d.1);
        slf.inner
            .controlled_unitary(g00, g01, g10, g11, control, target);
    }

    fn run(mut slf: PyRefMut<'_, Self>) {
        slf.inner.run();
    }

    fn state(slf: PyRef<'_, Self>) -> PyResult<PyQRegister> {
        Ok(PyQRegister {
            inner: slf.inner.psi.to_owned(),
            nqubits: slf.nqubits,
        })
    }

    fn __repr__(slf: PyRef<'_, Self>) -> String {
        let circuit_visual: VisualCircuit = VisualCircuit::new(slf.nqubits, &slf.inner.gates);
        circuit_visual.circuit
    }
}

#[pyfunction]
fn qft(nqubits: usize) -> PyResult<String> {
    let n: usize = 1 << nqubits;
    let mut states: Vec<Complex> = vec![Complex::new(0.0, 0.0); n];
    states[0].real = 1.0;

    let mut register: QRegister = QRegister::new(states);

    register.quantum_fourier_transform(nqubits);
    Ok(register.to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn qelvin(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyQRegister>()?;
    m.add_class::<PyQCircuit>()?;
    m.add_function(wrap_pyfunction!(qft, m)?)?;
    Ok(())
}
