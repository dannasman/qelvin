use std::fmt;
use std::mem;
use std::ops::*;
use std::collections::VecDeque;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(not(feature = "no-multi-thread"))]
extern crate rayon;
#[cfg(not(feature = "no-multi-thread"))]
use rayon::prelude::*;

pub const PI: f64 = std::f64::consts::PI;

#[derive(Clone, Copy)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Complex { real, imag }
    }
}

#[derive(Clone)]
pub struct QRegister(Vec<Complex>);

impl QRegister {
    pub fn new(states: Vec<Complex>) -> Self {
        Self(states)
    }

    pub fn apply_unitary(
        &mut self,
        g00: Complex,
        g01: Complex,
        g10: Complex,
        g11: Complex,
        t: usize,
    ) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let m1: __m256d = unsafe { _mm256_set_pd(g10.real, g10.real, g00.real, g00.real) };
        let m2: __m256d = unsafe { _mm256_set_pd(g10.imag, -g10.imag, g00.imag, -g00.imag) };
        let m3: __m256d = unsafe { _mm256_set_pd(g11.real, g11.real, g01.real, g01.real) };
        let m4: __m256d = unsafe { _mm256_set_pd(g11.imag, -g11.imag, g01.imag, -g01.imag) };

        let step_block = |state: &mut [Complex]| {
            for (i, j) in (0..hb).zip(hb..b) {
                let rz: f64 = state[i].real;
                let iz: f64 = state[i].imag;
                let ro: f64 = state[j].real;
                let io: f64 = state[j].imag;

                let m5: __m256d = unsafe { _mm256_set_pd(iz, rz, iz, rz) };
                let m6: __m256d = unsafe { _mm256_set_pd(io, ro, io, ro) };
                let mut m0: __m256d = unsafe { _mm256_mul_pd(m1, m5) };
                m0 = unsafe { _mm256_fmadd_pd(m2, _mm256_permute_pd(m5, 0b0101), m0) };
                m0 = unsafe { _mm256_fmadd_pd(m3, m6, m0) };
                m0 = unsafe { _mm256_fmadd_pd(m4, _mm256_permute_pd(m6, 0b0101), m0) };

                (state[i].real, state[i].imag, state[j].real, state[j].imag) =
                    unsafe { mem::transmute(m0) };
            }
        };

        self.par_chunks_mut(b).for_each(step_block);
    }

    pub fn apply_controlled_unitary(
        &mut self,
        g00: Complex,
        g01: Complex,
        g10: Complex,
        g11: Complex,
        c: usize,
        t: usize,
    ) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let m1: __m256d = unsafe { _mm256_set_pd(g10.real, g10.real, g00.real, g00.real) };
        let m2: __m256d = unsafe { _mm256_set_pd(g10.imag, -g10.imag, g00.imag, -g00.imag) };
        let m3: __m256d = unsafe { _mm256_set_pd(g11.real, g11.real, g01.real, g01.real) };
        let m4: __m256d = unsafe { _mm256_set_pd(g11.imag, -g11.imag, g01.imag, -g01.imag) };

        let step_block = |state: &mut [Complex]| {
            for (i, j) in (0..hb).zip(hb..b) {
                let rz = state[i].real;
                let iz = state[i].imag;
                let ro = state[j].real;
                let io = state[j].imag;

                let m5: __m256d = unsafe { _mm256_set_pd(iz, rz, iz, rz) };
                let m6: __m256d = unsafe { _mm256_set_pd(io, ro, io, ro) };
                let mut m0: __m256d = unsafe { _mm256_mul_pd(m1, m5) };
                m0 = unsafe { _mm256_fmadd_pd(m2, _mm256_permute_pd(m5, 0b0101), m0) };
                m0 = unsafe { _mm256_fmadd_pd(m3, m6, m0) };
                m0 = unsafe { _mm256_fmadd_pd(m4, _mm256_permute_pd(m6, 0b0101), m0) };

                let (m0_0, m0_1, m0_2, m0_3): (f64, f64, f64, f64) = unsafe { mem::transmute(m0) };

                (state[i].real, state[i].imag) = if (1 << c) & i == 1 {
                    (m0_0, m0_1)
                } else {
                    (state[i].real, state[i].imag)
                };
                (state[j].real, state[j].imag) = if (1 << c) & j == 1 {
                    (m0_2, m0_3)
                } else {
                    (state[j].real, state[j].imag)
                };
            }
        };

        self.par_chunks_mut(b).for_each(step_block);
    }

    pub fn apply_hadamard(&mut self, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let root: f64 = 0.5f64.sqrt();

        let m1: __m256d = unsafe { _mm256_set_pd(root, root, root, root) };

        let step_block = |state: &mut [Complex]| {
            for (i, j) in (0..hb).zip(hb..b) {
                let rz: f64 = state[i].real;
                let iz: f64 = state[i].imag;
                let ro: f64 = state[j].real;
                let io: f64 = state[j].imag;

                let m2: __m256d = unsafe { _mm256_set_pd(iz, rz, iz, rz) };
                let m3: __m256d = unsafe { _mm256_set_pd(-io, -ro, io, ro) };
                let m4: __m256d = unsafe { _mm256_mul_pd(m1, m2) };
                let m0: __m256d = unsafe { _mm256_fmadd_pd(m1, m3, m4) };

                (state[i].real, state[i].imag, state[j].real, state[j].imag) =
                    unsafe { mem::transmute(m0) };
            }
        };

        self.par_chunks_mut(b).for_each(step_block);
    }

    pub fn apply_pshift(&mut self, theta: f64, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let cos_theta: f64 = theta.cos();
        let sin_theta: f64 = theta.sin();

        let m1: __m128d = unsafe { _mm_set_pd(sin_theta, cos_theta) };
        let m2: __m128d = unsafe { _mm_set_pd(cos_theta, -sin_theta) };

        let step_block = |state: &mut [Complex]| {
            for j in hb..b {
                let ro: f64 = state[j].real;
                let io: f64 = state[j].imag;

                let m3: __m128d = unsafe { _mm_set_pd(ro, ro) };
                let m4: __m128d = unsafe { _mm_set_pd(io, io) };
                let m5: __m128d = unsafe { _mm_mul_pd(m1, m3) };
                let m0: __m128d = unsafe { _mm_fmadd_pd(m2, m4, m5) };

                (state[j].real, state[j].imag) = unsafe { mem::transmute(m0) };
            }
        };

        self.par_chunks_mut(b).for_each(step_block);
    }

    pub fn apply_controlled_pshift(&mut self, theta: f64, c: usize, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let cos_theta: f64 = theta.cos();
        let sin_theta: f64 = theta.sin();

        let m1: __m128d = unsafe { _mm_set_pd(sin_theta, cos_theta) };
        let m2: __m128d = unsafe { _mm_set_pd(cos_theta, -sin_theta) };

        let step_block = |state: &mut [Complex]| {
            for j in hb..b {
                if (1 << c) & j == 1 {
                    let ro: f64 = state[j].real;
                    let io: f64 = state[j].imag;

                    let m3: __m128d = unsafe { _mm_set_pd(ro, ro) };
                    let m4: __m128d = unsafe { _mm_set_pd(io, io) };
                    let m5: __m128d = unsafe { _mm_mul_pd(m1, m3) };
                    let m0: __m128d = unsafe { _mm_fmadd_pd(m2, m4, m5) };

                    (state[j].real, state[j].imag) = unsafe { mem::transmute(m0) };
                }
            }
        };

        self.par_chunks_mut(b).for_each(step_block);
    }

    pub fn apply_pauli_x(&mut self, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let step_block = |state: &mut [Complex]| {
            for (i, j) in (0..hb).zip(hb..b) {
                state.swap(i, j);
            }
        };

        self.par_chunks_mut(b).for_each(step_block);
    }

    pub fn apply_controlled_pauli_x(&mut self, c: usize, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let step_block = |state: &mut [Complex]| {
            for (i, j) in (0..hb).zip(hb..b) {
                let temp: Complex = state[i];

                if (1 << c) & i == 1 {
                    state[i].real = state[j].real;
                    state[i].imag = state[j].imag;
                }

                if (1 << c) & j == 1 {
                    state[j] = temp;
                }
            }
        };

        self.par_chunks_mut(b).for_each(step_block);
    }

    pub fn apply_pauli_y(&mut self, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let step_block = |state: &mut [Complex]| {
            for (i, j) in (0..hb).zip(hb..b) {
                let temp: Complex = state[i];
                state[i].real = state[j].imag;
                state[i].imag = -state[j].real;
                state[j].real = -temp.imag;
                state[j].imag = temp.real;
            }
        };

        self.par_chunks_mut(b).for_each(step_block);
    }

    pub fn apply_controlled_pauli_y(&mut self, c: usize, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let step_block = |state: &mut [Complex]| {
            for (i, j) in (0..hb).zip(hb..b) {
                let temp: Complex = state[i];

                if (1 << c) & i == 1 {
                    state[i].real = state[j].imag;
                    state[i].imag = -state[j].real;
                }

                if (1 << c) & j == 1 {
                    state[j].real = -temp.imag;
                    state[j].imag = temp.real;
                }
            }
        };

        self.par_chunks_mut(b).for_each(step_block);
    }

    pub fn apply_pauli_z(&mut self, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let step_block = |state: &mut [Complex]| {
            for j in hb..b {
                state[j].real = -state[j].real;
                state[j].imag = -state[j].imag;
            }
        };

        self.par_chunks_mut(b).for_each(step_block);
    }

    pub fn apply_controlled_pauli_z(&mut self, c: usize, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let step_block = |state: &mut [Complex]| {
            for j in hb..b {
                if (1 << c) & j == 1 {
                    state[j].real = -state[j].real;
                    state[j].imag = -state[j].imag;
                }
            }
        };

        self.par_chunks_mut(b).for_each(step_block);
    }

    pub fn quantum_fourier_transform(&mut self, nqubits: usize) {
        for j in 0..nqubits {
            for k in 0..j {
                let q: f64 = (1 << (j - k)) as f64;
                let theta: f64 = PI / q;
                self.apply_controlled_pshift(theta, j, k);
            }
            self.apply_hadamard(j);
        }
    }
}

impl fmt::Display for QRegister {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for state in self.iter() {
            write!(f, " {:.3}{:+.3}*j ", state.real, state.imag)?;
        }
        write!(f, "]")?;

        Ok(())
    }
}

impl Deref for QRegister {
    type Target = Vec<Complex>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for QRegister {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Copy)]
pub enum Gate {
    Hadamard(usize),
    PauliX(usize),
    PauliY(usize),
    PauliZ(usize),
    PhaseShift((f64, usize)),
    Unitary((Complex, Complex, Complex, Complex, usize)),
    CPauliX((usize, usize)),
    CPauliY((usize, usize)),
    CPauliZ((usize, usize)),
    CPhaseShift((f64, usize, usize)),
    CUnitary((Complex, Complex, Complex, Complex, usize, usize)),
}

pub struct QCircuit {
    pub psi: QRegister,
    gates: VecDeque<Gate>,
}

impl QCircuit {
    pub fn new(psi: QRegister, gates: VecDeque<Gate>) -> Self {
        QCircuit { psi, gates }
    }

    pub fn run(&mut self) {
        while let Some(gate) = self.gates.pop_front() {
            match gate {
                Gate::Hadamard(t) => self.psi.apply_hadamard(t),
                Gate::PauliX(t) => self.psi.apply_pauli_x(t),
                Gate::PauliY(t) => self.psi.apply_pauli_y(t),
                Gate::PauliZ(t) => self.psi.apply_pauli_z(t),
                Gate::PhaseShift((theta, t)) => self.psi.apply_pshift(theta, t),
                Gate::Unitary((g00, g01, g10, g11, t)) => self.psi.apply_unitary(g00, g01, g10, g11, t),
                Gate::CPauliX((c, t)) => self.psi.apply_controlled_pauli_x(c, t),
                Gate::CPauliY((c, t)) => self.psi.apply_controlled_pauli_y(c, t),
                Gate::CPauliZ((c, t)) => self.psi.apply_controlled_pauli_z(c, t),
                Gate::CPhaseShift((theta, c, t)) => self.psi.apply_controlled_pshift(theta, c, t),
                Gate::CUnitary((g00, g01, g10, g11, c, t)) => self.psi.apply_controlled_unitary(g00, g01, g10, g11, c, t),
            }
        }
    }

    pub fn hadamard(&mut self, t: usize) {
        self.gates.push_back(Gate::Hadamard(t));
    }

    pub fn pauli_x(&mut self, t: usize) {
        self.gates.push_back(Gate::PauliX(t));
    }

    pub fn pauli_y(&mut self, t: usize) {
        self.gates.push_back(Gate::PauliY(t));
    }

    pub fn pauli_z(&mut self, t: usize) {
        self.gates.push_back(Gate::PauliZ(t));
    }

    pub fn pshift(&mut self, theta: f64, t: usize) {
        self.gates.push_back(Gate::PhaseShift((theta, t)));
    }

    pub fn unitary(&mut self, g00: Complex, g01: Complex, g10: Complex, g11: Complex, t: usize) {
        self.gates.push_back(Gate::Unitary((g00, g01, g10, g11, t)));
    }

    pub fn controlled_pauli_x(&mut self, c: usize, t: usize) {
        self.gates.push_back(Gate::CPauliX((c, t)));
    }

    pub fn controlled_pauli_y(&mut self, c: usize, t: usize) {
        self.gates.push_back(Gate::CPauliY((c, t)));
    }

    pub fn controlled_pauli_z(&mut self, c: usize, t: usize) {
        self.gates.push_back(Gate::CPauliZ((c, t)));
    }

    pub fn controlled_pshift(&mut self, theta: f64, c: usize, t: usize) {
        self.gates.push_back(Gate::CPhaseShift((theta, c, t)));
    }

    pub fn controlled_unitary(&mut self, g00: Complex, g01: Complex, g10: Complex, g11: Complex, c: usize, t: usize) {
        self.gates.push_back(Gate::CUnitary((g00, g01, g10, g11, c, t)));
    }
}
