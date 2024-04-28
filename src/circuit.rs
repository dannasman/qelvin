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

        let step_block = |(nb, state): (usize, &mut [Complex])| {
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

                (state[i].real, state[i].imag) = if (1 << c) & (nb*b + i) > 0 {
                    (m0_0, m0_1)
                } else {
                    (state[i].real, state[i].imag)
                };
                (state[j].real, state[j].imag) = if (1 << c) & (nb*b + j) > 0 {
                    (m0_2, m0_3)
                } else {
                    (state[j].real, state[j].imag)
                };
            }
        };

        self.par_chunks_mut(b).enumerate().for_each(step_block);
    }

    pub fn apply_doubly_controlled_unitary(
        &mut self,
        g00: Complex,
        g01: Complex,
        g10: Complex,
        g11: Complex,
        c1: usize,
        c2: usize,
        t: usize,
    ) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let m1: __m256d = unsafe { _mm256_set_pd(g10.real, g10.real, g00.real, g00.real) };
        let m2: __m256d = unsafe { _mm256_set_pd(g10.imag, -g10.imag, g00.imag, -g00.imag) };
        let m3: __m256d = unsafe { _mm256_set_pd(g11.real, g11.real, g01.real, g01.real) };
        let m4: __m256d = unsafe { _mm256_set_pd(g11.imag, -g11.imag, g01.imag, -g01.imag) };

        let step_block = |(nb, state): (usize, &mut [Complex])| {
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

                (state[i].real, state[i].imag) = if (1 << c1) & (nb*b + i) > 0 && (1 << c2) & (nb*b + i) > 0 {
                    (m0_0, m0_1)
                } else {
                    (state[i].real, state[i].imag)
                };
                (state[j].real, state[j].imag) = if (1 << c1) & (nb*b + j) > 0 && (1 << c2) & (nb*b + j) > 0 {
                    (m0_2, m0_3)
                } else {
                    (state[j].real, state[j].imag)
                };
            }
        };

        self.par_chunks_mut(b).enumerate().for_each(step_block);
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

        let step_block = |(nb, state): (usize, &mut [Complex])| {
            for j in hb..b {
                if (1 << c) & (nb*b + j) > 0 {
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
        self.par_chunks_mut(b).enumerate().for_each(step_block);
    }

    pub fn apply_doubly_controlled_pshift(&mut self, theta: f64, c1: usize, c2: usize, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let cos_theta: f64 = theta.cos();
        let sin_theta: f64 = theta.sin();

        let m1: __m128d = unsafe { _mm_set_pd(sin_theta, cos_theta) };
        let m2: __m128d = unsafe { _mm_set_pd(cos_theta, -sin_theta) };

        let step_block = |(nb, state): (usize, &mut [Complex])| {
            for j in hb..b {
                if (1 << c1) & (nb*b + j) > 0 && (1 << c2) & (nb*b + j) > 0 {
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

        self.par_chunks_mut(b).enumerate().for_each(step_block);
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

        let step_block = |(nb, state): (usize, &mut [Complex])| {
            for (i, j) in (0..hb).zip(hb..b) {
                let temp_real: f64 = state[i].real;
                let temp_imag: f64 = state[i].imag;

                if (1 << c) & (nb*b + i) > 0 {
                    state[i].real = state[j].real;
                    state[i].imag = state[j].imag;
                }

                if (1 << c) & (nb*b + j) > 0 {
                    state[j].real = temp_real;
                    state[j].imag = temp_imag;
                }
            }
        };

        self.par_chunks_mut(b).enumerate().for_each(step_block);
    }

    pub fn apply_doubly_controlled_pauli_x(&mut self, c1: usize, c2: usize, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let step_block = |(nb, state): (usize, &mut [Complex])| {
            for (i, j) in (0..hb).zip(hb..b) {
                let temp_real: f64 = state[i].real;
                let temp_imag: f64 = state[i].imag;

                if (1 << c1) & (nb*b + i) > 0 && (1 << c2) & (nb*b + i) > 0 {
                    state[i].real = state[j].real;
                    state[i].imag = state[j].imag;
                }

                if (1 << c1) & (nb*b + j) > 0 && (1 << c2) & (nb*b + j) > 0 {
                    state[j].real = temp_real;
                    state[j].imag = temp_imag;
                }
            }
        };

        self.par_chunks_mut(b).enumerate().for_each(step_block);
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

        let step_block = |(nb, state): (usize, &mut [Complex])| {
            for (i, j) in (0..hb).zip(hb..b) {
                let temp: Complex = state[i];

                if (1 << c) & (nb*b + i) > 0 {
                    state[i].real = state[j].imag;
                    state[i].imag = -state[j].real;
                }

                if (1 << c) & (nb*b + j) > 0 {
                    state[j].real = -temp.imag;
                    state[j].imag = temp.real;
                }
            }
        };

        self.par_chunks_mut(b).enumerate().for_each(step_block);
    }

    pub fn apply_doubly_controlled_pauli_y(&mut self, c1: usize, c2: usize, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let step_block = |(nb, state): (usize, &mut [Complex])| {
            for (i, j) in (0..hb).zip(hb..b) {
                let temp: Complex = state[i];

                if (1 << c1) & (nb*b + i) > 0 && (1 << c2) & (nb*b + i) > 0 {
                    state[i].real = state[j].imag;
                    state[i].imag = -state[j].real;
                }

                if (1 << c1) & (nb*b + j) > 0 && (1 << c2) & (nb*b + j) > 0 {
                    state[j].real = -temp.imag;
                    state[j].imag = temp.real;
                }
            }
        };

        self.par_chunks_mut(b).enumerate().for_each(step_block);
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

        let step_block = |(nb, state): (usize, &mut [Complex])| {
            for j in hb..b {
                if (1 << c) & (nb*b + j) > 0 {
                    state[j].real = -state[j].real;
                    state[j].imag = -state[j].imag;
                }
            }
        };

        self.par_chunks_mut(b).enumerate().for_each(step_block);
    }

    pub fn apply_doubly_controlled_pauli_z(&mut self, c1: usize, c2: usize, t: usize) {
        let b: usize = 1 << (t + 1);
        let hb: usize = 1 << t;

        let step_block = |(nb, state): (usize, &mut [Complex])| {
            for j in hb..b {
                if (1 << c1) & (nb*b + j) > 0 && (1 << c2) & (nb*b + j) > 0 {
                    state[j].real = -state[j].real;
                    state[j].imag = -state[j].imag;
                }
            }
        };

        self.par_chunks_mut(b).enumerate().for_each(step_block);
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
        let nqubits: usize = self.len().ilog2() as usize;
        write!(f, "[")?;
        for (i, state) in self.iter().enumerate() {
            if i != 0 && i % nqubits == 0 {
                write!(f, "\n")?;
            }
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
    DCPauliX((usize, usize, usize)),
    DCPauliY((usize, usize, usize)),
    DCPauliZ((usize, usize, usize)),
    DCPhaseShift((f64, usize, usize, usize)),
    DCUnitary((Complex, Complex, Complex, Complex, usize, usize, usize)),
}

pub struct QCircuit {
    pub psi: QRegister,
    pub gates: VecDeque<Gate>,
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
                Gate::Unitary((g00, g01, g10, g11, t)) => {
                    self.psi.apply_unitary(g00, g01, g10, g11, t)
                }
                Gate::CPauliX((c, t)) => self.psi.apply_controlled_pauli_x(c, t),
                Gate::CPauliY((c, t)) => self.psi.apply_controlled_pauli_y(c, t),
                Gate::CPauliZ((c, t)) => self.psi.apply_controlled_pauli_z(c, t),
                Gate::CPhaseShift((theta, c, t)) => self.psi.apply_controlled_pshift(theta, c, t),
                Gate::CUnitary((g00, g01, g10, g11, c, t)) => {
                    self.psi.apply_controlled_unitary(g00, g01, g10, g11, c, t)
                }
                Gate::DCPauliX((c1, c2, t)) => self.psi.apply_doubly_controlled_pauli_x(c1, c2, t),
                Gate::DCPauliY((c1, c2, t)) => self.psi.apply_doubly_controlled_pauli_y(c1, c2, t),
                Gate::DCPauliZ((c1, c2, t)) => self.psi.apply_doubly_controlled_pauli_z(c1, c2, t),
                Gate::DCPhaseShift((theta, c1, c2, t)) => {
                    self.psi.apply_doubly_controlled_pshift(theta, c1, c2, t)
                }
                Gate::DCUnitary((g00, g01, g10, g11, c1, c2, t)) => self
                    .psi
                    .apply_doubly_controlled_unitary(g00, g01, g10, g11, c1, c2, t),
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

    pub fn controlled_unitary(
        &mut self,
        g00: Complex,
        g01: Complex,
        g10: Complex,
        g11: Complex,
        c: usize,
        t: usize,
    ) {
        self.gates
            .push_back(Gate::CUnitary((g00, g01, g10, g11, c, t)));
    }

    pub fn doubly_controlled_pauli_x(&mut self, c1: usize, c2: usize, t: usize) {
        self.gates.push_back(Gate::DCPauliX((c1, c2, t)));
    }

    pub fn doubly_controlled_pauli_y(&mut self, c1: usize, c2: usize, t: usize) {
        self.gates.push_back(Gate::DCPauliY((c1, c2, t)));
    }

    pub fn doubly_controlled_pauli_z(&mut self, c1: usize, c2: usize, t: usize) {
        self.gates.push_back(Gate::DCPauliZ((c1, c2, t)));
    }

    pub fn doubly_controlled_pshift(&mut self, theta: f64, c1: usize, c2: usize, t: usize) {
        self.gates.push_back(Gate::DCPhaseShift((theta, c1, c2, t)));
    }

    pub fn doubly_controlled_unitary(
        &mut self,
        g00: Complex,
        g01: Complex,
        g10: Complex,
        g11: Complex,
        c1: usize,
        c2: usize,
        t: usize,
    ) {
        self.gates
            .push_back(Gate::DCUnitary((g00, g01, g10, g11, c1, c2, t)));
    }
}

// TODO: MORE TESTS! MORE!
#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_hadamard() {
        let root: f64 = 0.5f64.sqrt();

        let mut states0: Vec<Complex> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let mut states1: Vec<Complex> = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];

        let mut psi0: QRegister = QRegister::new(states0);
        let mut psi1: QRegister = QRegister::new(states1);

        psi0.apply_hadamard(0);
        psi1.apply_hadamard(0);

        assert!((psi0[0].real - root).abs() < f64::EPSILON);
        assert!((psi0[1].real - root).abs() < f64::EPSILON);
        assert!((psi1[0].real - root).abs() < f64::EPSILON);
        assert!((psi1[1].real + root).abs() < f64::EPSILON);

        states0 = vec![Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)];
        states1 = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)];

        psi0 = QRegister::new(states0);
        psi1 = QRegister::new(states1);

        psi0.apply_hadamard(0);
        psi1.apply_hadamard(0);

        assert!((psi0[0].imag - root).abs() < f64::EPSILON);
        assert!((psi0[1].imag - root).abs() < f64::EPSILON);
        assert!((psi1[0].imag - root).abs() < f64::EPSILON);
        assert!((psi1[1].imag + root).abs() < f64::EPSILON);

        states0 = vec![Complex::new(0.5, 0.5), Complex::new(0.5, 0.5)];

        psi0 = QRegister::new(states0);

        psi0.apply_hadamard(0);

        assert!((psi0[0].real - root).abs() < f64::EPSILON);
        assert!((psi0[0].imag - root).abs() < f64::EPSILON);
        assert!(psi0[1].real.abs() < f64::EPSILON);
        assert!(psi0[1].imag.abs() < f64::EPSILON);
    }

    #[test]
    fn test_pauli_x() {
        let mut states0: Vec<Complex> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let mut states1: Vec<Complex> = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];

        let mut psi0: QRegister = QRegister::new(states0);
        let mut psi1: QRegister = QRegister::new(states1);

        psi0.apply_pauli_x(0);
        psi1.apply_pauli_x(0);

        assert!(psi0[0].real.abs() < f64::EPSILON);
        assert!((psi0[1].real - 1.0).abs() < f64::EPSILON);
        assert!((psi1[0].real - 1.0).abs() < f64::EPSILON);
        assert!(psi1[1].real.abs() < f64::EPSILON);

        states0 = vec![Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)];
        states1 = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)];

        psi0 = QRegister::new(states0);
        psi1 = QRegister::new(states1);

        psi0.apply_pauli_x(0);
        psi1.apply_pauli_x(0);

        assert!(psi0[0].imag.abs() < f64::EPSILON);
        assert!((psi0[1].imag - 1.0).abs() < f64::EPSILON);
        assert!((psi1[0].imag - 1.0).abs() < f64::EPSILON);
        assert!(psi1[1].imag.abs() < f64::EPSILON);
    }

    #[test]
    fn test_pauli_y() {
        let mut states0: Vec<Complex> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let mut states1: Vec<Complex> = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];

        let mut psi0: QRegister = QRegister::new(states0);
        let mut psi1: QRegister = QRegister::new(states1);

        psi0.apply_pauli_y(0);
        psi1.apply_pauli_y(0);

        assert!(psi0[0].real.abs() < f64::EPSILON);
        assert!(psi0[0].imag.abs() < f64::EPSILON);
        assert!(psi0[1].real.abs() < f64::EPSILON);
        assert!((psi0[1].imag - 1.0).abs() < f64::EPSILON);

        assert!(psi1[0].real.abs() < f64::EPSILON);
        assert!((psi1[0].imag + 1.0).abs() < f64::EPSILON);
        assert!(psi1[1].real.abs() < f64::EPSILON);
        assert!(psi1[1].imag.abs() < f64::EPSILON);

        states0 = vec![Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)];
        states1 = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)];

        psi0 = QRegister::new(states0);
        psi1 = QRegister::new(states1);

        psi0.apply_pauli_y(0);
        psi1.apply_pauli_y(0);

        assert!(psi0[0].real.abs() < f64::EPSILON);
        assert!(psi0[0].imag.abs() < f64::EPSILON);
        assert!((psi0[1].real + 1.0).abs() < f64::EPSILON);
        assert!(psi0[1].imag.abs() < f64::EPSILON);

        assert!((psi1[0].real - 1.0).abs() < f64::EPSILON);
        assert!(psi1[0].imag.abs() < f64::EPSILON);
        assert!(psi1[1].real.abs() < f64::EPSILON);
        assert!(psi1[1].imag.abs() < f64::EPSILON);
    }

    #[test]
    fn test_pauli_z() {
        let states0: Vec<Complex> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let states1: Vec<Complex> = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];

        let mut psi0: QRegister = QRegister::new(states0);
        let mut psi1: QRegister = QRegister::new(states1);

        psi0.apply_pauli_z(0);
        psi1.apply_pauli_z(0);

        assert!((psi0[0].real - 1.0).abs() < f64::EPSILON);
        assert!(psi0[1].real.abs() < f64::EPSILON);
        assert!(psi1[0].real.abs() < f64::EPSILON);
        assert!((psi1[1].real + 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pshift() {
        let theta: f64 = PI / 3.0;

        let states0: Vec<Complex> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let states1: Vec<Complex> = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];

        let mut psi0: QRegister = QRegister::new(states0);
        let mut psi1: QRegister = QRegister::new(states1);

        psi0.apply_pshift(theta, 0);
        psi1.apply_pshift(theta, 0);

        assert!((psi0[0].real - 1.0).abs() < f64::EPSILON);
        assert!(psi0[1].real.abs() < f64::EPSILON);
        assert!(psi1[0].real.abs() < f64::EPSILON);
        assert!(psi1[0].imag.abs() < f64::EPSILON);
        assert!((psi1[1].real - theta.cos()).abs() < f64::EPSILON);
        assert!((psi1[1].imag - theta.sin()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unitary() {
        let theta: f64 = PI / 3.0;

        let g00: Complex = Complex::new(theta.cos(), -theta.sin());
        let g01: Complex = Complex::new(0.0, 0.0);
        let g10: Complex = Complex::new(0.0, 0.0);
        let g11: Complex = Complex::new(theta.cos(), theta.sin());

        let states0: Vec<Complex> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
        let states1: Vec<Complex> = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];

        let mut psi0: QRegister = QRegister::new(states0);
        let mut psi1: QRegister = QRegister::new(states1);

        psi0.apply_unitary(g00, g01, g10, g11, 0);
        psi1.apply_unitary(g00, g01, g10, g11, 0);

        assert!((psi0[0].real - theta.cos()) < f64::EPSILON);
        assert!((psi0[0].imag + theta.sin()) < f64::EPSILON);
        assert!(psi0[1].real < f64::EPSILON);
        assert!(psi0[1].imag < f64::EPSILON);
        assert!(psi1[0].real < f64::EPSILON);
        assert!(psi1[0].imag < f64::EPSILON);
        assert!((psi1[1].real - theta.cos()) < f64::EPSILON);
        assert!((psi1[1].imag - theta.sin()) < f64::EPSILON);
    }

    #[test]
    fn test_controlled_pauli_x() {
        let states00: Vec<Complex> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states01: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states10: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states11: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        let mut psi00: QRegister = QRegister::new(states00);
        let mut psi01: QRegister = QRegister::new(states01);
        let mut psi10: QRegister = QRegister::new(states10);
        let mut psi11: QRegister = QRegister::new(states11);

        psi00.apply_controlled_pauli_x(0, 1);
        psi01.apply_controlled_pauli_x(0, 1);
        psi10.apply_controlled_pauli_x(0, 1);
        psi11.apply_controlled_pauli_x(0, 1);

        assert!((psi00[0b00].real - 1.0).abs() < f64::EPSILON);
        assert!(psi00[0b01].real.abs() < f64::EPSILON);
        assert!(psi00[0b10].real.abs() < f64::EPSILON);
        assert!(psi00[0b11].real.abs() < f64::EPSILON);

        assert!(psi01[0b00].real.abs() < f64::EPSILON);
        assert!(psi01[0b01].real.abs() < f64::EPSILON);
        assert!(psi01[0b10].real.abs() < f64::EPSILON);
        assert!((psi01[0b11].real - 1.0).abs() < f64::EPSILON);

        assert!(psi10[0b00].real.abs() < f64::EPSILON);
        assert!(psi10[0b01].real.abs() < f64::EPSILON);
        assert!((psi10[0b10].real - 1.0).abs() < f64::EPSILON);
        assert!(psi10[0b11].real.abs() < f64::EPSILON);

        assert!(psi11[0b00].real.abs() < f64::EPSILON);
        assert!((psi11[0b01].real - 1.0).abs() < f64::EPSILON);
        assert!(psi11[0b10].real.abs() < f64::EPSILON);
        assert!(psi11[0b11].real.abs() < f64::EPSILON);
    }

    #[test]
    fn test_controlled_pauli_y() {
        let states00: Vec<Complex> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states01: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states10: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states11: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        let mut psi00: QRegister = QRegister::new(states00);
        let mut psi01: QRegister = QRegister::new(states01);
        let mut psi10: QRegister = QRegister::new(states10);
        let mut psi11: QRegister = QRegister::new(states11);

        psi00.apply_controlled_pauli_y(0, 1);
        psi01.apply_controlled_pauli_y(0, 1);
        psi10.apply_controlled_pauli_y(0, 1);
        psi11.apply_controlled_pauli_y(0, 1);

        assert!((psi00[0b00].real - 1.0).abs() < f64::EPSILON);
        assert!(psi00[0b01].real.abs() < f64::EPSILON);
        assert!(psi00[0b10].real.abs() < f64::EPSILON);
        assert!(psi00[0b11].real.abs() < f64::EPSILON);

        assert!(psi01[0b00].real.abs() < f64::EPSILON);
        assert!(psi01[0b01].real.abs() < f64::EPSILON);
        assert!(psi01[0b10].real.abs() < f64::EPSILON);
        assert!(psi01[0b11].real.abs() < f64::EPSILON);
        assert!((psi01[0b11].imag - 1.0).abs() < f64::EPSILON);

        assert!(psi10[0b00].real.abs() < f64::EPSILON);
        assert!(psi10[0b01].real.abs() < f64::EPSILON);
        assert!((psi10[0b10].real - 1.0).abs() < f64::EPSILON);
        assert!(psi10[0b11].real.abs() < f64::EPSILON);

        assert!(psi11[0b00].real.abs() < f64::EPSILON);
        assert!((psi11[0b01].imag + 1.0).abs() < f64::EPSILON);
        assert!(psi11[0b01].real.abs() < f64::EPSILON);
        assert!(psi11[0b10].real.abs() < f64::EPSILON);
        assert!(psi11[0b11].real.abs() < f64::EPSILON);
    }

    #[test]
    fn test_controlled_pauli_z() {
        let states00: Vec<Complex> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states01: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states10: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states11: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        let mut psi00: QRegister = QRegister::new(states00);
        let mut psi01: QRegister = QRegister::new(states01);
        let mut psi10: QRegister = QRegister::new(states10);
        let mut psi11: QRegister = QRegister::new(states11);

        psi00.apply_controlled_pauli_z(0, 1);
        psi01.apply_controlled_pauli_z(0, 1);
        psi10.apply_controlled_pauli_z(0, 1);
        psi11.apply_controlled_pauli_z(0, 1);

        assert!((psi00[0b00].real - 1.0).abs() < f64::EPSILON);
        assert!(psi00[0b01].real.abs() < f64::EPSILON);
        assert!(psi00[0b10].real.abs() < f64::EPSILON);
        assert!(psi00[0b11].real.abs() < f64::EPSILON);

        assert!(psi01[0b00].real.abs() < f64::EPSILON);
        assert!((psi01[0b01].real - 1.0).abs() < f64::EPSILON);
        assert!(psi01[0b10].real.abs() < f64::EPSILON);
        assert!(psi01[0b11].real.abs() < f64::EPSILON);

        assert!(psi10[0b00].real.abs() < f64::EPSILON);
        assert!(psi10[0b01].real.abs() < f64::EPSILON);
        assert!((psi10[0b10].real - 1.0).abs() < f64::EPSILON);
        assert!(psi10[0b11].real.abs() < f64::EPSILON);

        assert!(psi11[0b00].real.abs() < f64::EPSILON);
        assert!(psi11[0b01].real.abs() < f64::EPSILON);
        assert!(psi11[0b10].real.abs() < f64::EPSILON);
        assert!((psi11[0b11].real + 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_controlled_pshift() {
        let theta: f64 = PI / 3.0;

        let states00: Vec<Complex> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states01: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states10: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states11: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        let mut psi00: QRegister = QRegister::new(states00);
        let mut psi01: QRegister = QRegister::new(states01);
        let mut psi10: QRegister = QRegister::new(states10);
        let mut psi11: QRegister = QRegister::new(states11);

        psi00.apply_controlled_pshift(theta, 0, 1);
        psi01.apply_controlled_pshift(theta, 0, 1);
        psi10.apply_controlled_pshift(theta, 0, 1);
        psi11.apply_controlled_pshift(theta, 0, 1);

        assert!((psi00[0b00].real - 1.0).abs() < f64::EPSILON);
        assert!(psi00[0b01].real.abs() < f64::EPSILON);
        assert!(psi00[0b10].real.abs() < f64::EPSILON);
        assert!(psi00[0b11].real.abs() < f64::EPSILON);

        assert!(psi01[0b00].real.abs() < f64::EPSILON);
        assert!((psi01[0b01].real - 1.0).abs() < f64::EPSILON);
        assert!(psi01[0b10].real.abs() < f64::EPSILON);
        assert!(psi01[0b11].real.abs() < f64::EPSILON);

        assert!(psi10[0b00].real.abs() < f64::EPSILON);
        assert!(psi10[0b01].real.abs() < f64::EPSILON);
        assert!((psi10[0b10].real - 1.0).abs() < f64::EPSILON);
        assert!(psi10[0b11].real.abs() < f64::EPSILON);

        assert!(psi11[0b00].real.abs() < f64::EPSILON);
        assert!(psi11[0b01].real.abs() < f64::EPSILON);
        assert!(psi11[0b10].real.abs() < f64::EPSILON);
        assert!((psi11[0b11].real - theta.cos()).abs() < f64::EPSILON);
        assert!((psi11[0b11].imag - theta.sin()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_controlled_unitary() {
        let theta: f64 = PI / 3.0;

        let g00: Complex = Complex::new(theta.cos(), -theta.sin());
        let g01: Complex = Complex::new(0.0, 0.0);
        let g10: Complex = Complex::new(0.0, 0.0);
        let g11: Complex = Complex::new(theta.cos(), theta.sin());

        let states00: Vec<Complex> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states01: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states10: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let states11: Vec<Complex> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        let mut psi00: QRegister = QRegister::new(states00);
        let mut psi01: QRegister = QRegister::new(states01);
        let mut psi10: QRegister = QRegister::new(states10);
        let mut psi11: QRegister = QRegister::new(states11);

        psi00.apply_controlled_unitary(g00, g01, g10, g11, 0, 1);
        psi01.apply_controlled_unitary(g00, g01, g10, g11, 0, 1);
        psi10.apply_controlled_unitary(g00, g01, g10, g11, 0, 1);
        psi11.apply_controlled_unitary(g00, g01, g10, g11, 0, 1);

        assert!((psi00[0b00].real - 1.0).abs() < f64::EPSILON);
        assert!(psi00[0b01].real.abs() < f64::EPSILON);
        assert!(psi00[0b10].real.abs() < f64::EPSILON);
        assert!(psi00[0b11].real.abs() < f64::EPSILON);

        assert!(psi01[0b00].real.abs() < f64::EPSILON);
        assert!((psi01[0b01].real - theta.cos()).abs() < f64::EPSILON);
        assert!((psi01[0b01].imag + theta.sin()).abs() < f64::EPSILON);
        assert!(psi01[0b10].real.abs() < f64::EPSILON);
        assert!(psi01[0b11].real.abs() < f64::EPSILON);

        assert!(psi10[0b00].real.abs() < f64::EPSILON);
        assert!(psi10[0b01].real.abs() < f64::EPSILON);
        assert!((psi10[0b10].real - 1.0).abs() < f64::EPSILON);
        assert!(psi10[0b11].real.abs() < f64::EPSILON);

        assert!(psi11[0b00].real.abs() < f64::EPSILON);
        assert!(psi11[0b01].real.abs() < f64::EPSILON);
        assert!(psi11[0b10].real.abs() < f64::EPSILON);
        assert!((psi11[0b11].real - theta.cos()).abs() < f64::EPSILON);
        assert!((psi11[0b11].imag - theta.sin()).abs() < f64::EPSILON);
    }
}
