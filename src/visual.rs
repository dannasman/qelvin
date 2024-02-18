use crate::circuit::*;
use std::cmp;
use std::collections::VecDeque;

#[derive(Clone)]
pub enum VisualBlock {
    HLine,
    VLine,
    Gate(String),
    Control((usize, usize)),
}

impl VisualBlock {
    pub fn get_first_line(&self) -> String {
        match self {
            VisualBlock::HLine => String::from("   "),
            VisualBlock::VLine => String::from(" | "),
            VisualBlock::Gate(_) => String::from(" - "),
            VisualBlock::Control((c, t)) => {
                if c < t {
                    String::from("   ")
                } else {
                    String::from(" | ")
                }
            }
        }
    }

    pub fn get_second_line(&self) -> String {
        match self {
            VisualBlock::HLine => String::from("---"),
            VisualBlock::VLine => String::from("-|-"),
            VisualBlock::Gate(s) => format!("|{}|", s),
            VisualBlock::Control((_, _)) => String::from("-o-"),
        }
    }

    pub fn get_third_line(&self) -> String {
        match self {
            VisualBlock::HLine => String::from("   "),
            VisualBlock::VLine => String::from(" | "),
            VisualBlock::Gate(_) => String::from(" - "),
            VisualBlock::Control((c, t)) => {
                if c < t {
                    String::from(" | ")
                } else {
                    String::from("   ")
                }
            }
        }
    }
}

pub struct VisualCircuit {
    pub circuit: String,
}

impl VisualCircuit {
    pub fn new(nqubits: usize, gates: &VecDeque<Gate>) -> Self {
        let mut occupied: Vec<bool> = vec![true; nqubits];
        let mut grid: Vec<Vec<VisualBlock>> = vec![Vec::new(); nqubits];
        gates.iter().for_each(|gate| match *gate {
            Gate::Hadamard(t) => {
                if !occupied[t] {
                    grid[t].pop();
                    grid[t].push(VisualBlock::Gate(String::from("H")));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("H")));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
            Gate::PauliX(t) => {
                if !occupied[t] {
                    grid[t].pop();
                    grid[t].push(VisualBlock::Gate(String::from("X")));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("X")));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
            Gate::PauliY(t) => {
                if !occupied[t] {
                    grid[t].pop();
                    grid[t].push(VisualBlock::Gate(String::from("Y")));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("Y")));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
            Gate::PauliZ(t) => {
                if !occupied[t] {
                    grid[t].pop();
                    grid[t].push(VisualBlock::Gate(String::from("Z")));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("Z")));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
            Gate::PhaseShift((_, t)) => {
                if !occupied[t] {
                    grid[t].pop();
                    grid[t].push(VisualBlock::Gate(String::from("P")));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("P")));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
            Gate::Unitary((_, _, _, _, t)) => {
                if !occupied[t] {
                    grid[t].pop();
                    grid[t].push(VisualBlock::Gate(String::from("U")));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("U")));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
            Gate::CPauliX((c, t)) => {
                let min: usize = cmp::min(t, c);
                let max: usize = cmp::max(t, c);
                if occupied[min..(max + 1)].iter().all(|&o| o == false) {
                    for i in min..(max + 1) {
                        grid[i].pop();
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("X")));
                            occupied[i] = true;
                        } else if i == c {
                            grid[i].push(VisualBlock::Control((c, t)));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::VLine);
                            occupied[i] = true;
                        }
                    }
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("X")));
                            occupied[i] = true;
                        } else if i == c {
                            grid[i].push(VisualBlock::Control((c, t)));
                            occupied[i] = true;
                        } else if i < max && i > min {
                            grid[i].push(VisualBlock::VLine);
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
            Gate::CPauliY((c, t)) => {
                let min: usize = cmp::min(t, c);
                let max: usize = cmp::max(t, c);
                if occupied[min..(max + 1)].iter().all(|&o| o == false) {
                    for i in min..(max + 1) {
                        grid[i].pop();
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("Y")));
                            occupied[i] = true;
                        } else if i == c {
                            grid[i].push(VisualBlock::Control((c, t)));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::VLine);
                            occupied[i] = true;
                        }
                    }
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("Y")));
                            occupied[i] = true;
                        } else if i == c {
                            grid[i].push(VisualBlock::Control((c, t)));
                            occupied[i] = true;
                        } else if i < max && i > min {
                            grid[i].push(VisualBlock::VLine);
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
            Gate::CPauliZ((c, t)) => {
                let min: usize = cmp::min(t, c);
                let max: usize = cmp::max(t, c);
                if occupied[min..(max + 1)].iter().all(|&o| o == false) {
                    for i in min..(max + 1) {
                        grid[i].pop();
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("Z")));
                            occupied[i] = true;
                        } else if i == c {
                            grid[i].push(VisualBlock::Control((c, t)));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::VLine);
                            occupied[i] = true;
                        }
                    }
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("Z")));
                            occupied[i] = true;
                        } else if i == c {
                            grid[i].push(VisualBlock::Control((c, t)));
                            occupied[i] = true;
                        } else if i < max && i > min {
                            grid[i].push(VisualBlock::VLine);
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
            Gate::CPhaseShift((_, c, t)) => {
                let min: usize = cmp::min(t, c);
                let max: usize = cmp::max(t, c);
                if occupied[min..(max + 1)].iter().all(|&o| o == false) {
                    for i in min..(max + 1) {
                        grid[i].pop();
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("P")));
                            occupied[i] = true;
                        } else if i == c {
                            grid[i].push(VisualBlock::Control((c, t)));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::VLine);
                            occupied[i] = true;
                        }
                    }
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("P")));
                            occupied[i] = true;
                        } else if i == c {
                            grid[i].push(VisualBlock::Control((c, t)));
                            occupied[i] = true;
                        } else if i < max && i > min {
                            grid[i].push(VisualBlock::VLine);
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
            Gate::CUnitary((_, _, _, _, c, t)) => {
                let min: usize = cmp::min(t, c);
                let max: usize = cmp::max(t, c);
                if occupied[min..(max + 1)].iter().all(|&o| o == false) {
                    for i in min..(max + 1) {
                        grid[i].pop();
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("U")));
                            occupied[i] = true;
                        } else if i == c {
                            grid[i].push(VisualBlock::Control((c, t)));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::VLine);
                            occupied[i] = true;
                        }
                    }
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("U")));
                            occupied[i] = true;
                        } else if i == c {
                            grid[i].push(VisualBlock::Control((c, t)));
                            occupied[i] = true;
                        } else if i < max && i > min {
                            grid[i].push(VisualBlock::VLine);
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
        });

        let mut circuit: String = String::new();
        let max_len: usize = (nqubits - 1).to_string().len() + 1;
        for i in 0..nqubits {
            let i_len: usize = i.to_string().len() + 1;
            circuit.push_str(&" ".repeat(max_len + 1));
            grid[i]
                .iter()
                .for_each(|g| circuit.push_str(&g.get_first_line()));
            circuit.push_str(" \n");

            circuit.push_str(&format!("{}q{}-", " ".repeat(max_len - i_len), i));
            grid[i]
                .iter()
                .for_each(|g| circuit.push_str(&g.get_second_line()));
            circuit.push_str("-\n");

            circuit.push_str(&" ".repeat(max_len + 1));
            grid[i]
                .iter()
                .for_each(|g| circuit.push_str(&g.get_third_line()));
            circuit.push_str(" \n");
        }
        VisualCircuit { circuit: circuit }
    }
}
