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
    pub fn get_line(&self, n: usize) -> String {
        match self {
            VisualBlock::HLine => match n {
                0..=1 | 3..=4 => String::from("           "),
                2 => String::from("-----------"),
                _ => panic!("line number out of bounds"),
            },
            VisualBlock::VLine => match n {
                0..=1 | 3..=4 => String::from("     |     "),
                2 => String::from("-----|-----"),
                _ => panic!("line number out of bounds"),
            },
            VisualBlock::Gate(s) => match n {
                0 | 4 => String::from(" --------- "),
                1 | 3 => String::from("|         |"),
                2 => format!("|{}|", s),
                _ => panic!("line number out of bounds"),
            },
            VisualBlock::Control((c, t)) => match n {
                0..=1 => {
                    if c < t {
                        String::from("           ")
                    } else {
                        String::from("     |     ")
                    }
                }
                3..=4 => {
                    if c < t {
                        String::from("     |     ")
                    } else {
                        String::from("           ")
                    }
                }
                2 => String::from("-----o-----"),
                _ => panic!("line number out of bounds"),
            },
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
                    grid[t].push(VisualBlock::Gate(String::from("    H    ")));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("    H    ")));
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
                    grid[t].push(VisualBlock::Gate(String::from("    X    ")));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("    X    ")));
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
                    grid[t].push(VisualBlock::Gate(String::from("    Y    ")));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("    Y    ")));
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
                    grid[t].push(VisualBlock::Gate(String::from("    Z    ")));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("    Z    ")));
                            occupied[i] = true;
                        } else {
                            grid[i].push(VisualBlock::HLine);
                            occupied[i] = false;
                        }
                    }
                }
            }
            Gate::PhaseShift((theta, t)) => {
                let theta_bounded = theta - (theta / (2.0 * PI)).trunc() * 2.0 * PI;
                if !occupied[t] {
                    grid[t].pop();
                    grid[t].push(VisualBlock::Gate(format!("P({:+.3})", theta_bounded)));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(format!("P({:+.3})", theta_bounded)));
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
                    grid[t].push(VisualBlock::Gate(String::from("    U    ")));
                    occupied[t] = true;
                } else {
                    for i in 0..nqubits {
                        if i == t {
                            grid[i].push(VisualBlock::Gate(String::from("    U    ")));
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
                            grid[i].push(VisualBlock::Gate(String::from("    X    ")));
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
                            grid[i].push(VisualBlock::Gate(String::from("    X    ")));
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
                            grid[i].push(VisualBlock::Gate(String::from("    Y    ")));
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
                            grid[i].push(VisualBlock::Gate(String::from("    Y    ")));
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
                            grid[i].push(VisualBlock::Gate(String::from("    Z    ")));
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
                            grid[i].push(VisualBlock::Gate(String::from("    Z    ")));
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
            Gate::CPhaseShift((theta, c, t)) => {
                let min: usize = cmp::min(t, c);
                let max: usize = cmp::max(t, c);
                let theta_bounded = theta - (theta / (2.0 * PI)).trunc() * 2.0 * PI;
                if occupied[min..(max + 1)].iter().all(|&o| o == false) {
                    for i in min..(max + 1) {
                        grid[i].pop();
                        if i == t {
                            grid[i].push(VisualBlock::Gate(format!("P({:+.3})", theta_bounded)));
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
                            grid[i].push(VisualBlock::Gate(format!("P({:+.3})", theta_bounded)));
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
                            grid[i].push(VisualBlock::Gate(String::from("    U    ")));
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
                            grid[i].push(VisualBlock::Gate(String::from("    U    ")));
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
        let max_len: usize = (nqubits - 1).to_string().len() + 3;
        let grid_len = grid[0].len();
        let mut k = 0;
        while k < grid_len {
            for i in 0..nqubits {
                let i_len: usize = i.to_string().len() + 3;
                for j in 0..5 {
                    let mut kk = k;
                    if j == 2 {
                        circuit.push_str(&format!("{}q{}---", " ".repeat(max_len - i_len), i));
                    } else {
                        circuit.push_str(&" ".repeat(max_len + 1));
                    }
                    while kk < k + 10 && kk < grid_len {
                        circuit.push_str(&grid[i][kk].get_line(j));
                        kk += 1;
                    }
                    if j == 2 {
                        circuit.push_str("----\n");
                    } else {
                        circuit.push_str("    \n");
                    }
                }
            }
            k += 10;
        }
        VisualCircuit { circuit: circuit }
    }
}
