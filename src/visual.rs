use crate::circuit::*;
use std::cmp;
use std::collections::VecDeque;

#[derive(Clone)]
pub enum VisualBlock {
    Gate(String),
    HLine(String),
    VLine(String),
    Control(String),
}

impl VisualBlock {
    pub fn get_first_line(&self) -> String {
        match self {
            VisualBlock::Gate(s) => String::from(&s[0..3]),
            VisualBlock::HLine(s) => String::from(&s[0..3]),
            VisualBlock::VLine(s) => String::from(&s[0..3]),
            VisualBlock::Control(s) => String::from(&s[0..3]),
        }
    }

    pub fn get_second_line(&self) -> String {
        match self {
            VisualBlock::Gate(s) => String::from(&s[3..6]),
            VisualBlock::HLine(s) => String::from(&s[3..6]),
            VisualBlock::VLine(s) => String::from(&s[3..6]),
            VisualBlock::Control(s) => String::from(&s[3..6]),
        }
    }

    pub fn get_third_line(&self) -> String {
        match self {
            VisualBlock::Gate(s) => String::from(&s[6..9]),
            VisualBlock::HLine(s) => String::from(&s[6..9]),
            VisualBlock::VLine(s) => String::from(&s[6..9]),
            VisualBlock::Control(s) => String::from(&s[6..9]),
        }
    }
}

pub struct VisualCircuit {
    pub circuit: String,
}

impl VisualCircuit {
    pub fn new(nqubits: usize, gates: &VecDeque<Gate>) -> Self {
        let mut grid: Vec<Vec<VisualBlock>> = vec![Vec::new(); nqubits];
        gates.iter().for_each(|gate| match *gate {
            Gate::Hadamard(t) => {
                for i in 0..nqubits {
                    if i == t {
                        grid[i].push(VisualBlock::Gate(String::from(" - |H| - ")));
                    } else {
                        grid[i].push(VisualBlock::Gate(String::from("   ---   ")));
                    }
                }
            }
            Gate::PauliX(t) => {
                for i in 0..nqubits {
                    if i == t {
                        grid[i].push(VisualBlock::Gate(String::from(" - |X| - ")));
                    } else {
                        grid[i].push(VisualBlock::Gate(String::from("   ---   ")));
                    }
                }
            }
            Gate::PauliY(t) => {
                for i in 0..nqubits {
                    if i == t {
                        grid[i].push(VisualBlock::Gate(String::from(" - |Y| - ")));
                    } else {
                        grid[i].push(VisualBlock::Gate(String::from("   ---   ")));
                    }
                }
            }
            Gate::PauliZ(t) => {
                for i in 0..nqubits {
                    if i == t {
                        grid[i].push(VisualBlock::Gate(String::from(" - |Z| - ")));
                    } else {
                        grid[i].push(VisualBlock::Gate(String::from("   ---   ")));
                    }
                }
            }
            Gate::PhaseShift((_, t)) => {
                for i in 0..nqubits {
                    if i == t {
                        grid[i].push(VisualBlock::Gate(String::from(" - |P| - ")));
                    } else {
                        grid[i].push(VisualBlock::Gate(String::from("   ---   ")));
                    }
                }
            }
            Gate::Unitary((_, _, _, _, t)) => {
                for i in 0..nqubits {
                    if i == t {
                        grid[i].push(VisualBlock::Gate(String::from(" - |U| - ")));
                    } else {
                        grid[i].push(VisualBlock::Gate(String::from("   ---   ")));
                    }
                }
            }
            Gate::CPauliX((c, t)) => {
                let min: usize = cmp::min(t, c);
                let max: usize = cmp::max(t, c);
                for i in 0..nqubits {
                    if i == t {
                        grid[i].push(VisualBlock::Gate(String::from(" - |X| - ")));
                    } else if i == c {
                        if c < t {
                            grid[i].push(VisualBlock::Gate(String::from("   -o- | ")));
                        } else {
                            grid[i].push(VisualBlock::Gate(String::from("| -o-   ")));
                        }
                    } else if i < max && i > min {
                        grid[i].push(VisualBlock::Gate(String::from(" |  |  | ")));
                    } else {
                        grid[i].push(VisualBlock::Gate(String::from("   ---   ")));
                    }
                }
            }
            Gate::CPauliY((c, t)) => {
                let min: usize = cmp::min(t, c);
                let max: usize = cmp::max(t, c);
                for i in 0..nqubits {
                    if i == t {
                        grid[i].push(VisualBlock::Gate(String::from(" - |Y| - ")));
                    } else if i == c {
                        if c < t {
                            grid[i].push(VisualBlock::Gate(String::from("   -o- | ")));
                        } else {
                            grid[i].push(VisualBlock::Gate(String::from("| -o-   ")));
                        }
                    } else if i < max && i > min {
                        grid[i].push(VisualBlock::Gate(String::from(" |  |  | ")));
                    } else {
                        grid[i].push(VisualBlock::Gate(String::from("   ---   ")));
                    }
                }
            }
            Gate::CPauliZ((c, t)) => {
                let min: usize = cmp::min(t, c);
                let max: usize = cmp::max(t, c);
                for i in 0..nqubits {
                    if i == t {
                        grid[i].push(VisualBlock::Gate(String::from(" - |Z| - ")));
                    } else if i == c {
                        if c < t {
                            grid[i].push(VisualBlock::Gate(String::from("   -o- | ")));
                        } else {
                            grid[i].push(VisualBlock::Gate(String::from("| -o-   ")));
                        }
                    } else if i < max && i > min {
                        grid[i].push(VisualBlock::Gate(String::from(" |  |  | ")));
                    } else {
                        grid[i].push(VisualBlock::Gate(String::from("   ---   ")));
                    }
                }
            }
            Gate::CPhaseShift((_, c, t)) => {
                let min: usize = cmp::min(t, c);
                let max: usize = cmp::max(t, c);
                for i in 0..nqubits {
                    if i == t {
                        grid[i].push(VisualBlock::Gate(String::from(" - |P| - ")));
                    } else if i == c {
                        if c < t {
                            grid[i].push(VisualBlock::Gate(String::from("   -o- | ")));
                        } else {
                            grid[i].push(VisualBlock::Gate(String::from(" | -o-   ")));
                        }
                    } else if i < max && i > min {
                        grid[i].push(VisualBlock::Gate(String::from(" |  |  | ")));
                    } else {
                        grid[i].push(VisualBlock::Gate(String::from("   ---   ")));
                    }
                }
            }
            Gate::CUnitary((_, _, _, _, c, t)) => {
                let min: usize = cmp::min(t, c);
                let max: usize = cmp::max(t, c);
                for i in 0..nqubits {
                    if i == t {
                        grid[i].push(VisualBlock::Gate(String::from(" - |U| - ")));
                    } else if i == c {
                        if c < t {
                            grid[i].push(VisualBlock::Gate(String::from("   -o- | ")));
                        } else {
                            grid[i].push(VisualBlock::Gate(String::from("| -o-   ")));
                        }
                    } else if i < max && i > min {
                        grid[i].push(VisualBlock::Gate(String::from(" |  |  | ")));
                    } else {
                        grid[i].push(VisualBlock::Gate(String::from("   ---   ")));
                    }
                }
            }
        });

        let mut circuit: String = String::new();
        for i in 0..nqubits {
            grid[i]
                .iter()
                .for_each(|g| circuit.push_str(&g.get_first_line()));
            circuit.push_str("\n");

            grid[i]
                .iter()
                .for_each(|g| circuit.push_str(&g.get_second_line()));
            circuit.push_str("\n");

            grid[i]
                .iter()
                .for_each(|g| circuit.push_str(&g.get_third_line()));
            circuit.push_str("\n");
        }
        VisualCircuit { circuit: circuit }
    }
}
