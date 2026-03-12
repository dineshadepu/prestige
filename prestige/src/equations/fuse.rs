use super::ir::EquationIR;
use std::collections::HashSet;

pub struct FusedEquations {

    pub reads: Vec<String>,

    pub writes: Vec<String>,

    pub bodies: Vec<String>,

}

pub fn fuse(eqs: Vec<EquationIR>) -> FusedEquations {

    let mut reads = HashSet::new();
    let mut writes = HashSet::new();
    let mut bodies = Vec::new();

    for eq in eqs {

        for r in eq.reads {
            reads.insert(r);
        }

        for w in eq.writes {
            writes.insert(w);
        }

        bodies.push(eq.body.to_string());
    }

    FusedEquations {

        reads: reads.into_iter().collect(),
        writes: writes.into_iter().collect(),
        bodies,

    }
}
