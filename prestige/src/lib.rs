pub mod equations;

use prestige_macros::equation;
use crate::equations::ir;

#[equation]
fn eq1(i: usize, j: usize, force: &mut [f64], mass: &[f64]) {

    force[i] += mass[j];

}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::equations::debug::debug_equation;

    #[test]
    fn test_equation_ir() {

        let ir = eq1::ir();

        debug_equation(&ir);

    }

}
