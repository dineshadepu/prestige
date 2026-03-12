pub mod equations;
pub mod codegen;

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

    use crate::equations::fuse::fuse;
    use crate::codegen::simple_cpu::generate_simple_cpu;

    #[test]
    fn test_fusion() {

        let eqs = vec![
            eq1::ir(),
        ];

        for eq in &eqs {
            debug_equation(eq);
        }

        let fused = fuse(eqs);

        println!("Fused Reads: {:?}", fused.reads);
        println!("Fused Writes: {:?}", fused.writes);

        let code = generate_simple_cpu(&fused);

        println!("Generated CPU Code:\n{}", code);

    }
}
