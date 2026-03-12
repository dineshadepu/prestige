use crate::equations::fuse::FusedEquations;

pub fn generate_simple_cpu(ir: &FusedEquations) -> String {

    let mut code = String::new();

    code.push_str("for i in 0..n {\n");
    code.push_str("    for j in 0..n {\n");

    for b in &ir.bodies {

        code.push_str("        ");
        code.push_str(b);
        code.push_str("\n");

    }

    code.push_str("    }\n");
    code.push_str("}\n");

    code
}
