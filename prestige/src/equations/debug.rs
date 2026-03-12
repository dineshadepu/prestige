use super::ir::EquationIR;

pub fn debug_equation(eq: &EquationIR) {

    println!("----------------------");
    println!("Equation : {}", eq.name);

    println!("Reads:");
    for r in &eq.reads {
        println!("  {}", r);
    }

    println!("Writes:");
    for w in &eq.writes {
        println!("  {}", w);
    }

    println!("Body:");
    println!("{}", eq.body);

    println!("----------------------");
}
