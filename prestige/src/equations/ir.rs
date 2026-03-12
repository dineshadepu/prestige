use proc_macro2::TokenStream;

#[derive(Clone)]
pub struct EquationIR {

    pub name: String,

    pub reads: Vec<String>,

    pub writes: Vec<String>,

    pub body: TokenStream,

}
