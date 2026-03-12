use proc_macro::TokenStream;
use quote::quote;
use std::collections::HashSet;
use syn::visit::Visit;
use syn::{parse_macro_input, ItemFn};
use syn::{Expr, ExprIndex, ExprBinary, BinOp};

#[proc_macro_attribute]
pub fn equation(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    let name = &input.sig.ident;
    let body = &input.block;

    // ------------------------------------------------
    // Detect slice reads / writes
    // ------------------------------------------------

    let mut visitor = SliceVisitor {
        reads: HashSet::new(),
        writes: HashSet::new(),
    };

    visitor.visit_block(body);

    let reads: Vec<String> = visitor.reads.into_iter().collect();
    let writes: Vec<String> = visitor.writes.into_iter().collect();

    let reads_tokens = reads.iter().map(|s| quote! { #s.to_string() });
    let writes_tokens = writes.iter().map(|s| quote! { #s.to_string() });

    // ------------------------------------------------
    // Generate expanded code
    // ------------------------------------------------

    let expanded = quote! {

        pub struct #name;

        impl #name {

            pub fn ir() -> crate::equations::ir::EquationIR {

                crate::equations::ir::EquationIR{
                    name: stringify!(#name).to_string(),
                    reads: vec![#(#reads_tokens),*],
                    writes: vec![#(#writes_tokens),*],
                    body: quote::quote!(#body),
                }

            }

        }

    };

    TokenStream::from(expanded)
}

// ------------------------------------------------
// AST visitor
// ------------------------------------------------

struct SliceVisitor {
    reads: HashSet<String>,
    writes: HashSet<String>,
}
impl<'ast> Visit<'ast> for SliceVisitor {

    // detect: force[i] = ...
    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {

        if let Expr::Index(idx) = &*node.left {

            if let Expr::Path(path) = &*idx.expr {

                let name = path.path.segments[0].ident.to_string();
                self.writes.insert(name);

            }

        }

        syn::visit::visit_expr_assign(self, node);
    }

    // detect: force[i] += ...
    fn visit_expr_binary(&mut self, node: &'ast ExprBinary) {

        match node.op {

            BinOp::AddAssign(_)
            | BinOp::SubAssign(_)
            | BinOp::MulAssign(_)
            | BinOp::DivAssign(_)
            | BinOp::RemAssign(_)
            | BinOp::BitXorAssign(_)
            | BinOp::BitAndAssign(_)
            | BinOp::BitOrAssign(_)
            | BinOp::ShlAssign(_)
            | BinOp::ShrAssign(_) => {

                if let Expr::Index(idx) = &*node.left {

                    if let Expr::Path(path) = &*idx.expr {

                        let name = path.path.segments[0].ident.to_string();
                        self.writes.insert(name);

                    }

                }

            }

            _ => {}
        }

        syn::visit::visit_expr_binary(self, node);
    }

    // detect reads: mass[j]
    fn visit_expr_index(&mut self, node: &'ast ExprIndex) {

        if let Expr::Path(path) = &*node.expr {

            let name = path.path.segments[0].ident.to_string();
            self.reads.insert(name);

        }

        syn::visit::visit_expr_index(self, node);
    }

}
