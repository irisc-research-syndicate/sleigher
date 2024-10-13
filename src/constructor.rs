use sleigh_rs::{display::DisplayElement, table::{Constructor, VariantId}};

use crate::{execution::SleighExecution, with_context, SleighBitConstraints, SleighContext, SleighPattern, SleighTable, WithCtx};

with_context!(SleighConstructor, SleighTable<'a>, Constructor, ConstructorContext, constructor); // todo: ctx to Table

impl<'a> SleighConstructor<'a> {
    pub fn variant(&self, variant_id: VariantId) -> (SleighBitConstraints<'a>, SleighBitConstraints<'a>) {
        let (a, b) = self.inner.variant(variant_id);
        (a.into(), b.into())
    }

    pub fn variants(&self) -> impl Iterator<Item = (VariantId, SleighBitConstraints<'a>, SleighBitConstraints<'a>)> {
        self.inner.variants().map(|(id, a , b)|
            (id, a.into(), b.into())
        )
    }

    pub fn pattern(&self) -> SleighPattern {
        self.self_ctx(&self.inner.pattern)
    }

    pub fn execution(&'a self) -> Option<SleighExecution<'a>> {
        self.inner.execution.as_ref().map(|exec| self.self_ctx(exec))
    }

    pub fn matches(&self, pc: u64, data: &[u8]) -> bool {
        self.pattern().matches(pc, data)
    }

}

impl<'a> std::fmt::Display for SleighConstructor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.inner.display.mneumonic.iter().map(|m| m.as_ref()).chain(
            self.inner.display.elements().map(|display_element| match display_element {
            DisplayElement::Varnode(varnode_id) => self.sleigh().inner.varnode(*varnode_id).name(),
            DisplayElement::Context(context_id) => self.sleigh().inner.context(*context_id).name(),
            DisplayElement::TokenField(token_field_id) => self.sleigh().inner.token_field(*token_field_id).name(),
            DisplayElement::InstStart(_inst_start) => "INST_START",
            DisplayElement::InstNext(_inst_next) => "INST_NEXT",
            DisplayElement::Table(table_id) => self.sleigh().inner.table(*table_id).name(),
            DisplayElement::Disassembly(variable_id) => self.inner.pattern.disassembly_var(*variable_id).name(),
            DisplayElement::Literal(lit) => &lit,
            DisplayElement::Space => " ",
        })).collect::<Vec<_>>().join(""))
    }
}