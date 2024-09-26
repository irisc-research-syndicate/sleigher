use sleigh_rs::{display::DisplayElement, table::{Constructor, VariantId}};

use crate::{SleighBitConstraints, SleighPattern, SleighSleigh, Wrapper};

pub type SleighConstructor<'a> = Wrapper<'a, SleighSleigh<'a>, Constructor>;

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
        self.wrap(&self.inner.pattern)
    }
}

impl<'a> std::fmt::Display for SleighConstructor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.inner.display.mneumonic.iter().map(|m| m.as_ref()).chain(
            self.inner.display.elements().map(|display_element| match display_element {
            DisplayElement::Varnode(varnode_id) => self.ctx.inner.varnode(*varnode_id).name(),
            DisplayElement::Context(context_id) => self.ctx.inner.context(*context_id).name(),
            DisplayElement::TokenField(token_field_id) => self.ctx.inner.token_field(*token_field_id).name(),
            DisplayElement::InstStart(_inst_start) => "INST_START",
            DisplayElement::InstNext(_inst_next) => "INST_NEXT",
            DisplayElement::Table(table_id) => self.ctx.inner.table(*table_id).name(),
            DisplayElement::Disassembly(variable_id) => self.inner.pattern.disassembly_var(*variable_id).name(),
            DisplayElement::Literal(lit) => lit,
            DisplayElement::Space => " ",
        })).collect::<Vec<_>>().join(""))
    }
}