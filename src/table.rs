use std::collections::HashMap;

use sleigh_rs::{disassembly::VariableId, display::DisplayElement, table::{ConstructorId, Matcher, Table, VariantId}};
use anyhow::{Context, Result};

use crate::{SleighBitConstraints, SleighConstructor, SleighSleigh, Wrapper};

pub type SleighTable<'a> = Wrapper<'a, SleighSleigh<'a>, Table>;

impl<'a> SleighTable<'a> {
    pub fn disassemble(&self, data: &[u8]) -> Result<SleighDisassembledTable<'a>> {
        let (variant_id, constructor) = self
            .find_match(data)
            .context("Unable to find matching constructor")?;
        let variables = constructor
            .pattern()
            .evaluate(data)
            .context("Could not evaluate pattern")?;

        Ok(SleighDisassembledTable {
            sleigh: self.ctx.clone(),
            variant_id,
            constructor,
            data: data.to_vec(),
            variables: variables,
        })
    }

    pub fn matcher_order(&'a self) -> impl Iterator<Item = SleighMatcher<'a>> {
        self.inner.matcher_order().iter().map(|matcher| self.subs(matcher))
    }

    pub fn constructors(&'a self) -> impl Iterator<Item = SleighConstructor<'a>> {
        self.inner.constructors().iter().map(|constructor| self.wrap(constructor))
    }

    pub fn constructor(&self, id: ConstructorId) -> SleighConstructor<'a> {
        self.wrap(self.inner.constructor(id))
    }

    pub fn find_match(&self, data: &[u8]) -> Option<(VariantId, SleighConstructor<'a>)> {
        for matcher in self.matcher_order() {
            let constructor = self.constructor(matcher.inner.constructor);
            if matcher.matches(data) {
                return Some((matcher.inner.variant_id, constructor));
            }
        }
        None
    }

    pub fn name(&self) -> &str {
        self.inner.name()
    }
}

pub type SleighMatcher<'a> = Wrapper<'a, SleighTable<'a>, Matcher>;

impl<'a> SleighMatcher<'a> {
    pub fn constructor(&self) -> SleighConstructor<'a> {
        self.ctx.constructor(self.inner.constructor)
    }

    pub fn variant(&self) -> (SleighBitConstraints<'a>, SleighBitConstraints<'a>) {
        self.constructor().variant(self.inner.variant_id)
    }

    pub fn matches(&self, data: &[u8]) -> bool {
        self.variant().1.matches(data)
    }
}

#[derive(Debug)]
pub struct SleighDisassembledTable<'a> {
    pub sleigh: SleighSleigh<'a>,
    pub constructor: SleighConstructor<'a>,
    pub variant_id: VariantId,
    pub data: Vec<u8>,
    pub variables: HashMap<VariableId, i64>
}

impl<'a> std::fmt::Display for SleighDisassembledTable<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let display = &self.constructor.inner.display;
        let mut parts = display.mneumonic.iter().cloned().collect::<Vec<_>>();
        for elem in display.elements() {
            parts.push(match elem {
                DisplayElement::Varnode(_varnode_id) => "VARNODE".to_string(),
                DisplayElement::Context(_context_id) => "CONTEXT".to_string(),
                DisplayElement::TokenField(token_field_id) => {
                    let token_field = self.sleigh.token_field(*token_field_id);
                    let token_field_value = token_field.decode(&self.data);
                    format!("{}", token_field_value)
                },
                DisplayElement::InstStart(_inst_start) => "INST_START".to_string(),
                DisplayElement::InstNext(_inst_next) => "INST_NEXT".to_string(),
                DisplayElement::Table(table_id) => {
                    format!("{}", self.sleigh.table(*table_id).disassemble(&self.data).unwrap())
                },
                DisplayElement::Disassembly(variable_id) => {
                    self.variables.get(variable_id).map_or("UNKNOWN_DISASSEMBLY_VARIABLE".to_string(), |val| format!("{}", val))
                },
                DisplayElement::Literal(lit) => lit.to_string(),
                DisplayElement::Space => " ".to_string(),
            });
        }
        f.write_str(&parts.join(""))
    }
}