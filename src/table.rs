use std::collections::HashMap;

use sleigh_rs::{disassembly::VariableId, display::DisplayElement, table::{ConstructorId, Matcher, Table, VariantId}};
use anyhow::{Context, Result};

use crate::{with_context, SleighBitConstraints, SleighConstructor, SleighContext, SleighSleigh, WithCtx};

with_context!(SleighTable, SleighSleigh<'a>, Table, TableContext, table);
with_context!(SleighMatcher, SleighTable<'a>, Matcher, MatcherContext, matcher);

impl<'a> SleighTable<'a> {
    pub fn disassemble(&self, pc: u64, data: &[u8]) -> Result<SleighDisassembledTable> {
        let (variant_id, constructor) = self
            .find_match(pc, data)
            .context("Unable to find matching constructor")?;
        let variables = constructor
            .pattern()
            .evaluate(pc, data)
            .context("Could not evaluate pattern")?;

        Ok(SleighDisassembledTable {
            pc,
            constructor,
            variant_id,
            data: data.to_vec(),
            variables: variables,
        })
    }

    pub fn matcher_order(&self) -> impl Iterator<Item = SleighMatcher> {
        self.inner.matcher_order().iter().map(|matcher| self.self_ctx(matcher))
    }

    pub fn constructors(&self) -> impl Iterator<Item = SleighConstructor> {
        self.inner.constructors().iter().map(|constructor| self.self_ctx(constructor))
    }

    pub fn constructor(&self, id: ConstructorId) -> SleighConstructor {
        self.self_ctx(self.inner.constructor(id))
    }

    pub fn find_match(&self, pc: u64, data: &[u8]) -> Option<(VariantId, SleighConstructor)> {
        for matcher in self.matcher_order() {
            if matcher.matches(data) {
                let constructor = self.constructor(matcher.inner.constructor);
                if constructor.matches(pc, data) {
                    return Some((matcher.inner.variant_id, constructor));
                }
            }
        }
        None
    }

    pub fn name(&self) -> &str {
        self.inner.name()
    }
}


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
    pub pc: u64,
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
                    let token_field = self.constructor.sleigh().token_field(*token_field_id);
                    let token_field_value = token_field.decode(&self.data);
                    format!("{}", token_field_value)
                },
                DisplayElement::InstStart(_inst_start) => "INST_START".to_string(),
                DisplayElement::InstNext(_inst_next) => "INST_NEXT".to_string(),
                DisplayElement::Table(table_id) => {
                    format!("{}", self.constructor.sleigh().table(*table_id).disassemble(self.pc, &self.data).unwrap())
                },
                DisplayElement::Disassembly(variable_id) => {
                    self.variables.get(&variable_id).map_or("UNKNOWN_DISASSEMBLY_VARIABLE".to_string(), |val| format!("{}", val))
                },
                DisplayElement::Literal(lit) => lit.to_string(),
                DisplayElement::Space => " ".to_string(),
            });
        }
        f.write_str(&parts.join(""))
    }
}