
use std::collections::HashMap;

use sleigh_rs::{disassembly::{Assertation, VariableId}, display::DisplayElement, pattern::{BitConstraint, Block, Pattern}, table::{Constructor, ConstructorId, Matcher, Table, VariantId}, token::{Token, TokenField}, Endian, Number, Sleigh, TableId, TokenFieldId};
use anyhow::{anyhow, bail, ensure, Context, Result};


#[derive(Clone, Copy)]
pub struct Wrapper<'a, Ctx, T: ?Sized> {
    pub ctx: &'a Ctx,
    pub inner: &'a T
}

impl<'a, Ctx, T: ?Sized> Wrapper<'a, Ctx, T> {
    pub fn new(ctx: &'a Ctx, inner: &'a T) -> Self {
        Wrapper {
            ctx,
            inner,
        }
    }

    pub fn wrap<N>(&self, inner: &'a N) -> Wrapper<'a, Ctx, N> {
        Wrapper {
            ctx: self.ctx,
            inner,
        }
    }

    pub fn subs<N>(&'a self, inner: &'a N) -> Wrapper<'a, Self, N> {
        Wrapper {
            ctx: self,
            inner
        }
    }
}

impl<'a, T: ?Sized> From<&'a T> for Wrapper<'a, (), T> {
    fn from(value: &'a T) -> Self {
        Wrapper::new(&(), value)
    }
}

impl<'a, Ctx, T: std::fmt::Debug + ?Sized> std::fmt::Debug for Wrapper<'a, Ctx, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.inner, f)
    }
}

pub type SleighSleigh<'a> = Wrapper<'a, (), Sleigh>;
pub type SleighTable<'a> = Wrapper<'a, SleighSleigh<'a>, Table>;
pub type SleighMatcher<'a> = Wrapper<'a, SleighTable<'a>, Matcher>;
pub type SleighConstructor<'a> = Wrapper<'a, SleighSleigh<'a>, Constructor>;
pub type SleighBitConstraints<'a> = Wrapper<'a, (), [BitConstraint]>;
pub type SleighToken<'a> = Wrapper<'a, SleighSleigh<'a>, Token>;
pub type SleighPattern<'a> = Wrapper<'a, SleighSleigh<'a>, Pattern>;
pub type SleighBlock<'a> = Wrapper<'a, SleighPattern<'a>, Block>;
pub type SleighAssertion<'a> = Wrapper<'a, SleighBlock<'a>, Assertation>;
pub type SleighTokenField<'a> = Wrapper<'a, SleighSleigh<'a>, TokenField>;

fn number_to_i64(n: Number) -> i64 {
    match n {
        Number::Positive(pos) => pos as i64,
        Number::Negative(neg) => -(neg as i64),
    }
}

impl<'a> SleighToken<'a> {
    pub fn decode(&self, data: &[u8]) -> usize {
        match (self.inner.endian, self.inner.len_bytes().get()) {
            (Endian::Big, 4) => u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize,
            _ => todo!()
        }
    }
}

pub struct SleighDecodedTokenField<'a> {
    pub token_field: SleighTokenField<'a>,
    pub value: usize,
}

impl<'a> std::fmt::Display for SleighDecodedTokenField<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&match self.token_field.inner.meaning() {
            sleigh_rs::meaning::Meaning::NoAttach(value_fmt) =>
                format!("{}", self.value),
            sleigh_rs::meaning::Meaning::Varnode(attach_varnode_id) => {
                let attach_varnode = self.token_field.ctx.inner.attach_varnode(attach_varnode_id);
                let varnode_id = attach_varnode.find_value(self.value).unwrap();
                self.token_field.ctx.inner.varnode(varnode_id).name().to_string()
            }

            sleigh_rs::meaning::Meaning::Literal(attach_literal_id) => todo!(),
            sleigh_rs::meaning::Meaning::Number(print_base, attach_number_id) => todo!(),
        })
    }
}

impl<'a> SleighTokenField<'a> {
    pub fn token(&self) -> SleighToken<'a> {
        self.wrap(self.ctx.inner.token(self.inner.token))
    }

    pub fn decode(&self, data: &[u8]) -> SleighDecodedTokenField {
        let value = (self.token().decode(data) >> self.inner.bits.start()) & ((1 << self.inner.bits.len().get()) - 1);
        SleighDecodedTokenField {
            token_field: self.clone(),
            value: value
        }
    }
}

impl<'a> SleighPattern<'a> {
    pub fn blocks(&self) -> impl Iterator<Item = SleighBlock> {
        self.inner.blocks().iter().map(|block| self.subs(block))
    }

    pub fn token_field(&self, id: TokenFieldId) -> SleighTokenField<'a> {
        self.wrap(self.ctx.inner.token_field(id))
    }

    pub fn evaluate(&self, data: &[u8]) -> Result<HashMap<VariableId, i64>> {
        let mut variables = HashMap::new();

        for block in self.blocks() {
            for assertion in block.pre_disassembly() {
                match assertion.inner {
                    Assertation::GlobalSet(_global_set) => todo!(),
                    Assertation::Assignment(assignment) => {
                        let mut stack: Vec<i64> = vec![];
                        for expr in assignment.right.elements() {
                            let value = match expr {
                                sleigh_rs::disassembly::ExprElement::Value { value, location: _ } => {
                                    match value {
                                        sleigh_rs::disassembly::ReadScope::Integer(number) => number_to_i64(*number),
                                        sleigh_rs::disassembly::ReadScope::TokenField(token_field_id) => {
                                            self.token_field(*token_field_id).decode(data).value as i64
                                        },
                                        sleigh_rs::disassembly::ReadScope::InstStart(_) => 0i64,
                                        sleigh_rs::disassembly::ReadScope::InstNext(_) => 0i64,
                                        //sleigh_rs::disassembly::ReadScope::Context(context_id) => todo!(),
                                        //sleigh_rs::disassembly::ReadScope::Local(variable_id) => todo!(),
                                        
                                        _ => bail!("Unknown ReadScope while evaluating Assertion: {:?}", value),
                                    }
                                },
                                sleigh_rs::disassembly::ExprElement::Op(op) => {
                                    let a = stack.pop().ok_or(anyhow!("Stack underrun: {:?}", op))?;
                                    let b = stack.pop().ok_or(anyhow!("Stack underrun: {:?}", op))?;
                                    match op {
                                        sleigh_rs::disassembly::Op::Add => a + b,
                                        sleigh_rs::disassembly::Op::Sub => a - b, // ??
                                        sleigh_rs::disassembly::Op::Mul => a * b,
                                        sleigh_rs::disassembly::Op::Div => a / b, // ??
                                        sleigh_rs::disassembly::Op::And => a & b,
                                        sleigh_rs::disassembly::Op::Or => a | b,
                                        sleigh_rs::disassembly::Op::Xor => a ^ b,
                                        sleigh_rs::disassembly::Op::Asr => todo!(),
                                        sleigh_rs::disassembly::Op::Lsl => todo!(),
                                    }
                                },
                                sleigh_rs::disassembly::ExprElement::OpUnary(op) => {
                                    let a = stack.pop().ok_or(anyhow!("Stack underrun: {:?}", op))?;
                                    match op {
                                        sleigh_rs::disassembly::OpUnary::Negation => !a,
                                        sleigh_rs::disassembly::OpUnary::Negative => -a,
                                    }
                                },
                            };
                            stack.push(value);
                        }
                        ensure!(stack.len() != 0, "Stack empty after evaluation");
                        ensure!(stack.len() == 1, "Stack has more than one element after evaluation");
                        let value = stack.pop().unwrap();
                        match assignment.left {
                            sleigh_rs::disassembly::WriteScope::Context(_context_id) => todo!(),
                            sleigh_rs::disassembly::WriteScope::Local(variable_id) => variables.insert(variable_id, value),
                        };
                    },
                }
            }
        }

        Ok(variables)
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
                DisplayElement::Varnode(varnode_id) => "VARNODE".to_string(),
                DisplayElement::Context(context_id) => "CONTEXT".to_string(),
                DisplayElement::TokenField(token_field_id) => {
                    let token_field = self.sleigh.token_field(*token_field_id);
                    let token_field_value = token_field.decode(&self.data);
                    format!("{}", token_field_value)
                },
                DisplayElement::InstStart(inst_start) => "INST_START".to_string(),
                DisplayElement::InstNext(inst_next) => "INST_NEXT".to_string(),
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
}

impl<'a> SleighSleigh<'a> {
    pub fn disassemble_instruction(&'a self, data: &[u8]) -> Result<SleighDisassembledTable<'a>> {
        self.instruction_table().disassemble(data)
    }
}

impl<'a> SleighBlock<'a> {
    pub fn pre_disassembly(&self) -> impl Iterator<Item = SleighAssertion> {
        self.inner.pre_disassembler().iter().map(|assertion| self.subs(assertion))
    }
}


impl<'a> SleighSleigh<'a> {
    pub fn tables(&self) -> impl Iterator<Item = SleighTable> {
        self.inner.tables().iter().map(|table| self.subs(table))
    }

    pub fn table(&self, id: TableId) -> SleighTable {
        self.subs(self.inner.table(id))
    }

    pub fn instruction_table(&self) -> SleighTable {
        self.table(self.inner.instruction_table())
    }

    pub fn token_field(&self, id: TokenFieldId) -> SleighTokenField {
        self.subs(self.inner.token_field(id))
    }
}

impl<'a> SleighTable<'a> {
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

fn get_bit(data: &[u8], idx: usize) -> Option<bool> {
    if idx > data.len() * 8 {
        return None;
    }
    Some((data[idx >> 3] >> (idx & 7)) & 1 == 1)
}

impl<'a> SleighBitConstraints<'a> {
    pub fn matches(&self, data: &[u8]) -> bool {
        self.inner.iter().enumerate().all(|(idx, bit)| 
            bit.value().map_or(true, |expected|
                get_bit(data, idx).is_some_and(|actual|
                    expected == actual
                )
            )
        )
    }
}

impl<'a> std::fmt::Display for SleighBitConstraints<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.inner.chunks(8).map(|bits| bits.iter().rev().map(|bit| match bit {
            BitConstraint::Unrestrained => "x",
            BitConstraint::Defined(false) => "0",
            BitConstraint::Defined(true) => "1",
            BitConstraint::Restrained => "X",
        }).collect()).collect::<Vec<String>>().join("_"))
    }
}