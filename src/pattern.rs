use std::collections::HashMap;
use anyhow::{anyhow, bail, ensure, Result};

use sleigh_rs::{disassembly::{Assertation, VariableId}, pattern::{Block, Pattern}, Number, TokenFieldId};

use crate::{SleighSleigh, SleighTokenField, Wrapper};


pub type SleighPattern<'a> = Wrapper<'a, SleighSleigh<'a>, Pattern>;

fn number_to_i64(n: Number) -> i64 {
    match n {
        Number::Positive(pos) => pos as i64,
        Number::Negative(neg) => -(neg as i64),
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

pub type SleighBlock<'a> = Wrapper<'a, SleighPattern<'a>, Block>;
pub type SleighAssertion<'a> = Wrapper<'a, SleighBlock<'a>, Assertation>;

impl<'a> SleighBlock<'a> {
    pub fn pre_disassembly(&self) -> impl Iterator<Item = SleighAssertion> {
        self.inner.pre_disassembler().iter().map(|assertion| self.subs(assertion))
    }
}