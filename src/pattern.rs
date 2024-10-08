use std::collections::HashMap;
use anyhow::{bail, Result};

use sleigh_rs::{disassembly::{Assertation, Expr, ReadScope, VariableId}, pattern::{Block, Pattern}, Number, TokenFieldId};

use crate::{with_context, SleighConstructor, SleighContext, SleighSleigh, SleighTokenField, WithCtx};

with_context!(SleighPattern, SleighConstructor<'a>, Pattern, PatternContext, pattern); // todo: ctx to Constructor

fn number_to_i64(n: Number) -> i64 {
    match n {
        Number::Positive(pos) => pos as i64,
        Number::Negative(neg) => -(neg as i64),
    }
}

with_context!(SleighReadScope, SleighSleigh<'a>, ReadScope, ReadScopeContext, readscope);

impl<'a> SleighReadScope<'a> {
    pub fn evaluate(&self, pc: u64, data: &[u8]) -> Result<i64> {
        Ok(match self.inner {
            sleigh_rs::disassembly::ReadScope::Integer(number) => number_to_i64(*number),
            sleigh_rs::disassembly::ReadScope::TokenField(token_field_id) => {
                let token_field = self.sleigh().token_field(*token_field_id);
                let value = token_field.decode(data).value as i64;
                //log::debug!("{}: {} {} {}", token_field.name(), value, token_field.raw_value_is_signed(), token_field.execution_value_is_signed());
                value
            },
            sleigh_rs::disassembly::ReadScope::InstStart(_) => pc as i64,
            sleigh_rs::disassembly::ReadScope::InstNext(_) => (pc + 4) as i64,
            //sleigh_rs::disassembly::ReadScope::Context(context_id) => todo!(),
            //sleigh_rs::disassembly::ReadScope::Local(variable_id) => todo!(),
            _ => bail!("Unknown ReadScope while evaluating ReadScope: {:?}", self),
        })
    }
}

type SleighExpr<'a> = WithCtx<'a, SleighSleigh<'a>, Expr>;

impl<'a> SleighExpr<'a> {
    pub fn evaluate(&self, pc: u64, data: &[u8]) -> Result<i64> {
        match self.inner {
            Expr::Value(expr_element) => {
                match expr_element {
                    sleigh_rs::disassembly::ExprElement::Value { value, location: _ } => {
                        self.same_ctx(value).evaluate(pc, data)
                    },
                    sleigh_rs::disassembly::ExprElement::Op(_span, op_unary, expr) => {
                        let a = self.same_ctx(expr.as_ref()).evaluate(pc, data)?;
                        Ok(match op_unary {
                            sleigh_rs::disassembly::OpUnary::Negation => !a,
                            sleigh_rs::disassembly::OpUnary::Negative => -a,
                        })
                    },
                }
            },
            Expr::Op(_span, op, a, b) => {
                let a = self.same_ctx(a.as_ref()).evaluate(pc, data)?;
                let b = self.same_ctx(b.as_ref()).evaluate(pc, data)?;
                Ok(match op {
                    sleigh_rs::disassembly::Op::Add => a + b,
                    sleigh_rs::disassembly::Op::Sub => a - b,
                    sleigh_rs::disassembly::Op::Mul => a * b,
                    sleigh_rs::disassembly::Op::Div => a / b,
                    sleigh_rs::disassembly::Op::And => a & b,
                    sleigh_rs::disassembly::Op::Or => a | b,
                    sleigh_rs::disassembly::Op::Xor => a ^ b,
                    sleigh_rs::disassembly::Op::Asr => a >> b,
                    sleigh_rs::disassembly::Op::Lsl => a << b,
                })
            },
        }
    }
}

impl<'a> SleighPattern<'a> {
    pub fn blocks(&self) -> impl Iterator<Item = SleighBlock> {
        self.inner.blocks().iter().map(|block| self.self_ctx(block))
    }

    pub fn token_field(&self, id: TokenFieldId) -> SleighTokenField {
        self.sleigh().token_field(id)
    }

    pub fn evaluate(&self, pc:u64, data: &[u8]) -> Result<HashMap<VariableId, i64>> {
        let mut variables = HashMap::new();

        for block in self.blocks() {
            for assertion in block.pre_disassembly() {
                match assertion.inner {
                    Assertation::GlobalSet(_global_set) => todo!(),
                    Assertation::Assignment(assignment) => {
                        let value = self.sleigh().self_ctx(&assignment.right).evaluate(pc, data)?;
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

with_context!(SleighBlock, SleighPattern<'a>, Block, BlockContext, block);
with_context!(SleighAssertion, SleighBlock<'a>, Assertation, AssertionContext, assertion);

impl<'a> SleighBlock<'a> {
    pub fn pre_disassembly(&self) -> impl Iterator<Item = SleighAssertion> {
        self.inner.pre_disassembler().iter().map(|assertion| self.self_ctx(assertion))
    }
}