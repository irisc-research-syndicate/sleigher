use std::collections::HashMap;
use anyhow::Result;

use sleigh_rs::{disassembly::{Assertation, Expr, ReadScope, VariableId}, pattern::{Block, ConstraintValue, Pattern, Verification}, Number, TokenFieldId};

use crate::{with_context, SleighConstructor, SleighContext, SleighSleigh, SleighTokenField, WithCtx};

with_context!(SleighPattern, SleighConstructor<'a>, Pattern, PatternContext, pattern);
with_context!(SleighReadScope, SleighSleigh<'a>, ReadScope, ReadScopeContext, readscope);
with_context!(SleighBlock, SleighPattern<'a>, Block, BlockContext, block);
with_context!(SleighExpr, SleighSleigh<'a>, Expr, ExprContext, expr);
with_context!(SleighAssertion, SleighBlock<'a>, Assertation, AssertionContext, assertion);
with_context!(SleighVerification, SleighBlock<'a>, Verification, VerificationContext, verification);
with_context!(SleighConstraintValue, SleighVerification<'a>, ConstraintValue, ConstraintValueContext, constraint_value);

fn number_to_i64(n: Number) -> i64 {
    match n {
        Number::Positive(pos) => pos as i64,
        Number::Negative(neg) => -(neg as i64),
    }
}

impl<'a> SleighReadScope<'a> {
    pub fn evaluate(&self, pc: u64, data: &[u8]) -> i64 {
        match self.inner {
            sleigh_rs::disassembly::ReadScope::Integer(number) => number_to_i64(*number),
            sleigh_rs::disassembly::ReadScope::TokenField(token_field_id) => {
                let token_field = self.sleigh().token_field(*token_field_id);
                let value = token_field.decode(data).value as i64;
                value
            },
            sleigh_rs::disassembly::ReadScope::InstStart(_) => pc as i64,
            sleigh_rs::disassembly::ReadScope::InstNext(_) => (pc + 4) as i64,
            _ => panic!("Unknown ReadScope while evaluating ReadScope: {:?}", self),
        }
    }
}


impl<'a> SleighExpr<'a> {
    pub fn evaluate(&self, pc: u64, data: &[u8]) -> i64 {
        match self.inner {
            Expr::Value(expr_element) => {
                match expr_element {
                    sleigh_rs::disassembly::ExprElement::Value { value, location: _ } => {
                        self.same_ctx(value).evaluate(pc, data)
                    },
                    sleigh_rs::disassembly::ExprElement::Op(_span, op_unary, expr) => {
                        let a = self.same_ctx(expr.as_ref()).evaluate(pc, data);
                        match op_unary {
                            sleigh_rs::disassembly::OpUnary::Negation => !a,
                            sleigh_rs::disassembly::OpUnary::Negative => -a,
                        }
                    },
                }
            },
            Expr::Op(_span, op, a, b) => {
                let a = self.same_ctx(a.as_ref()).evaluate(pc, data);
                let b = self.same_ctx(b.as_ref()).evaluate(pc, data);
                match op {
                    sleigh_rs::disassembly::Op::Add => a + b,
                    sleigh_rs::disassembly::Op::Sub => a - b,
                    sleigh_rs::disassembly::Op::Mul => a * b,
                    sleigh_rs::disassembly::Op::Div => a / b,
                    sleigh_rs::disassembly::Op::And => a & b,
                    sleigh_rs::disassembly::Op::Or => a | b,
                    sleigh_rs::disassembly::Op::Xor => a ^ b,
                    sleigh_rs::disassembly::Op::Asr => a >> b,
                    sleigh_rs::disassembly::Op::Lsl => a << b,
                }
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

    pub fn evaluate(&self, pc: u64, data: &[u8]) -> Result<HashMap<VariableId, i64>> {
        let mut variables = HashMap::new();

        for block in self.blocks() {
            for assertion in block.pre_disassembly() {
                match assertion.inner {
                    Assertation::GlobalSet(_global_set) => todo!(),
                    Assertation::Assignment(assignment) => {
                        let value = self.sleigh().self_ctx(&assignment.right).evaluate(pc, data);
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

    pub fn matches(&self, pc: u64, data: &[u8]) -> bool {
        for block in self.blocks() {
            for verification in block.verifications() {
                match verification.inner {
                    Verification::TableBuild { produced_table, verification: verification_cmp } => {
                        let table = self.sleigh().table(produced_table.table);
                        let _table_disassembled = match table.disassemble(pc, data) {
                            Ok(table_disassembled) => table_disassembled,
                            Err(_) => return false,
                        };
                        if let Some((_op, _value)) = verification_cmp {
                            log::warn!("")
                        }

                    },
                    Verification::TokenFieldCheck { field, op, value } => {
                        let token_field = self.sleigh().token_field(*field);
                        let token_field_decoded = token_field.decode(data);
                        let constraint_value = verification.self_ctx(value).evaluate(pc, data);
                        let check = match op {
                            sleigh_rs::pattern::CmpOp::Eq => token_field_decoded.value == constraint_value as usize,
                            sleigh_rs::pattern::CmpOp::Ne => token_field_decoded.value != constraint_value as usize,
                            op => panic!("pattern CmpOP {:?} not implemented", op)
                        };
                        if !check {
                            return false;
                        } 
                    },
                    verification => panic!("Verification {:?} not implemented", verification),
                };
            }
        }
        true
    }
}


impl<'a> SleighConstraintValue<'a> {
    fn expr(&self) -> SleighExpr {
        self.sleigh().self_ctx(self.inner.expr())
    }

    fn evaluate(&self, pc: u64, data: &[u8]) -> i64 {
        self.expr().evaluate(pc, data)
    }
}

impl<'a> SleighBlock<'a> {
    pub fn pre_disassembly(&self) -> impl Iterator<Item = SleighAssertion> {
        self.inner.pre_disassembler().iter().map(|assertion| self.self_ctx(assertion))
    }

    pub fn verifications(&self) -> impl Iterator<Item = SleighVerification> {
        self.inner.verifications().iter().map(|verification| self.self_ctx(verification))
    }
}