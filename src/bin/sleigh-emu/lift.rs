use std::collections::HashMap;
use std::ops::Deref;
use std::{cell::RefCell, rc::Rc};

use anyhow::{bail, Context, Result};
use sleigh_rs::execution::{BlockId, VariableId};
use sleigh_rs::SpaceId;
use sleigher::value::{Value, Var};
use sleigher::{SleighContext, SleighTable};
use sleigher::execution::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarId(usize);

#[derive(Debug, Clone)]
pub struct Locals {
    next_local: usize,
    variables: HashMap<VariableId, VarId>,
}

impl Locals {
    pub fn alloc_new(&mut self) -> VarId {
        let var = VarId(self.next_local);
        self.next_local += 1;
        var
    }

    pub fn get_var(&mut self, var: VariableId) -> VarId {
        match self.variables.entry(var) {
            std::collections::hash_map::Entry::Occupied(occupied_entry) => occupied_entry.get().clone(),
            std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                let var = VarId(self.next_local);
                self.next_local += 1;
                vacant_entry.insert(var);
                var
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LifterInner {
    pub locals: Locals,
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct Lifter(Rc<RefCell<LifterInner>>);

impl Deref for Lifter {
    type Target = Rc<RefCell<LifterInner>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct TableLifter {
    pub lifter: Lifter,
    pub export: Option<Value>,
}

impl std::ops::Deref for TableLifter {
    type Target = Lifter;

    fn deref(&self) -> &Self::Target {
        &self.lifter
    }
}

impl Lifter {
    pub fn alloc_var(&self) -> VarId {
        self.borrow_mut().locals.alloc_new()
    }

    pub fn get_var(&self, var: VariableId) -> VarId {
        self.borrow_mut().locals.get_var(var)

    }

    pub fn push_stmt(&self, stmt: Statement) {
        self.borrow_mut().statements.push(stmt)
    }

    pub fn lift(&self, table: SleighTable, pc: u64, data: &[u8]) -> Result<(Vec<Statement>, Option<Value>)> {
        let (variant_id, constructor) = table.find_match(pc, data).context("Table did not match")?;
        let execution = constructor.execution().context("construction have no execution")?;

        let table_lifter = TableLifter {
            lifter: self.clone(),
            export: None
        };

        let mut current_block = Some(execution.entry_block());
        while let Some(block) = current_block.take() {
            let next_block_id = table_lifter.lift_block(block)?;
            current_block = next_block_id.map(|block_id| execution.block(block_id));
        }
        todo!();
    }
}

impl TableLifter {
    pub fn lift_block(&self, block: SleighBlock) -> Result<Option<BlockId>> {
        for stmt in block.statements() {
            let effect= self.lift_statement(stmt)?;
            match effect {
                StatementEffect::Continue => continue,
                StatementEffect::Break => break,
                StatementEffect::Branch(block_id) => return Ok(Some(block_id)),
            }
        }
        Ok(None)
    }

    pub fn lift_statement(&self, stmt: SleighStatement) -> Result<StatementEffect> {
        match stmt.inner {
            // sleigh_rs::execution::Statement::Delayslot(_) => todo!(),
            sleigh_rs::execution::Statement::Export(export) => self.lift_export(stmt.self_ctx(export)),
            sleigh_rs::execution::Statement::CpuBranch(cpu_branch) => self.lift_cpu_branch(stmt.self_ctx(cpu_branch)),
            sleigh_rs::execution::Statement::LocalGoto(local_goto) => self.lift_local_goto(stmt.self_ctx(local_goto)),
            // sleigh_rs::execution::Statement::UserCall(user_call) => todo!(),
            sleigh_rs::execution::Statement::Build(build) => self.lift_build(stmt.self_ctx(build)),
            // sleigh_rs::execution::Statement::Declare(variable_id) => todo!(),
            sleigh_rs::execution::Statement::Assignment(assignment) => self.lift_assignment(stmt.self_ctx(assignment)),
            sleigh_rs::execution::Statement::MemWrite(mem_write) => self.lift_mem_write(stmt.self_ctx(mem_write)),
            stmt => bail!("Statement not implemented: {:?}", stmt)
        }
    }

    pub fn lift_export(&self, local_goto: SleighExport) -> Result<StatementEffect> {
        todo!()
    }

    pub fn lift_cpu_branch(&self, local_goto: SleighCpuBranch) -> Result<StatementEffect> {
        todo!()
    }

    pub fn lift_local_goto(&self, local_goto: SleighLocalGoto) -> Result<StatementEffect> {
        todo!()
    }

    pub fn lift_build(&self, build: SleighBuild) -> Result<StatementEffect> {
        todo!()
    }

    pub fn lift_assignment(&self, assign: SleighAssignment) -> Result<StatementEffect> {
        let right = self.lift_expr(assign.statement().self_ctx(&assign.inner.right))?;
        match &assign.inner.var {
            sleigh_rs::execution::WriteValue::Varnode(write_varnode) => {
                let varnode_ref = assign.sleigh().varnode(write_varnode.id).referance();
            },
            sleigh_rs::execution::WriteValue::Bitrange(write_bitrange) => todo!(),
            sleigh_rs::execution::WriteValue::TokenField(write_token_field) => todo!(),
            sleigh_rs::execution::WriteValue::TableExport(write_table) => todo!(),
            sleigh_rs::execution::WriteValue::Local(write_exe_var) => {
                let var = self.get_var(write_exe_var.id);
            },
        }
        todo!()
    }

    pub fn lift_mem_write(&self, mem_write: SleighMemWrite) -> Result<StatementEffect> {
        todo!()
    }

    pub fn lift_expr(&self, expr: SleighExpr) -> Result<VarId> {
        match expr.inner {
            sleigh_rs::execution::Expr::Value(expr_element) => match expr_element {
                sleigh_rs::execution::ExprElement::Value(expr_value) => self.lift_value_expr(expr.self_ctx(expr_value)),
                sleigh_rs::execution::ExprElement::UserCall(user_call) => todo!(),
                sleigh_rs::execution::ExprElement::Reference(reference) => todo!(),
                sleigh_rs::execution::ExprElement::Op(expr_unary_op) => self.lift_unary_expr(expr.self_ctx(expr_unary_op)),
                sleigh_rs::execution::ExprElement::New(expr_new) => todo!(),
                sleigh_rs::execution::ExprElement::CPool(expr_cpool) => todo!(),
            } ,
            sleigh_rs::execution::Expr::Op(expr_binary_op) => self.lift_binary_expr(expr.self_ctx(expr_binary_op)),
        }
    }

    pub fn lift_value_expr(&self, expr_value: SleighExprValue) -> Result<VarId> {
        let value = match expr_value.inner {
            sleigh_rs::execution::ExprValue::Int(expr_number) => {
                Value::Int(match expr_number.number {
                    sleigh_rs::Number::Positive(x) => x,
                    sleigh_rs::Number::Negative(x) => -(x as i64) as u64,
                })
            },
            sleigh_rs::execution::ExprValue::TokenField(expr_token_field) => todo!(),
            sleigh_rs::execution::ExprValue::InstStart(expr_inst_start) => todo!(),
            sleigh_rs::execution::ExprValue::InstNext(expr_inst_next) => todo!(),
            sleigh_rs::execution::ExprValue::Varnode(expr_varnode) => {
                Value::Ref(expr_value.sleigh().varnode(expr_varnode.id).referance())
            },
            sleigh_rs::execution::ExprValue::Context(expr_context) => todo!(),
            sleigh_rs::execution::ExprValue::Bitrange(expr_bitrange) => todo!(),
            sleigh_rs::execution::ExprValue::Table(expr_table) => todo!(),
            sleigh_rs::execution::ExprValue::DisVar(expr_dis_var) => todo!(),
            sleigh_rs::execution::ExprValue::ExeVar(expr_exe_var) => todo!(),
        };
        let var = self.alloc_var();
        self.push_stmt(Statement::Assignment(Assignment{
            var: var,
            expr: Expr::Value(value),
        }));
        Ok(var)
    }

    pub fn lift_unary_expr(&self, unary_expr: SleighExprUnaryOp) -> Result<VarId> {
        let input = self.lift_expr(unary_expr.statement().self_ctx(&unary_expr.inner.input))?;
        let op = match &unary_expr.inner.op {
            sleigh_rs::execution::Unary::Dereference(memory_location) => {
                UnaryOp::Deref(memory_location.space, memory_location.len_bytes.get() as usize)
            },
            sleigh_rs::execution::Unary::Negation => UnaryOp::Neg,
            sleigh_rs::execution::Unary::BitNegation => UnaryOp::Not,
            op => bail!(format!("Unknown UnaryOp: {:?}", op)),
        };
        let var = self.alloc_var();
        self.push_stmt(Statement::Assignment(Assignment {
            var,
            expr: Expr::UnaryOp(UnaryOpExpr { op, input })
        }));
        Ok(var)
    }

    pub fn lift_binary_expr(&self, binary_expr: SleighExprBinaryOp) -> Result<VarId> {
        let left = self.lift_expr(binary_expr.statement().self_ctx(&binary_expr.inner.left))?;
        let right = self.lift_expr(binary_expr.statement().self_ctx(&binary_expr.inner.right))?;
        let var = self.alloc_var();
        let op = match binary_expr.inner.op {
            sleigh_rs::execution::Binary::Add => BinaryOp::Add,
            sleigh_rs::execution::Binary::Sub => BinaryOp::Sub,
            op => bail!(format!("Unknown BinaryOp: {:?}", op))
        };
        self.push_stmt(Statement::Assignment(Assignment {
            var,
            expr: Expr::BinaryOp(BinaryOpExpr { op, left, right, })
        }));
        Ok(var)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatementEffect {
    Continue,
    Break,
    Branch(BlockId),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Statement {
    WriteRef(WriteRef),
    Assignment(Assignment),
    CpuBranch(CpuBranch)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WriteRef {
    pub referance: VarId,
    pub value: VarId,
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Assignment {
    pub var: VarId,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Value(Value),
    Deref,
    BinaryOp(BinaryOpExpr),
    UnaryOp(UnaryOpExpr),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CpuBranch {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BinaryOpExpr {
    pub op: BinaryOp,
    pub left: VarId,
    pub right: VarId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add, Sub,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnaryOpExpr {
    pub op: UnaryOp,
    pub input: VarId
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg, Not, 
    Deref(SpaceId, usize)
}