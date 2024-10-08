use sleigh_rs::{execution::{Assignment, Block, BlockId, Build, CpuBranch, Execution, Export, ExportConst, Expr, ExprBinaryOp, ExprElement, ExprUnaryOp, ExprValue, LocalGoto, MemWrite, Statement, VariableId, WriteValue}, varnode::Varnode, SpaceId};

use crate::{with_context, SleighConstructor, SleighContext, SleighSleigh, SleighTable, WithCtx};

with_context!(SleighExecution, SleighConstructor<'a>, Execution, ExecutionContext, execution);
with_context!(SleighBlock, SleighExecution<'a>, Block, BlockContext, block);
with_context!(SleighStatement, SleighBlock<'a>, Statement, StatementContext, statement);
with_context!(SleighBuild, SleighStatement<'a>, Build, BuildContext, build);
with_context!(SleighAssignment, SleighStatement<'a>, Assignment, AssignmentContext, assignment);
with_context!(SleighWriteValue, SleighStatement<'a>, WriteValue, WriteValueContext, write_value);
with_context!(SleighExpr, SleighStatement<'a>, Expr, ExprContext, expr);
with_context!(SleighExprElement, SleighExpr<'a>, ExprElement, ExprElementContext, expr_elem);
with_context!(SleighExprValue, SleighExpr<'a>, ExprValue, ExprValueContext, expr_value);
with_context!(SleighExprBinaryOp, SleighExpr<'a>, ExprBinaryOp, ExprBinaryOpContext, expr_binop);
with_context!(SleighExprUnaryOp, SleighExpr<'a>, ExprUnaryOp, ExprUnaryOpContext, expr_unaryop);
with_context!(SleighExport, SleighStatement<'a>, Export, ExportContext, export);
with_context!(SleighExportConst, SleighStatement<'a>, ExportConst, ExportConstContext, export_const);
with_context!(SleighMemWrite, SleighStatement<'a>, MemWrite, MemWriteContext, mem_write);
with_context!(SleighCpuBranch, SleighStatement<'a>, CpuBranch, CpuBranchContext, cpu_branch);
with_context!(SleighLocalGoto, SleighStatement<'a>, LocalGoto, LocalGotoContext, local_goto);


impl<'a> SleighAssignment<'a> {
    pub fn right(&self) -> SleighExpr {
        self.same_ctx(&self.inner.right)
    }

    pub fn var(&self) -> SleighWriteValue {
        self.same_ctx(&self.inner.var)
    }
}

impl<'a> SleighExecution<'a> {
    pub fn entry_block(&self) -> SleighBlock {
        self.block(self.inner.entry_block)
    }

    pub fn block(&self, id: BlockId) -> SleighBlock {
        self.self_ctx(self.inner.block(id))
    }
}

impl<'a> SleighBlock<'a> {
    pub fn next(&self) -> Option<SleighBlock> {
        self.inner.next.as_ref().map(|id| self.execution().block(*id))
    }

    pub fn statements(&self) -> impl Iterator<Item = SleighStatement> {
        self.inner.statements.iter().map(|stmt| self.self_ctx(stmt))
    }
}

impl<'a> SleighBuild<'a> {
    pub fn table(&self) -> SleighTable {
        let foo = self.sleigh();
        foo.table(self.inner.table.id)
    }
}

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Address(pub u64);

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#018x}", self.0)
    }
}

impl std::fmt::Debug for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Address({:#018x})", self.0)
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Ref(pub SpaceId, pub usize, pub Address);

impl std::fmt::Display for Ref {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.0.0, self.2, self.1)
    }
}

impl std::fmt::Debug for Ref {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ref({}:{:?}:{})", self.0.0, self.2, self.1)
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum Value {
    Int(u64),
    Ref(Ref),
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(arg0) => write!(f, "Int({:#018x})", arg0),
            Self::Ref(arg0) => write!(f, "Ref({})", arg0),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum Var {
    Ref(Ref),
    Local(VariableId),
}

impl std::fmt::Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ref(arg0) => write!(f, "Ref({})", arg0),
            Self::Local(arg0) => f.debug_tuple("Local").field(arg0).finish(),
        }
    }
}

impl Value {
    pub fn to_var(&self) -> Var {
        match self {
            Value::Int(address) => Var::Ref(Ref(SpaceId(0), 4, Address(*address))), // fixme: Default memory space is 0???
            Value::Ref(x) => Var::Ref(*x),
        }
    }

    pub fn to_u64(&self) -> u64 {
        match self {
            Value::Int(x) => *x,
            Value::Ref(x) => x.2.0,
        }
    }
}