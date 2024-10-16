use sleigh_rs::{execution::{Assignment, Block, BlockId, Build, CpuBranch, Execution, Export, ExportConst, Expr, ExprBinaryOp, ExprElement, ExprUnaryOp, ExprValue, LocalGoto, MemWrite, Statement, VariableId, WriteValue}, varnode::Varnode, SpaceId};

use crate::{with_context, WithCtx, SleighConstructor, SleighContext, SleighTable};

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
