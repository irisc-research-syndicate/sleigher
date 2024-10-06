use sleigh_rs::{Sleigh, TableId, TokenFieldId};

use crate::{WithCtx, with_context, SleighTable, SleighTokenField};

with_context!(SleighSleigh, (), Sleigh, SleighContext, sleigh);

impl<'a> SleighSleigh<'a> {
    pub fn tables(&self) -> impl Iterator<Item = SleighTable> {
        self.inner.tables().iter().map(|table| self.self_ctx(table))
    }

    pub fn table(&self, id: TableId) -> SleighTable {
        self.self_ctx(self.inner.table(id))
    }

    pub fn instruction_table(&self) -> SleighTable {
        self.table(self.inner.instruction_table())
    }

    pub fn token_fields(&self) -> impl Iterator<Item = SleighTokenField> {
        self.inner.token_fields().iter().map(|token_field| self.self_ctx(token_field))
    }

    pub fn token_field(&self, id: TokenFieldId) -> SleighTokenField {
        self.self_ctx(self.inner.token_field(id))
    }

}
