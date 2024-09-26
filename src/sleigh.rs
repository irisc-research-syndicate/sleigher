use sleigh_rs::{Sleigh, TableId, TokenFieldId};
use anyhow::Result;

use crate::{SleighDisassembledTable, SleighTable, SleighTokenField, WithCtx};

pub type SleighSleigh<'a> = WithCtx<'a, (), Sleigh>;

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

    pub fn token_field(&self, id: TokenFieldId) -> SleighTokenField {
        self.self_ctx(self.inner.token_field(id))
    }

    pub fn disassemble_instruction(&'a self, data: &[u8]) -> Result<SleighDisassembledTable<'a>> {
        self.instruction_table().disassemble(data)
    }
}
