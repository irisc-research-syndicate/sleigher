use sleigh_rs::{Sleigh, TableId, TokenFieldId};
use anyhow::Result;

use crate::{SleighDisassembledTable, SleighTable, SleighTokenField, Wrapper};

pub type SleighSleigh<'a> = Wrapper<'a, (), Sleigh>;

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

    pub fn disassemble_instruction(&'a self, data: &[u8]) -> Result<SleighDisassembledTable<'a>> {
        self.instruction_table().disassemble(data)
    }
}
