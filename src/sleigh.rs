use sleigh_rs::{meaning::AttachVarnode, varnode::Varnode, Sleigh, TableId, TokenFieldId, VarnodeId};

use crate::{value::{Address, Ref}, with_context, SleighTable, SleighTokenField, WithCtx};

with_context!(SleighSleigh, (), Sleigh, SleighContext, sleigh);
with_context!(SleighVarnode, SleighSleigh<'a>, Varnode, VarnodeContext, varnode);
with_context!(SleighAttachVarnode, SleighSleigh<'a>, AttachVarnode, AttachVarnodeContext, attach_varnode);

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

    pub fn varnode(&self, id: VarnodeId) -> SleighVarnode {
        self.self_ctx(self.inner.varnode(id))
    }

    pub fn varnodes(&self) -> impl Iterator<Item = SleighVarnode> {
        self.inner.varnodes().iter().map(|varnode| self.self_ctx(varnode))
    }

    pub fn varnode_by_name(&self, name: &str) -> Option<SleighVarnode> {
        self.varnodes().find(|varnode| varnode.name() == name)
    }
}

impl<'a> SleighVarnode<'a> {
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn referance(&self) -> Ref {
        Ref(self.inner.space, self.inner.len_bytes.get() as usize, Address(self.inner.address))
    }
}