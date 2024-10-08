use sleigh_rs::{token::{Token, TokenField}, Endian, FieldBits};

use crate::{SleighSleigh, WithCtx};

pub type SleighTokenField<'a> = WithCtx<'a, SleighSleigh<'a>, TokenField>;

#[derive(Debug)]
pub struct SleighDecodedTokenField<'a> {
    pub token_field: SleighTokenField<'a>,
    pub value: usize,
}

impl<'a> std::fmt::Display for SleighDecodedTokenField<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&match self.token_field.inner.meaning() {
            sleigh_rs::meaning::Meaning::NoAttach(_value_fmt) =>
                format!("{}", self.value),
            sleigh_rs::meaning::Meaning::Varnode(attach_varnode_id) => {
                let attach_varnode = self.token_field.ctx.inner.attach_varnode(attach_varnode_id);
                let varnode_id = attach_varnode.find_value(self.value).unwrap();
                self.token_field.ctx.inner.varnode(varnode_id).name().to_string()
            }

            sleigh_rs::meaning::Meaning::Literal(_attach_literal_id) => todo!(),
            sleigh_rs::meaning::Meaning::Number(_print_base, _attach_number_id) => todo!(),
        })
    }
}

impl<'a> SleighTokenField<'a> {
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn bits(&self) -> &FieldBits {
        &self.inner.bits
    }

    pub fn raw_value_is_signed(&self) -> bool {
        self.inner.raw_value_is_signed()
    }

    pub fn execution_value_is_signed(&self) -> bool {
        self.inner.execution_value_is_signed(self.ctx.inner)
    }

    pub fn token(&self) -> SleighToken<'a> {
        self.same_ctx(self.ctx.inner.token(self.inner.token))
    }

    pub fn decode(&self, data: &[u8]) -> SleighDecodedTokenField {
        let value = (self.token().decode(data) >> self.inner.bits.start()) & ((1 << self.inner.bits.len().get()) - 1);
        let value = if self.raw_value_is_signed() && (value & (1 << (self.inner.bits.len().get() - 1))) != 0 {
            -((1 << self.inner.bits.len().get()) - value as isize) as usize
        } else {
            value
        };
        SleighDecodedTokenField {
            token_field: self.clone(),
            value: value
        }
    }
}

impl<'a> std::fmt::Display for SleighTokenField<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}..{}", self.bits().start(), self.bits().end().get()))
    }
}

pub type SleighToken<'a> = WithCtx<'a, SleighSleigh<'a>, Token>;

impl<'a> SleighToken<'a> {
    pub fn decode(&self, data: &[u8]) -> usize {
        match (self.inner.endian, self.inner.len_bytes().get()) {
            (Endian::Big, 4) => u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize,
            _ => todo!()
        }
    }
}