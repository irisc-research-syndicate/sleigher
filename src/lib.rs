
mod sleigh;
mod table;
mod constructor;
mod pattern;
mod bitconstraint;
mod token;

pub use sleigh::SleighSleigh;
pub use table::{SleighTable, SleighDisassembledTable};
pub use constructor::SleighConstructor;
pub use pattern::SleighPattern;
pub use bitconstraint::SleighBitConstraints;
pub use token::{SleighToken, SleighDecodedTokenField, SleighTokenField};

#[derive(Clone, Copy)]
pub struct WithCtx<'a, Ctx, T: ?Sized> {
    pub ctx: &'a Ctx,
    pub inner: &'a T
}

impl<'a, Ctx, T: ?Sized> WithCtx<'a, Ctx, T> {
    pub fn new(ctx: &'a Ctx, inner: &'a T) -> Self {
        WithCtx {
            ctx,
            inner,
        }
    }

    pub fn same_ctx<N>(&self, inner: &'a N) -> WithCtx<'a, Ctx, N> {
        WithCtx {
            ctx: self.ctx,
            inner,
        }
    }

    pub fn self_ctx<N>(&'a self, inner: &'a N) -> WithCtx<'a, Self, N> {
        WithCtx {
            ctx: self,
            inner
        }
    }
}

impl<'a, T: ?Sized> From<&'a T> for WithCtx<'a, (), T> {
    fn from(value: &'a T) -> Self {
        WithCtx::new(&(), value)
    }
}

impl<'a, Ctx, T: std::fmt::Debug + ?Sized> std::fmt::Debug for WithCtx<'a, Ctx, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.inner, f) // TODO: add context type name std::any::type_name
    }
}