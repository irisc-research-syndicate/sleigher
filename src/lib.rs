
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
pub struct Wrapper<'a, Ctx, T: ?Sized> {
    pub ctx: &'a Ctx,
    pub inner: &'a T
}

impl<'a, Ctx, T: ?Sized> Wrapper<'a, Ctx, T> {
    pub fn new(ctx: &'a Ctx, inner: &'a T) -> Self {
        Wrapper {
            ctx,
            inner,
        }
    }

    pub fn wrap<N>(&self, inner: &'a N) -> Wrapper<'a, Ctx, N> {
        Wrapper {
            ctx: self.ctx,
            inner,
        }
    }

    pub fn subs<N>(&'a self, inner: &'a N) -> Wrapper<'a, Self, N> {
        Wrapper {
            ctx: self,
            inner
        }
    }
}

impl<'a, T: ?Sized> From<&'a T> for Wrapper<'a, (), T> {
    fn from(value: &'a T) -> Self {
        Wrapper::new(&(), value)
    }
}

impl<'a, Ctx, T: std::fmt::Debug + ?Sized> std::fmt::Debug for Wrapper<'a, Ctx, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.inner, f) // TODO: add context type name std::any::type_name
    }
}