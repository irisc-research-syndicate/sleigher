use sleigh_rs::{varnode::Varnode, SpaceId};

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
        write!(f, "{}:{}:{}", self.0 .0, self.2, self.1)
    }
}

impl std::fmt::Debug for Ref {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ref({}:{:?}:{})", self.0 .0, self.2, self.1)
    }
}

impl std::convert::From<&Varnode> for Ref {
    fn from(varnode: &Varnode) -> Self {
        Ref(
            varnode.space,
            varnode.len_bytes.get() as usize,
            Address(varnode.address),
        )
    }
}
