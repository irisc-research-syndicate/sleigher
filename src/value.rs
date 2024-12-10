use sleigh_rs::{execution::VariableId, varnode::Varnode, SpaceId};

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

impl std::convert::From<&Varnode> for Ref {
    fn from(varnode: &Varnode) -> Self {
        Ref(varnode.space, varnode.len_bytes.get() as usize, Address(varnode.address))
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