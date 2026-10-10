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

/// Sign extend the low `bits` bits of `value`, 1 to 64 bits
pub fn sign_extend(value: u64, bits: u32) -> i64 {
    debug_assert!((1..=64).contains(&bits));
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

fn parse_digits(s: &str, radix: u32) -> Option<(&str, u64)> {
    let len = s.find(|c: char| !c.is_digit(radix)).unwrap_or(s.len());
    let value = u64::from_str_radix(&s[..len], radix).ok()?;
    Some((&s[len..], value))
}

fn parse_hex(s: &str) -> Option<(&str, u64)> {
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    parse_digits(s, 16)
}

fn parse_dec(s: &str) -> Option<(&str, u64)> {
    parse_digits(s, 10)
}

/// A decimal or `0x` hex number, negative only if `signed`, and the rest of `s`
pub fn parse_number(signed: bool, s: &str) -> Option<(&str, i64)> {
    let (s, sign) = match s.strip_prefix('-') {
        Some(s) if signed => (s, true),
        _ => (s, false),
    };
    let (s, value) = parse_hex(s).or_else(|| parse_dec(s))?;
    let value = if sign { -(value as i64) } else { value as i64 };
    Some((s, value))
}
