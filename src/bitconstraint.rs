use sleigh_rs::pattern::BitConstraint;

use crate::Wrapper;


pub type SleighBitConstraints<'a> = Wrapper<'a, (), [BitConstraint]>;

fn get_bit(data: &[u8], idx: usize) -> Option<bool> {
    if idx > data.len() * 8 {
        return None;
    }
    Some((data[idx >> 3] >> (idx & 7)) & 1 == 1)
}

impl<'a> SleighBitConstraints<'a> {
    pub fn matches(&self, data: &[u8]) -> bool {
        self.inner.iter().enumerate().all(|(idx, bit)| 
            bit.value().map_or(true, |expected|
                get_bit(data, idx).is_some_and(|actual|
                    expected == actual
                )
            )
        )
    }
}

impl<'a> std::fmt::Display for SleighBitConstraints<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.inner.chunks(8).map(|bits| bits.iter().rev().map(|bit| match bit {
            BitConstraint::Unrestrained => "x",
            BitConstraint::Defined(false) => "0",
            BitConstraint::Defined(true) => "1",
            BitConstraint::Restrained => "X",
        }).collect()).collect::<Vec<String>>().join("_"))
    }
}