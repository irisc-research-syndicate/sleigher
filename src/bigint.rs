//! Fixed width integers of any number of bytes, the values of p-code varnodes

use std::cmp::Ordering;

/// A `size` byte integer, held as little endian 64-bit limbs with the bits above `size` bytes
/// clear. Arithmetic wraps at the size; the signed operations read the top bit as the sign.
/// Operands of a binary operation are resized to the size of the left one.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BigInt {
    size: u32,
    limbs: Vec<u64>,
}

/// Mask with the low `bits` bits set
fn low_bits(bits: u32) -> u64 {
    1u64.checked_shl(bits).map_or(u64::MAX, |bit| bit - 1)
}

impl BigInt {
    pub fn zero(size: u32) -> Self {
        Self {
            size,
            limbs: vec![0; (size as usize).div_ceil(8).max(1)],
        }
    }

    /// All bits set
    pub fn ones(size: u32) -> Self {
        !Self::zero(size)
    }

    /// `value` truncated to `size` bytes
    pub fn from_u64(value: u64, size: u32) -> Self {
        let mut result = Self::zero(size);
        result.limbs[0] = value;
        result.masked()
    }

    /// `value` truncated to `size` bytes
    pub fn from_u128(value: u128, size: u32) -> Self {
        Self::from_u64(value as u64, size) | (Self::from_u64((value >> 64) as u64, size) << 64)
    }

    /// A one byte boolean
    pub fn from_bool(value: bool) -> Self {
        Self::from_u64(value as u64, 1)
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let mut result = Self::zero(bytes.len() as u32);
        for (index, byte) in bytes.iter().enumerate() {
            result.limbs[index / 8] |= (*byte as u64) << (8 * (index % 8));
        }
        result
    }

    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut bytes = bytes.to_vec();
        bytes.reverse();
        Self::from_le_bytes(&bytes)
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        (0..self.size as usize)
            .map(|index| (self.limbs[index / 8] >> (8 * (index % 8))) as u8)
            .collect()
    }

    pub fn to_be_bytes(&self) -> Vec<u8> {
        let mut bytes = self.to_le_bytes();
        bytes.reverse();
        bytes
    }

    /// Size in bytes
    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn bits(&self) -> u32 {
        8 * self.size
    }

    /// The low 64 bits
    pub fn to_u64(&self) -> u64 {
        self.limbs[0]
    }

    /// The low 128 bits
    pub fn to_u128(&self) -> u128 {
        self.limbs[0] as u128 | (self.limbs.get(1).copied().unwrap_or(0) as u128) << 64
    }

    /// The value, if it fits in 64 bits
    pub fn to_u64_checked(&self) -> Option<u64> {
        self.limbs[1..]
            .iter()
            .all(|limb| *limb == 0)
            .then_some(self.limbs[0])
    }

    /// The value as a shift amount, any amount past 64 bits is as good as `u64::MAX`
    pub fn to_shift(&self) -> u64 {
        self.to_u64_checked().unwrap_or(u64::MAX)
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|limb| *limb == 0)
    }

    /// Bit `index`, counted from the least significant bit
    pub fn bit(&self, index: u32) -> bool {
        index < self.bits() && (self.limbs[index as usize / 64] >> (index % 64)) & 1 != 0
    }

    pub fn is_negative(&self) -> bool {
        self.size > 0 && self.bit(self.bits() - 1)
    }

    /// Clear the bits above the size
    fn masked(mut self) -> Self {
        let last = self.limbs.len() - 1;
        self.limbs[last] &= low_bits(self.bits() - 64 * last as u32);
        self
    }

    /// Zero extended or truncated to `size` bytes
    pub fn resize(&self, size: u32) -> Self {
        let mut result = Self::zero(size);
        let len = result.limbs.len().min(self.limbs.len());
        result.limbs[..len].copy_from_slice(&self.limbs[..len]);
        result.masked()
    }

    /// Sign extended or truncated to `size` bytes
    pub fn sext(&self, size: u32) -> Self {
        let result = self.resize(size);
        if self.is_negative() && size > self.size {
            result | (Self::ones(size) << self.bits() as u64)
        } else {
            result
        }
    }

    /// Unsigned comparison, of values of any size
    pub fn ucmp(&self, other: &Self) -> Ordering {
        let len = self.limbs.len().max(other.limbs.len());
        let limb = |value: &Self, index: usize| value.limbs.get(index).copied().unwrap_or(0);
        (0..len)
            .rev()
            .map(|index| limb(self, index).cmp(&limb(other, index)))
            .find(|ordering| ordering.is_ne())
            .unwrap_or(Ordering::Equal)
    }

    /// Signed comparison, of values of any size
    pub fn scmp(&self, other: &Self) -> Ordering {
        match (self.is_negative(), other.is_negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => {
                let size = self.size.max(other.size);
                self.sext(size).ucmp(&other.sext(size))
            }
        }
    }

    /// Whether the unsigned addition overflows
    pub fn carry(&self, other: &Self) -> bool {
        (self.clone() + other.clone()).ucmp(self).is_lt()
    }

    /// Unsigned quotient and remainder, `None` when dividing by zero
    pub fn udiv_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        let divisor = divisor.resize(self.size);
        if divisor.is_zero() {
            return None;
        }
        if let (Some(a), Some(b)) = (self.to_u64_checked(), divisor.to_u64_checked()) {
            return Some((
                Self::from_u64(a / b, self.size),
                Self::from_u64(a % b, self.size),
            ));
        }
        // Long division a bit at a time: the remainder can grow one bit past the size before
        // the divisor is subtracted, that bit is kept in `carry`
        let mut quotient = Self::zero(self.size);
        let mut remainder = Self::zero(self.size);
        for index in (0..self.bits()).rev() {
            let carry = remainder.is_negative();
            remainder = remainder << 1;
            remainder.limbs[0] |= self.bit(index) as u64;
            if carry || remainder.ucmp(&divisor).is_ge() {
                remainder = remainder - divisor.clone();
                quotient.limbs[index as usize / 64] |= 1 << (index % 64);
            }
        }
        Some((quotient, remainder))
    }

    /// Signed quotient and remainder truncated towards zero, `None` when dividing by zero
    pub fn sdiv_rem(&self, divisor: &Self) -> Option<(Self, Self)> {
        let divisor = divisor.resize(self.size);
        let abs = |value: &Self| {
            if value.is_negative() {
                -value.clone()
            } else {
                value.clone()
            }
        };
        let (quotient, remainder) = abs(self).udiv_rem(&abs(&divisor))?;
        let quotient = if self.is_negative() != divisor.is_negative() {
            -quotient
        } else {
            quotient
        };
        let remainder = if self.is_negative() {
            -remainder
        } else {
            remainder
        };
        Some((quotient, remainder))
    }

    /// Arithmetic shift right
    pub fn sar(&self, amount: u64) -> Self {
        if self.is_negative() {
            !(!self.clone() >> amount)
        } else {
            self.clone() >> amount
        }
    }

    pub fn popcount(&self) -> u32 {
        self.limbs.iter().map(|limb| limb.count_ones()).sum()
    }

    /// Leading zeros within the size
    pub fn lzcount(&self) -> u32 {
        let used = self
            .limbs
            .iter()
            .rposition(|limb| *limb != 0)
            .map_or(0, |index| {
                64 * index as u32 + 64 - self.limbs[index].leading_zeros()
            });
        self.bits() - used
    }

    /// Combine with `other` limb by limb, `other` resized to this size
    fn zip(mut self, other: &Self, f: impl Fn(u64, u64) -> u64) -> Self {
        let other = other.resize(self.size);
        for (limb, other) in self.limbs.iter_mut().zip(other.limbs) {
            *limb = f(*limb, other);
        }
        self.masked()
    }

    #[cfg(test)]
    pub fn from_hex(hex: &str, size: u32) -> Self {
        let hex = hex.trim_start_matches("0x").replace('_', "");
        let mut result = Self::zero(size);
        for (index, digit) in hex.chars().rev().enumerate() {
            let digit = digit.to_digit(16).expect("hex digit") as u64;
            result = result | (Self::from_u64(digit, size) << (4 * index as u64));
        }
        result
    }
}

impl std::ops::Add for BigInt {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        let other = other.resize(self.size);
        let mut carry = false;
        for (limb, other) in self.limbs.iter_mut().zip(other.limbs) {
            let (sum, carry1) = limb.overflowing_add(other);
            let (sum, carry2) = sum.overflowing_add(carry as u64);
            *limb = sum;
            carry = carry1 || carry2;
        }
        self.masked()
    }
}

impl std::ops::Sub for BigInt {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        let other = other.resize(self.size);
        let mut borrow = false;
        for (limb, other) in self.limbs.iter_mut().zip(other.limbs) {
            let (difference, borrow1) = limb.overflowing_sub(other);
            let (difference, borrow2) = difference.overflowing_sub(borrow as u64);
            *limb = difference;
            borrow = borrow1 || borrow2;
        }
        self.masked()
    }
}

impl std::ops::Mul for BigInt {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let other = other.resize(self.size);
        let len = self.limbs.len();
        let mut result = Self::zero(self.size);
        for i in 0..len {
            let mut carry = 0u128;
            for j in 0..len - i {
                let product = self.limbs[i] as u128 * other.limbs[j] as u128
                    + result.limbs[i + j] as u128
                    + carry;
                result.limbs[i + j] = product as u64;
                carry = product >> 64;
            }
        }
        result.masked()
    }
}

impl std::ops::Neg for BigInt {
    type Output = Self;

    fn neg(self) -> Self {
        Self::zero(self.size) - self
    }
}

impl std::ops::Not for BigInt {
    type Output = Self;

    fn not(mut self) -> Self {
        for limb in &mut self.limbs {
            *limb = !*limb;
        }
        self.masked()
    }
}

impl std::ops::BitAnd for BigInt {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        self.zip(&other, |a, b| a & b)
    }
}

impl std::ops::BitOr for BigInt {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        self.zip(&other, |a, b| a | b)
    }
}

impl std::ops::BitXor for BigInt {
    type Output = Self;

    fn bitxor(self, other: Self) -> Self {
        self.zip(&other, |a, b| a ^ b)
    }
}

impl std::ops::Shl<u64> for BigInt {
    type Output = Self;

    fn shl(self, amount: u64) -> Self {
        let mut result = Self::zero(self.size);
        if amount >= self.bits() as u64 {
            return result;
        }
        let (limbs, bits) = (amount as usize / 64, amount as u32 % 64);
        for index in limbs..self.limbs.len() {
            let from = index - limbs;
            let carried = if bits > 0 && from > 0 {
                self.limbs[from - 1] >> (64 - bits)
            } else {
                0
            };
            result.limbs[index] = (self.limbs[from] << bits) | carried;
        }
        result.masked()
    }
}

impl std::ops::Shr<u64> for BigInt {
    type Output = Self;

    fn shr(self, amount: u64) -> Self {
        let mut result = Self::zero(self.size);
        if amount >= self.bits() as u64 {
            return result;
        }
        let (limbs, bits) = (amount as usize / 64, amount as u32 % 64);
        for index in 0..self.limbs.len() - limbs {
            let high = self.limbs[index + limbs];
            let carried = match self.limbs.get(index + limbs + 1) {
                Some(next) if bits > 0 => next << (64 - bits),
                _ => 0,
            };
            result.limbs[index] = (high >> bits) | carried;
        }
        result
    }
}

impl std::fmt::LowerHex for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut digits = String::new();
        for limb in self.limbs.iter().rev() {
            if digits.is_empty() {
                if *limb != 0 {
                    digits = format!("{:x}", limb);
                }
            } else {
                digits += &format!("{:016x}", limb);
            }
        }
        if digits.is_empty() {
            digits.push('0');
        }
        f.pad_integral(true, "0x", &digits)
    }
}

impl std::fmt::Debug for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#x}:{}", self, self.size)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn hex(hex: &str, size: u32) -> BigInt {
        BigInt::from_hex(hex, size)
    }

    #[test]
    fn bytes_round_trip() {
        let bytes: Vec<u8> = (1..=20).collect();
        let value = BigInt::from_le_bytes(&bytes);
        assert_eq!(value.size(), 20);
        assert_eq!(value.to_le_bytes(), bytes);
        assert_eq!(value.to_u64(), 0x0807060504030201);
        assert_eq!(BigInt::from_be_bytes(&bytes).to_be_bytes(), bytes);
        assert_eq!(
            format!("{:#x}", value),
            "0x14131211100f0e0d0c0b0a090807060504030201"
        );
        assert_eq!(format!("{:?}", BigInt::zero(3)), "0x0:3");
    }

    #[test]
    fn masking() {
        assert_eq!(BigInt::from_u64(0x1234, 1).to_u64(), 0x34);
        assert_eq!(BigInt::ones(3).to_u64(), 0xffffff);
        assert_eq!(BigInt::ones(9), hex("ff_ffffffff_ffffffff", 9));
        assert_eq!(BigInt::ones(9).popcount(), 72);
        assert_eq!(
            (BigInt::ones(12) + BigInt::from_u64(1, 12)),
            BigInt::zero(12)
        );
        assert_eq!(BigInt::zero(12) - BigInt::from_u64(1, 12), BigInt::ones(12));
        assert_eq!(-BigInt::from_u64(1, 2), BigInt::ones(2));
    }

    #[test]
    fn carries_cross_limbs() {
        let a = hex("ffffffffffffffff", 16);
        assert_eq!(
            a.clone() + BigInt::from_u64(1, 16),
            hex("1_0000000000000000", 16)
        );
        assert_eq!(hex("1_0000000000000000", 16) - BigInt::from_u64(1, 16), a);
        assert!(BigInt::ones(16).carry(&BigInt::from_u64(1, 16)));
        assert!(!a.carry(&BigInt::from_u64(1, 16)));
        assert!(BigInt::from_u64(0xff, 1).carry(&BigInt::from_u64(1, 1)));
    }

    #[test]
    fn multiplication() {
        let a = hex("ffffffffffffffff", 16);
        assert_eq!(
            a.clone() * a.clone(),
            hex("fffffffffffffffe_0000000000000001", 16)
        );
        // Wraps at the size
        assert_eq!(a.resize(9) * hex("100", 9), hex("ff_ffffffff_ffffff00", 9));
        assert_eq!(BigInt::ones(64) * BigInt::ones(64), BigInt::from_u64(1, 64));
    }

    #[test]
    fn division() {
        let a = hex("123456789abcdef0_fedcba9876543210", 16);
        let b = hex("1_0000000000000000", 16);
        let (q, r) = a.udiv_rem(&b).unwrap();
        assert_eq!(q, hex("123456789abcdef0", 16));
        assert_eq!(r, hex("fedcba9876543210", 16));
        let (q, r) = BigInt::ones(16).udiv_rem(&hex("3", 16)).unwrap();
        assert_eq!(q, hex("55555555555555555555555555555555", 16));
        assert!(r.is_zero());
        let (q, r) = BigInt::ones(16).udiv_rem(&BigInt::ones(16)).unwrap();
        assert_eq!((q.to_u64(), r.to_u64()), (1, 0));
        assert!(a.udiv_rem(&BigInt::zero(16)).is_none());

        // -7 / 2 = -3 rem -1 at 16 bytes and at 2 bytes
        for size in [2, 16] {
            let (q, r) = (-BigInt::from_u64(7, size))
                .sdiv_rem(&BigInt::from_u64(2, size))
                .unwrap();
            assert_eq!(q, -BigInt::from_u64(3, size));
            assert_eq!(r, -BigInt::from_u64(1, size));
        }
        // The most negative value divided by -1 wraps
        let min = BigInt::from_u64(1, 16) << 127;
        let (q, _) = min.sdiv_rem(&BigInt::ones(16)).unwrap();
        assert_eq!(q, min);
    }

    #[test]
    fn shifts() {
        let a = hex("8000000000000001_0000000000000003", 16);
        assert_eq!(a.clone() << 4, hex("10_0000000000000030", 16));
        assert_eq!(a.clone() << 64, hex("3_0000000000000000", 16));
        assert_eq!(
            a.clone() << 127,
            hex("8000000000000000_0000000000000000", 16)
        );
        assert_eq!(a.clone() << 128, BigInt::zero(16));
        assert_eq!(a.clone() >> 4, hex("0800000000000000_1000000000000000", 16));
        assert_eq!(a.clone() >> 65, hex("4000000000000000", 16));
        assert_eq!(a.clone() >> 128, BigInt::zero(16));
        assert_eq!(a.sar(4), hex("f800000000000000_1000000000000000", 16));
        assert_eq!(a.sar(127), BigInt::ones(16));
        assert_eq!(a.sar(u64::MAX), BigInt::ones(16));
        assert_eq!(hex("80", 1).sar(3), hex("f0", 1));
        assert_eq!(hex("40", 1).sar(3), hex("08", 1));
        // Shifting into the top byte of a size that is not a multiple of 8 bytes
        assert_eq!(
            BigInt::from_u64(0xff, 9) << 68,
            hex("f0_0000000000000000", 9)
        );
    }

    #[test]
    fn extension_and_comparison() {
        let minus_one = BigInt::ones(4);
        assert_eq!(minus_one.sext(16), BigInt::ones(16));
        assert_eq!(minus_one.resize(16), hex("ffffffff", 16));
        assert_eq!(hex("7fffffff", 4).sext(16), hex("7fffffff", 16));
        assert_eq!(hex("1234_5678", 16).sext(2), hex("5678", 2));

        let one = BigInt::from_u64(1, 16);
        assert!(BigInt::ones(16).ucmp(&one).is_gt());
        assert!(BigInt::ones(16).scmp(&one).is_lt());
        assert!(BigInt::ones(16).scmp(&BigInt::ones(4)).is_eq());
        assert!((BigInt::ones(16) << 1).scmp(&BigInt::ones(16)).is_lt());
        assert!(hex("1_0000000000000000", 16).ucmp(&BigInt::ones(8)).is_gt());
    }

    #[test]
    fn counts() {
        assert_eq!(BigInt::zero(64).lzcount(), 512);
        assert_eq!(BigInt::from_u64(1, 64).lzcount(), 511);
        assert_eq!((BigInt::from_u64(1, 64) << 300).lzcount(), 211);
        assert_eq!(BigInt::ones(64).popcount(), 512);
        assert_eq!(BigInt::from_u64(1, 3).lzcount(), 23);
        assert_eq!(hex("f0f0_0000000000000000", 10).popcount(), 8);
    }
}
