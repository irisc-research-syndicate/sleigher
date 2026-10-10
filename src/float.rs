//! Floating point values of p-code float ops. Like Ghidra, ops compute on the host's f64
//! and round the result to the output's format.

use anyhow::{bail, Result};

use crate::bigint::BigInt;

/// The value of a float varnode
pub fn to_f64(value: &BigInt) -> Result<f64> {
    Ok(match value.size() {
        4 => f32::from_bits(value.to_u64() as u32) as f64,
        8 => f64::from_bits(value.to_u64()),
        size => bail!("no {} byte float format", size),
    })
}

/// `value` as a `size` byte float, rounded to nearest even
pub fn from_f64(value: f64, size: u32) -> Result<BigInt> {
    Ok(match size {
        4 => BigInt::from_u64((value as f32).to_bits() as u64, 4),
        8 => BigInt::from_u64(value.to_bits(), 8),
        size => bail!("no {} byte float format", size),
    })
}

/// The signed integer `value` as a `size` byte float, rounded once
pub fn from_int(value: &BigInt, size: u32) -> Result<BigInt> {
    let magnitude = if value.is_negative() {
        -value.clone()
    } else {
        value.clone()
    };
    // The top 64 bits, any bits below are folded into the lowest so they still round up
    let shift = (magnitude.bits() - magnitude.lzcount()).saturating_sub(64);
    let sticky = !(magnitude.clone() << (magnitude.bits() - shift) as u64).is_zero();
    let top = (magnitude >> shift as u64).to_u64() | sticky as u64;
    let signed = |magnitude: f64| {
        if value.is_negative() {
            -magnitude
        } else {
            magnitude
        }
    };
    match size {
        // Through f64 the conversion would round twice
        4 => from_f64(signed((top as f32 * 2f32.powi(shift as i32)) as f64), 4),
        _ => from_f64(signed(top as f64 * 2f64.powi(shift as i32)), size),
    }
}

/// `value` truncated towards zero to a `size` byte signed integer. Up to 8 bytes this follows
/// Ghidra: the value saturates at 64 bits, NaN is 0, and the result is truncated to the size.
/// Wider integers saturate at their own size.
pub fn to_int(value: f64, size: u32) -> BigInt {
    if size <= 8 {
        return BigInt::from_u64(value as i64 as u64, size);
    }
    let max = BigInt::ones(size) >> 1;
    if value.is_nan() {
        return BigInt::zero(size);
    }
    if value.abs() >= 2f64.powi(8 * size as i32 - 1) {
        return if value < 0.0 { !max } else { max };
    }
    let value = value.trunc();
    if value == 0.0 {
        return BigInt::zero(size);
    }
    // Any whole number is its mantissa (with the implicit bit) times a power of two
    let bits = value.abs().to_bits();
    let mantissa = BigInt::from_u64((bits & ((1 << 52) - 1)) | (1 << 52), size);
    let exponent = (bits >> 52) as i32 - 1075;
    let magnitude = if exponent >= 0 {
        mantissa << exponent as u64
    } else {
        mantissa >> -exponent as u64
    };
    if value < 0.0 {
        -magnitude
    } else {
        magnitude
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn int_conversions() {
        let n = BigInt::from_u64;
        let float = |value: &BigInt, size| to_f64(&from_int(value, size).unwrap()).unwrap();
        assert_eq!(float(&n(5, 4), 8), 5.0);
        assert_eq!(float(&n(0xfffffffb, 4), 8), -5.0);
        assert_eq!(float(&BigInt::ones(16), 4), -1.0);
        assert_eq!(float(&(n(3, 16) << 100), 8), 3.0 * 2f64.powi(100));
        assert_eq!(float(&-(n(3, 16) << 100), 8), -3.0 * 2f64.powi(100));
        // 2^60 + 2^36 + 1 is just past halfway between two f32 values, and lands on the
        // halfway point once rounded to f64 first
        assert_eq!(
            float(&n((1 << 60) + (1 << 36) + 1, 8), 4),
            ((1u64 << 60) + (1 << 37)) as f64
        );
        // The same for bits below the top 64 of a wide integer
        let wide = (n((1 << 63) + (1 << 39), 32) << 64) + n(1, 32);
        assert_eq!(
            float(&wide, 4),
            ((1u64 << 63) + (1 << 40)) as f64 * 2f64.powi(64)
        );
        assert_eq!(float(&(BigInt::from_u64(1, 64) << 500), 4), f64::INFINITY);

        assert_eq!(to_int(-2.7, 4), n(0xfffffffe, 4));
        assert_eq!(to_int(f64::NAN, 4), n(0, 4));
        assert_eq!(to_int(1e30, 8), n(i64::MAX as u64, 8));
        assert_eq!(to_int(3e9, 4), n(3_000_000_000, 4));
        assert_eq!(to_int(-2.7, 16), BigInt::ones(16) - n(1, 16));
        assert_eq!(
            to_int(2f64.powi(100) + 2f64.powi(48), 16),
            (n(1, 16) << 100) + (n(1, 16) << 48)
        );
        assert_eq!(to_int(1e300, 16), BigInt::ones(16) >> 1);
        assert_eq!(to_int(f64::NEG_INFINITY, 16), !(BigInt::ones(16) >> 1));
        assert_eq!(to_int(0.5, 16), n(0, 16));
    }
}
