//! Floating point values of p-code float ops, in half (2 byte), single (4), double (8) and
//! x87 extended (10) precision. Arithmetic computes on the host's f64 and rounds the result to
//! the output's format, like Ghidra's decompiler does (Ghidra's emulator may use wider floats).
//! On extended precision values arithmetic, square roots and comparisons therefore keep only
//! double's precision and exponent range: 2^4096 reads as infinity and values below double's
//! subnormals as 0. Conversions and rounding to whole numbers are exact in every format.

use anyhow::{bail, Result};

use crate::bigint::BigInt;

/// A decoded float. A finite value is `(-1)^negative * mantissa * 2^exponent`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Parts {
    Nan {
        negative: bool,
    },
    Infinite {
        negative: bool,
    },
    Finite {
        negative: bool,
        mantissa: u128,
        exponent: i32,
    },
}

/// An IEEE 754 style binary format: sign, exponent and fraction from the top bit down. x87's
/// extended precision stores the integer bit of the mantissa, the others imply it.
#[derive(Debug, Clone, Copy)]
struct Format {
    exponent_bits: u32,
    /// Mantissa bits stored below the exponent
    fraction_bits: u32,
    explicit_integer: bool,
}

const DOUBLE: Format = Format {
    exponent_bits: 11,
    fraction_bits: 52,
    explicit_integer: false,
};

impl Format {
    fn of(size: u32) -> Result<Self> {
        let (exponent_bits, fraction_bits, explicit_integer) = match size {
            2 => (5, 10, false),
            4 => (8, 23, false),
            8 => (11, 52, false),
            10 => (15, 64, true),
            size => bail!("no {} byte float format", size),
        };
        Ok(Self {
            exponent_bits,
            fraction_bits,
            explicit_integer,
        })
    }

    /// Mantissa bits including the integer bit
    fn precision(&self) -> u32 {
        self.fraction_bits + !self.explicit_integer as u32
    }

    fn bias(&self) -> i32 {
        (1 << (self.exponent_bits - 1)) - 1
    }

    fn max_exponent(&self) -> u128 {
        (1 << self.exponent_bits) - 1
    }

    /// The stored integer bit of the special values, for x87
    fn integer_bit(&self) -> u128 {
        (self.explicit_integer as u128) << (self.fraction_bits - 1)
    }

    fn decode(&self, bits: u128) -> Parts {
        let fraction_bits = self.fraction_bits;
        let negative = (bits >> (fraction_bits + self.exponent_bits)) & 1 != 0;
        let biased = (bits >> fraction_bits) & self.max_exponent();
        let mantissa = bits & ((1 << fraction_bits) - 1);
        if biased == self.max_exponent() {
            return if mantissa & !self.integer_bit() == 0 {
                Parts::Infinite { negative }
            } else {
                Parts::Nan { negative }
            };
        }
        // Subnormals have the exponent of the smallest normal number
        let exponent = (biased as i32).max(1) - self.bias() - (self.precision() as i32 - 1);
        let mantissa = if self.explicit_integer || biased == 0 {
            mantissa
        } else {
            mantissa | (1 << fraction_bits)
        };
        Parts::Finite {
            negative,
            mantissa,
            exponent,
        }
    }

    /// Encode `parts`, rounding to nearest even
    fn encode(&self, parts: Parts) -> u128 {
        let fraction_bits = self.fraction_bits;
        let sign = |negative: bool| (negative as u128) << (fraction_bits + self.exponent_bits);
        let infinity = (self.max_exponent() << fraction_bits) | self.integer_bit();
        let (negative, mantissa, exponent) = match parts {
            Parts::Nan { negative } => {
                let quiet = 1 << (self.precision() - 2);
                return sign(negative) | infinity | quiet;
            }
            Parts::Infinite { negative } => return sign(negative) | infinity,
            Parts::Finite { negative, .. } if parts.is_zero() => return sign(negative),
            Parts::Finite {
                negative,
                mantissa,
                exponent,
            } => (negative, mantissa, exponent),
        };
        // With the mantissa's top bit at bit 127 the value is in [2^(exponent + 127), 2 times
        // that); numbers below the smallest normal keep fewer bits
        let shift = mantissa.leading_zeros();
        let (mantissa, exponent) = (mantissa << shift, exponent as i64 - shift as i64);
        let biased = exponent + 127 + self.bias() as i64;
        let drop = (128 - self.precision()) as i64 + (1 - biased).max(0);
        let mut mantissa = round_shift(mantissa, drop);
        let mut biased = biased.max(1) as u128;
        if self.explicit_integer {
            if mantissa >> self.precision() != 0 {
                // Rounded up to the next power of two
                mantissa >>= 1;
                biased += 1;
            }
            if mantissa >> (self.precision() - 1) == 0 {
                biased = 0;
            }
        } else if mantissa >> fraction_bits != 0 {
            // The implicit integer bit, possibly carried into the exponent
            mantissa -= 1 << fraction_bits;
        } else {
            biased = 0;
        }
        let bits = (biased << fraction_bits) + mantissa;
        sign(negative) | bits.min(infinity)
    }
}

impl Parts {
    fn is_zero(&self) -> bool {
        matches!(self, Parts::Finite { mantissa: 0, .. })
    }
}

/// `value` shifted right by `drop` bits, rounded to nearest even
fn round_shift(value: u128, drop: i64) -> u128 {
    if drop <= 0 {
        return value;
    }
    if drop > 128 {
        return 0;
    }
    let kept = value.checked_shr(drop as u32).unwrap_or(0);
    let rest = value - kept.checked_shl(drop as u32).unwrap_or(0);
    let half = 1 << (drop - 1);
    if rest > half || (rest == half && kept & 1 == 1) {
        kept + 1
    } else {
        kept
    }
}

fn decode(value: &BigInt) -> Result<Parts> {
    Ok(Format::of(value.size())?.decode(value.to_u128()))
}

fn encode(parts: Parts, size: u32) -> Result<BigInt> {
    Ok(BigInt::from_u128(Format::of(size)?.encode(parts), size))
}

/// The value of a float varnode
pub fn to_f64(value: &BigInt) -> Result<f64> {
    Ok(match value.size() {
        4 => f32::from_bits(value.to_u64() as u32) as f64,
        8 => f64::from_bits(value.to_u64()),
        _ => f64::from_bits(DOUBLE.encode(decode(value)?) as u64),
    })
}

/// `value` as a `size` byte float, rounded to nearest even
pub fn from_f64(value: f64, size: u32) -> Result<BigInt> {
    match size {
        4 => Ok(BigInt::from_u64((value as f32).to_bits() as u64, 4)),
        8 => Ok(BigInt::from_u64(value.to_bits(), 8)),
        _ => encode(DOUBLE.decode(value.to_bits() as u128), size),
    }
}

/// The float `value` as a `size` byte float, rounded to nearest even
pub fn convert(value: &BigInt, size: u32) -> Result<BigInt> {
    encode(decode(value)?, size)
}

/// The signed integer `value` as a `size` byte float, rounded to nearest even
pub fn from_int(value: &BigInt, size: u32) -> Result<BigInt> {
    let negative = value.is_negative();
    let magnitude = if negative {
        -value.clone()
    } else {
        value.clone()
    };
    // The top 128 bits, any bits below are folded into the lowest so they still round up
    let shift = (magnitude.bits() - magnitude.lzcount()).saturating_sub(128);
    let sticky = !(magnitude.clone() << (magnitude.bits() - shift) as u64).is_zero();
    let mantissa = (magnitude >> shift as u64).to_u128() | sticky as u128;
    let parts = Parts::Finite {
        negative,
        mantissa,
        exponent: shift as i32,
    };
    encode(parts, size)
}

/// The float `value` truncated towards zero to a `size` byte signed integer. Up to 8 bytes
/// this follows Ghidra: the value saturates at 64 bits, NaN is 0, and the result is truncated
/// to the size. Wider integers saturate at their own size.
pub fn to_int(value: &BigInt, size: u32) -> Result<BigInt> {
    let bits = 8 * size.max(8);
    let saturated = |negative: bool| {
        let max = BigInt::ones(bits / 8) >> 1;
        if negative {
            !max
        } else {
            max
        }
    };
    let result = match decode(value)? {
        Parts::Nan { .. } => BigInt::zero(size),
        Parts::Infinite { negative } => saturated(negative),
        Parts::Finite {
            negative,
            mantissa,
            exponent,
        } => {
            if (128 - mantissa.leading_zeros()) as i64 + exponent as i64 > bits as i64 - 1 {
                saturated(negative)
            } else {
                // Room for the whole mantissa before it is shifted into place
                let mantissa = BigInt::from_u128(mantissa, bits / 8 + 16);
                let magnitude = if exponent >= 0 {
                    mantissa << exponent as u64
                } else {
                    mantissa >> exponent.unsigned_abs() as u64
                };
                let magnitude = magnitude.resize(bits / 8);
                if negative {
                    -magnitude
                } else {
                    magnitude
                }
            }
        }
    };
    Ok(result.resize(size))
}

/// How `round_to_whole` rounds
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rounding {
    Ceil,
    Floor,
    /// Halfway cases away from zero, like round() in FloatFormat::opRound of Ghidra's
    /// decompiler (float.cc); Ghidra's Java emulator uses floor(x + 0.5)
    Round,
}

/// The float `value` rounded to a whole number, in its own format
pub fn round_to_whole(value: &BigInt, rounding: Rounding) -> Result<BigInt> {
    let parts = match decode(value)? {
        Parts::Finite {
            negative,
            mantissa,
            exponent,
        } if exponent < 0 => {
            let drop = exponent.unsigned_abs();
            let whole = mantissa.checked_shr(drop).unwrap_or(0);
            let fraction = mantissa - whole.checked_shl(drop).unwrap_or(0);
            let up = match rounding {
                Rounding::Ceil => !negative && fraction != 0,
                Rounding::Floor => negative && fraction != 0,
                Rounding::Round => drop <= 128 && fraction >= 1 << (drop - 1),
            };
            Parts::Finite {
                negative,
                mantissa: whole + up as u128,
                exponent: 0,
            }
        }
        parts => parts,
    };
    encode(parts, value.size())
}

#[cfg(test)]
mod test {
    use super::*;

    fn int_to_f64(value: &BigInt, size: u32) -> f64 {
        to_f64(&from_int(value, size).unwrap()).unwrap()
    }

    fn f64_to_int(value: f64, size: u32) -> BigInt {
        to_int(&from_f64(value, 8).unwrap(), size).unwrap()
    }

    fn extended(sign_exponent: u64, mantissa: u64) -> BigInt {
        BigInt::from_u128(((sign_exponent as u128) << 64) | mantissa as u128, 10)
    }

    #[test]
    fn int_conversions() {
        let n = BigInt::from_u64;
        assert_eq!(int_to_f64(&n(5, 4), 8), 5.0);
        assert_eq!(int_to_f64(&n(0xfffffffb, 4), 8), -5.0);
        assert_eq!(int_to_f64(&BigInt::ones(16), 4), -1.0);
        assert_eq!(int_to_f64(&(n(3, 16) << 100), 8), 3.0 * 2f64.powi(100));
        assert_eq!(int_to_f64(&-(n(3, 16) << 100), 8), -3.0 * 2f64.powi(100));
        // 2^60 + 2^36 + 1 is just past halfway between two f32 values, and lands on the
        // halfway point once rounded to f64 first
        assert_eq!(
            int_to_f64(&n((1 << 60) + (1 << 36) + 1, 8), 4),
            ((1u64 << 60) + (1 << 37)) as f64
        );
        // The same for bits below the top 128 of a wide integer
        let wide = (n((1 << 63) + (1 << 10), 32) << 128) + n(1, 32);
        assert_eq!(
            int_to_f64(&wide, 8),
            ((1u64 << 63) + (1 << 11)) as f64 * 2f64.powi(128)
        );
        assert_eq!(
            int_to_f64(&(BigInt::from_u64(1, 64) << 500), 4),
            f64::INFINITY
        );

        assert_eq!(f64_to_int(-2.7, 4), n(0xfffffffe, 4));
        assert_eq!(f64_to_int(f64::NAN, 4), n(0, 4));
        assert_eq!(f64_to_int(1e30, 8), n(i64::MAX as u64, 8));
        assert_eq!(f64_to_int(-1e30, 8), n(i64::MIN as u64, 8));
        assert_eq!(f64_to_int(3e9, 4), n(3_000_000_000, 4));
        assert_eq!(f64_to_int(-2.7, 16), BigInt::ones(16) - n(1, 16));
        assert_eq!(
            f64_to_int(2f64.powi(100) + 2f64.powi(48), 16),
            (n(1, 16) << 100) + (n(1, 16) << 48)
        );
        assert_eq!(f64_to_int(1e300, 16), BigInt::ones(16) >> 1);
        assert_eq!(f64_to_int(f64::NEG_INFINITY, 16), !(BigInt::ones(16) >> 1));
        assert_eq!(f64_to_int(0.5, 16), n(0, 16));
        assert_eq!(f64_to_int(5e-324, 8), n(0, 8));
    }

    #[test]
    fn formats_round_trip_doubles() {
        let values = [
            0.0,
            -0.0,
            1.0,
            -2.5,
            0.1,
            1e300,
            -1e-300,
            5e-324,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for value in values {
            let wide = from_f64(value, 10).unwrap();
            assert_eq!(
                to_f64(&wide).unwrap().to_bits(),
                value.to_bits(),
                "{}",
                value
            );
            // Through the soft path for doubles and singles too
            let parts = DOUBLE.decode(value.to_bits() as u128);
            assert_eq!(DOUBLE.encode(parts) as u64, value.to_bits(), "{}", value);
            let single = Format::of(4).unwrap().encode(parts) as u32;
            assert_eq!(single, (value as f32).to_bits(), "{}", value);
        }
        assert!(to_f64(&from_f64(f64::NAN, 10).unwrap()).unwrap().is_nan());
        assert!(to_f64(&from_f64(f64::NAN, 2).unwrap()).unwrap().is_nan());
    }

    #[test]
    fn extended_precision() {
        assert_eq!(from_f64(1.0, 10).unwrap(), extended(0x3fff, 1 << 63));
        assert_eq!(from_f64(-2.0, 10).unwrap(), extended(0xc000, 1 << 63));
        assert_eq!(
            from_f64(f64::INFINITY, 10).unwrap(),
            extended(0x7fff, 1 << 63)
        );
        assert_eq!(
            from_f64(f64::NAN, 10).unwrap(),
            extended(0x7fff, 0xc000_0000_0000_0000)
        );
        // The smallest double is a normal extended number
        assert_eq!(
            from_f64(5e-324, 10).unwrap(),
            extended(0x3fff - 1074, 1 << 63)
        );
        // Rounded to double: 1 + 2^-53 is halfway to the next double and rounds to even,
        // 1 + 2^-53 + 2^-63 rounds up
        assert_eq!(
            to_f64(&extended(0x3fff, (1 << 63) | (1 << 10))).unwrap(),
            1.0
        );
        assert_eq!(
            to_f64(&extended(0x3fff, (1 << 63) | (1 << 10) | 1)).unwrap(),
            1.0 + f64::EPSILON
        );
        // Out of the double range
        assert_eq!(to_f64(&extended(0x7ffe, 1 << 63)).unwrap(), f64::INFINITY);
        assert_eq!(
            to_f64(&extended(0x8001, 1 << 63)).unwrap().to_bits(),
            (-0.0f64).to_bits()
        );
        // A denormal and the smallest normal underflow, values around the smallest double
        // round to it once
        assert_eq!(to_f64(&extended(0, 1 << 62)).unwrap().to_bits(), 0);
        assert_eq!(to_f64(&extended(1, 1 << 63)).unwrap().to_bits(), 0);
        assert_eq!(to_f64(&extended(0x3fff - 1074, 1 << 63)).unwrap(), 5e-324);
        assert_eq!(
            to_f64(&extended(0x3fff - 1075, (1 << 63) | 1)).unwrap(),
            5e-324
        );

        // Integers convert exactly, beyond double precision
        let big = (1u64 << 62) + 1;
        let float = from_int(&BigInt::from_u64(big, 8), 10).unwrap();
        assert_eq!(float, extended(0x3fff + 62, big << 1));
        assert_eq!(to_int(&float, 8).unwrap(), BigInt::from_u64(big, 8));
        let min = BigInt::from_u64(i64::MIN as u64, 8);
        let float = from_int(&min, 10).unwrap();
        assert_eq!(float, extended(0xbfff + 63, 1 << 63));
        assert_eq!(to_int(&float, 8).unwrap(), min);
        // 2^64 + 1 needs 65 bits and rounds to even
        let wide = (BigInt::from_u64(1, 16) << 64) + BigInt::from_u64(1, 16);
        assert_eq!(from_int(&wide, 10).unwrap(), extended(0x3fff + 64, 1 << 63));
        assert_eq!(
            convert(&extended(0x3fff, (1 << 63) | 1), 8).unwrap(),
            from_f64(1.0, 8).unwrap()
        );
    }

    #[test]
    fn half_precision() {
        let half = |bits: u64| BigInt::from_u64(bits, 2);
        assert_eq!(from_f64(1.0, 2).unwrap(), half(0x3c00));
        assert_eq!(from_f64(-2.0, 2).unwrap(), half(0xc000));
        assert_eq!(from_f64(65504.0, 2).unwrap(), half(0x7bff));
        assert_eq!(from_f64(65520.0, 2).unwrap(), half(0x7c00));
        assert_eq!(from_f64(2f64.powi(-24), 2).unwrap(), half(0x0001));
        assert_eq!(from_f64(2f64.powi(-25), 2).unwrap(), half(0x0000));
        assert_eq!(from_f64(1.5 * 2f64.powi(-25), 2).unwrap(), half(0x0001));
        // A subnormal rounding up into the normals
        assert_eq!(
            from_f64(2f64.powi(-14) - 2f64.powi(-26), 2).unwrap(),
            half(0x0400)
        );
        // 1 + 2^-11 is halfway and rounds to even
        assert_eq!(from_f64(1.0 + 2f64.powi(-11), 2).unwrap(), half(0x3c00));
        assert_eq!(to_f64(&half(0x3555)).unwrap(), 0.333251953125);
        assert_eq!(to_f64(&half(0x0001)).unwrap(), 2f64.powi(-24));
        assert_eq!(to_f64(&half(0xfc00)).unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn whole_numbers() {
        let round = |value: f64, rounding| {
            to_f64(&round_to_whole(&from_f64(value, 8).unwrap(), rounding).unwrap()).unwrap()
        };
        use Rounding::*;
        assert_eq!(round(1.5, Ceil), 2.0);
        assert_eq!(round(-1.5, Ceil), -1.0);
        assert_eq!(round(-0.5, Ceil).to_bits(), (-0.0f64).to_bits());
        assert_eq!(round(1.5, Floor), 1.0);
        assert_eq!(round(-1.5, Floor), -2.0);
        assert_eq!(round(-1e-300, Floor), -1.0);
        assert_eq!(round(2.5, Round), 3.0);
        assert_eq!(round(-2.5, Round), -3.0);
        assert_eq!(round(0.49999999999999994, Round), 0.0);
        assert_eq!(round(1e300, Round), 1e300);
        assert!(round(f64::NAN, Floor).is_nan());
        // Extended precision keeps the bits a double would lose
        let value = extended(0x3fff + 62, ((1 << 62) + 1) << 1 | 1);
        assert_eq!(
            round_to_whole(&value, Floor).unwrap(),
            extended(0x3fff + 62, ((1 << 62) + 1) << 1)
        );
    }
}
