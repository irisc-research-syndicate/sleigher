use std::ops::Range;

use sleigh_rs::pattern::BitConstraint;
use sleigh_rs::{ContextId, Sleigh};

use crate::value::sign_extend;

/// The values of a spec's context variables, packed the way sleigh-rs packs them for the
/// context half of a constructor's bit pattern
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Context {
    bits: Vec<bool>,
}

impl Context {
    /// Every context variable zero
    pub fn new(sleigh: &Sleigh) -> Self {
        Self {
            bits: vec![false; sleigh.context_memory().memory_bits as usize],
        }
    }

    /// The context variable called `name`
    pub fn id(sleigh: &Sleigh, name: &str) -> Option<ContextId> {
        sleigh.global_scope_by_name(name)?.context()
    }

    /// Where the bits of `id` are, most significant first
    fn range(sleigh: &Sleigh, id: ContextId) -> Range<usize> {
        let bits = sleigh.context_memory().context(id);
        bits.start() as usize..bits.end().get() as usize
    }

    pub fn get(&self, sleigh: &Sleigh, id: ContextId) -> i64 {
        let field = &self.bits[Self::range(sleigh, id)];
        let value = field
            .iter()
            .fold(0u64, |value, bit| (value << 1) | *bit as u64);
        if sleigh.context(id).is_signed() {
            sign_extend(value, field.len() as u32)
        } else {
            value as i64
        }
    }

    pub fn set(&mut self, sleigh: &Sleigh, id: ContextId, value: i64) {
        let field = &mut self.bits[Self::range(sleigh, id)];
        let len = field.len();
        for (i, bit) in field.iter_mut().enumerate() {
            *bit = (value >> (len - 1 - i)) & 1 != 0;
        }
    }

    /// Reset the variables that do not flow on to the next instruction
    pub fn clear_noflow(&mut self, sleigh: &Sleigh) {
        for (i, context) in sleigh.contexts().iter().enumerate() {
            if context.noflow {
                self.set(sleigh, ContextId(i), 0);
            }
        }
    }

    /// Whether the context satisfies the context bits of a constructor variant
    pub fn matches(&self, constraints: &[BitConstraint]) -> bool {
        debug_assert_eq!(constraints.len(), self.bits.len());
        constraints
            .iter()
            .zip(&self.bits)
            .all(|(constraint, bit)| constraint.value().is_none_or(|value| value == *bit))
    }
}

/// A `globalset`: from `address` on, the context variable has `value`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextCommit {
    pub address: u64,
    pub context: ContextId,
    pub value: i64,
}

#[cfg(test)]
mod test {
    use super::*;

    fn load() -> Sleigh {
        sleigh_rs::file_to_sleigh("examples/context.slaspec".as_ref()).unwrap()
    }

    fn id(sleigh: &Sleigh, name: &str) -> ContextId {
        Context::id(sleigh, name).unwrap()
    }

    #[test]
    fn get_set() {
        let sleigh = load();
        let mut context = Context::new(&sleigh);
        let names = ["mode", "shift", "width"];
        for name in names {
            assert_eq!(context.get(&sleigh, id(&sleigh, name)), 0);
        }
        context.set(&sleigh, id(&sleigh, "mode"), 1);
        context.set(&sleigh, id(&sleigh, "shift"), 2);
        context.set(&sleigh, id(&sleigh, "width"), 1);
        assert_eq!(context.get(&sleigh, id(&sleigh, "mode")), 1);
        assert_eq!(context.get(&sleigh, id(&sleigh, "shift")), 2);
        assert_eq!(context.get(&sleigh, id(&sleigh, "width")), 1);

        // Values are truncated to the variable and leave the neighbours alone
        context.set(&sleigh, id(&sleigh, "shift"), 7);
        assert_eq!(context.get(&sleigh, id(&sleigh, "shift")), 3);
        context.set(&sleigh, id(&sleigh, "shift"), 0);
        assert_eq!(context.get(&sleigh, id(&sleigh, "mode")), 1);
        assert_eq!(context.get(&sleigh, id(&sleigh, "width")), 1);

        assert!(Context::id(&sleigh, "r0").is_none());
        assert!(Context::id(&sleigh, "nope").is_none());
    }

    /// `add` and `sub` share an opcode and only differ in the mode their pattern requires
    #[test]
    fn matches_constructor_variants() {
        let sleigh = load();
        let instruction = sleigh.table(sleigh.instruction_table());
        let accepts = |mnemonic: &str, context: &Context| {
            instruction
                .matcher_order()
                .iter()
                .filter(|matcher| {
                    let constructor = instruction.constructor(matcher.constructor);
                    constructor.display.mneumonic.as_deref() == Some(mnemonic)
                })
                .any(|matcher| {
                    let constructor = instruction.constructor(matcher.constructor);
                    context.matches(constructor.variant(matcher.variant_id).0)
                })
        };

        let mut context = Context::new(&sleigh);
        assert!(accepts("add", &context));
        assert!(!accepts("sub", &context));
        assert!(accepts("mov", &context));

        context.set(&sleigh, id(&sleigh, "mode"), 1);
        assert!(!accepts("add", &context));
        assert!(accepts("sub", &context));
        assert!(accepts("mov", &context));
    }

    /// The `SRC` operand of `mov` is a register or an immediate depending on `width`
    #[test]
    fn matches_subtable_variants() {
        let sleigh = load();
        let src = sleigh
            .tables()
            .iter()
            .find(|table| table.name() == "SRC")
            .unwrap();
        let accepted = |context: &Context| {
            src.matcher_order()
                .iter()
                .filter(|matcher| {
                    let constructor = src.constructor(matcher.constructor);
                    context.matches(constructor.variant(matcher.variant_id).0)
                })
                .map(|matcher| matcher.constructor)
                .collect::<Vec<_>>()
        };

        let mut context = Context::new(&sleigh);
        let register = accepted(&context);
        context.set(&sleigh, id(&sleigh, "width"), 1);
        let immediate = accepted(&context);
        assert_eq!(register.len(), 1);
        assert_eq!(immediate.len(), 1);
        assert_ne!(register, immediate);
    }
}
