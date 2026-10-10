use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ops::Range;

use sleigh_rs::pattern::BitConstraint;
use sleigh_rs::{ContextId, Sleigh};

use crate::value::{parse_number, sign_extend};

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

    /// Every context variable zero except the named `values`
    pub fn from_values(sleigh: &Sleigh, values: &[(&str, i64)]) -> anyhow::Result<Self> {
        let mut context = Self::new(sleigh);
        for (name, value) in values {
            let id = Self::id(sleigh, name)
                .ok_or_else(|| anyhow::anyhow!("no context variable {:?}", name))?;
            context.set(sleigh, id, *value);
        }
        Ok(context)
    }

    /// Every context variable zero except those set by `name=value` arguments, values in
    /// decimal or `0x` hex
    pub fn parse_values(sleigh: &Sleigh, args: &[String]) -> anyhow::Result<Self> {
        let values = args
            .iter()
            .map(|arg| {
                let (name, value) = arg
                    .split_once('=')
                    .ok_or_else(|| anyhow::anyhow!("{:?} is not name=value", arg))?;
                let value = match parse_number(true, value.trim()) {
                    Some(("", value)) => value,
                    _ => anyhow::bail!("{:?} is not a number", value),
                };
                Ok((name.trim(), value))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Self::from_values(sleigh, &values)
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

/// The context as it flows from instruction to instruction, and the values `globalset`s
/// committed to addresses
#[derive(Debug, Clone)]
pub struct ContextFlow {
    /// The context the next instruction starts from, before the commits to its address.
    /// Unlike Ghidra's context, which is kept by address, it follows the instructions in the
    /// order they are visited, unless context was flowed to the address with `flow_to`.
    pub next: Context,
    /// The context kept for addresses `flow_to` reached first
    flowed: HashMap<u64, Context>,
    commits: HashMap<u64, HashMap<ContextId, i64>>,
}

impl ContextFlow {
    pub fn new(context: Context) -> Self {
        Self {
            next: context,
            flowed: HashMap::new(),
            commits: HashMap::new(),
        }
    }

    /// The context to decode the instruction at `address` with
    pub fn at(&self, sleigh: &Sleigh, address: u64) -> Context {
        let mut context = self.flowed.get(&address).unwrap_or(&self.next).clone();
        for (&id, &value) in self.commits.get(&address).into_iter().flatten() {
            context.set(sleigh, id, value);
        }
        context
    }

    /// Keep `next` for `address`, as Ghidra keeps context by address, unless some context
    /// already flowed there. Called after `advance`, for each address the instruction flows
    /// to. Returns whether it flowed.
    pub fn flow_to(&mut self, address: u64) -> bool {
        match self.flowed.entry(address) {
            Entry::Occupied(_) => false,
            Entry::Vacant(entry) => {
                entry.insert(self.next.clone());
                true
            }
        }
    }

    /// Record what `globalset`s commit
    pub fn commit(&mut self, commits: &[ContextCommit]) {
        for commit in commits {
            self.commits
                .entry(commit.address)
                .or_default()
                .insert(commit.context, commit.value);
        }
    }

    /// Move on from an instruction decoded with `context` that made `commits`. Its own
    /// context changes do not flow on, only what it commits.
    pub fn advance(&mut self, sleigh: &Sleigh, mut context: Context, commits: &[ContextCommit]) {
        self.commit(commits);
        context.clear_noflow(sleigh);
        self.next = context;
    }
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

    #[test]
    fn parse_values() {
        let sleigh = load();
        let values = ["mode=1", "shift=0x3", "bank = 1"].map(String::from);
        let context = Context::parse_values(&sleigh, &values).unwrap();
        assert_eq!(context.get(&sleigh, id(&sleigh, "mode")), 1);
        assert_eq!(context.get(&sleigh, id(&sleigh, "shift")), 3);
        assert_eq!(context.get(&sleigh, id(&sleigh, "bank")), 1);
        assert_eq!(context.get(&sleigh, id(&sleigh, "width")), 0);

        for bad in ["mode", "mode=x", "nope=1", "r0=1"] {
            assert!(
                Context::parse_values(&sleigh, &[bad.to_string()]).is_err(),
                "{:?}",
                bad
            );
        }
    }

    #[test]
    fn flow_to() {
        let sleigh = load();
        let mode = id(&sleigh, "mode");
        let shift = id(&sleigh, "shift");
        let mut flow = ContextFlow::new(Context::new(&sleigh));
        let first = flow.at(&sleigh, 0x0);
        let commits = [ContextCommit {
            address: 0x8,
            context: shift,
            value: 1,
        }];
        flow.advance(&sleigh, first, &commits);

        // An address keeps the first context that flowed to it, commits apply on top
        assert!(flow.flow_to(0x8));
        flow.next.set(&sleigh, mode, 1);
        assert!(!flow.flow_to(0x8));
        let kept = flow.at(&sleigh, 0x8);
        assert_eq!(kept.get(&sleigh, mode), 0);
        assert_eq!(kept.get(&sleigh, shift), 1);
        // Elsewhere the context follows the visit order
        assert_eq!(flow.at(&sleigh, 0xc).get(&sleigh, mode), 1);
    }
}
