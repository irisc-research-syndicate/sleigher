use std::collections::{BTreeMap, HashMap};

use anyhow::{bail, Result};

use sleigh_rs::disassembly::{
    AddrScope, Assertation, Expr, ExprElement, Op, OpUnary, ReadScope, VariableId, WriteScope,
};
use sleigh_rs::display::DisplayElement;
use sleigh_rs::execution::Statement;
use sleigh_rs::meaning::Meaning;
use sleigh_rs::pattern::{BitConstraint, Block, CmpOp, ConstraintValue, Verification};
use sleigh_rs::table::{Constructor, Table};
use sleigh_rs::{ContextId, Endian, Number, PrintBase, Sleigh, Span, TableId, TokenFieldId};

use crate::context::{Context, ContextCommit, ContextFlow};
use crate::pcode::{BranchTarget, LiftError, OpCode, PcodeOp, VarnodeSpace};
use crate::value::sign_extend;

#[derive(Debug, Clone)]
pub struct Disassembler<'sleigh> {
    pub sleigh: &'sleigh Sleigh,
}

impl<'sleigh> std::ops::Deref for Disassembler<'sleigh> {
    type Target = Sleigh;

    fn deref(&self) -> &Self::Target {
        self.sleigh
    }
}

impl<'sleigh> Disassembler<'sleigh> {
    pub fn new(sleigh: &'sleigh Sleigh) -> Self {
        warn_unsupported(sleigh);
        Disassembler { sleigh }
    }

    pub fn disassemble(
        &'sleigh self,
        inst_start: u64,
        context: &Context,
        bytes: &[u8],
    ) -> Result<DisassembledInstruction<'sleigh>> {
        self.disassemble_in_flow(inst_start, &mut ContextFlow::new(context.clone()), bytes)
    }

    /// Like `disassemble`, with the context `flow` has for `inst_start`. The instructions
    /// filling a delay slot are decoded from the bytes that follow, with the context flowing
    /// to them, and `flow` moves on past them.
    pub fn disassemble_in_flow(
        &'sleigh self,
        inst_start: u64,
        flow: &mut ContextFlow,
        bytes: &[u8],
    ) -> Result<DisassembledInstruction<'sleigh>> {
        let (table, commits) = self.decode(inst_start, flow, bytes)?;
        let delay_slot_len = table.delay_slot_len();
        let mut delay_slots = vec![];
        let mut fallthrough = table.inst_next;
        let mut delay_slot_error = None;
        while fallthrough < table.inst_next + delay_slot_len {
            let offset = (fallthrough - inst_start) as usize;
            let address = fallthrough;
            let err = match self.decode(address, flow, &bytes[offset..]) {
                Ok((slot, _)) if slot.len == 0 => "empty".to_string(),
                Ok((slot, _)) => {
                    // A nested delay slot is still listed as in the delay slot
                    let nested = slot.delay_slot_len() > 0;
                    fallthrough = slot.inst_next;
                    delay_slots.push(slot);
                    if !nested {
                        continue;
                    }
                    "nested delay slot".to_string()
                }
                Err(err) => err.to_string(),
            };
            delay_slot_error = Some(LiftError::Invalid(format!(
                "delay slot at {:#x}: {}",
                address, err
            )));
            break;
        }

        let pcode = match delay_slot_error {
            Some(err) => Err(err),
            None => crate::pcode::lift(&table, fallthrough, &delay_slots),
        };
        if let Err(err) = &pcode {
            log::debug!("Could not lift {}: {}", table, err);
        }
        Ok(DisassembledInstruction {
            table,
            pcode,
            commits,
            delay_slots,
        })
    }

    /// One instruction, without its delay slot, with the context `flow` has for `inst_start`;
    /// `flow` moves on past it
    fn decode(
        &'sleigh self,
        inst_start: u64,
        flow: &mut ContextFlow,
        bytes: &[u8],
    ) -> Result<(DisassembledTable<'sleigh>, Vec<ContextCommit>)> {
        let context = flow.at(self, inst_start);
        let table = self.disassemble_table(
            inst_start,
            self.table(self.instruction_table()),
            &mut context.clone(),
            bytes,
        )?;
        let mut commits = vec![];
        table.commits(&mut commits);
        flow.advance(self, context, &commits);
        Ok((table, commits))
    }

    /// Disassemble the code in `bytes`, loaded at `base`, by following control flow from
    /// `entry` and keeping the context by address, as Ghidra does: each address is decoded
    /// with the context that flowed to it first and what `globalset`s committed to it, so a
    /// commit to a branch destination is seen wherever the destination is in the code.
    ///
    /// Each address is decoded once. A commit to an address that was already decoded is not
    /// seen there, nor at the addresses its context flowed on to. Word-addressed code spaces
    /// are not supported.
    pub fn disassemble_flow(
        &'sleigh self,
        base: u64,
        bytes: &[u8],
        entry: u64,
        context: &Context,
    ) -> Result<BTreeMap<u64, DisassembledInstruction<'sleigh>>> {
        let wordsize = self.space(self.default_space()).wordsize.get();
        if wordsize != 1 {
            bail!(
                "word-addressed code (wordsize {}) is not supported",
                wordsize
            );
        }
        let mut flow = ContextFlow::new(context.clone());
        flow.flow_to(entry);
        let mut instructions = BTreeMap::new();
        let mut pending = vec![entry];
        while let Some(address) = pending.pop() {
            let Some(code) = address
                .checked_sub(base)
                .and_then(|offset| bytes.get(offset as usize..))
            else {
                log::debug!("{:#x}: outside the code", address);
                continue;
            };
            let instruction = match self.disassemble_in_flow(address, &mut flow, code) {
                Ok(instruction) => instruction,
                Err(err) => {
                    log::debug!("{:#x}: {}", address, err);
                    continue;
                }
            };
            // Pushed last, the fall-through is followed first
            for target in instruction.flows().into_iter().rev() {
                if flow.flow_to(target) {
                    pending.push(target);
                }
            }
            let first = instructions.insert(address, instruction).is_none();
            debug_assert!(first, "{:#x} decoded twice", address);
        }
        Ok(instructions)
    }

    pub fn disassemble_table(
        &'sleigh self,
        inst_start: u64,
        table: &'sleigh Table,
        context: &mut Context,
        bytes: &[u8],
    ) -> Result<DisassembledTable<'sleigh>> {
        let mut disassembled =
            DisassembledTable::disassemble(self, inst_start, table, context, bytes)?;
        // inst_next is the address of the next instruction, which is only known
        // once the whole instruction (including all subtables) has been matched.
        disassembled.resolve(inst_start + disassembled.len as u64, context)?;
        disassembled.set_context(context);
        Ok(disassembled)
    }

    /// The most bytes any delay slot in the spec needs
    pub fn max_delay_slot_len(&self) -> u64 {
        self.tables()
            .iter()
            .flat_map(|table| table.constructors())
            .map(delay_slot_len)
            .max()
            .unwrap_or(0)
    }

    pub fn extract_token_field(&self, token_field_id: TokenFieldId, bytes: &[u8]) -> i64 {
        let token_field = self.token_field(token_field_id);
        let token = self.token(token_field.token);
        let token_bytes = &bytes[..token.len_bytes().get() as usize];

        log::trace!(
            "Extracting {} from {}/{:?}: {:02x?}",
            token_field.name(),
            token.name(),
            token.endian,
            token_bytes
        );

        let token_value = match token.endian {
            Endian::Little => token_bytes
                .iter()
                .rev()
                .fold(0u64, |n, b| (n << 8) | (*b as u64)),
            Endian::Big => token_bytes.iter().fold(0u64, |n, b| (n << 8) | (*b as u64)),
        };

        log::trace!("Token value: {:#x}", token_value);

        let token_field_raw =
            token_value >> token_field.bits.start() & ((1 << token_field.bits.len().get()) - 1);

        log::trace!("Token field raw: {:#x}", token_field_raw);

        let token_field_value = if token_field.raw_value_is_signed() {
            sign_extend(token_field_raw, token_field.bits.len().get() as u32)
        } else {
            token_field_raw as i64
        };

        log::trace!("Token field value: {:#x}", token_field_value);

        token_field_value
    }
}

#[derive(Debug, Clone)]
pub struct DisassembledInstruction<'sleigh> {
    pub table: DisassembledTable<'sleigh>,
    /// The instruction's semantics, or why they could not be lifted
    pub pcode: Result<Vec<PcodeOp>, LiftError>,
    /// Context the instruction sets for other addresses
    pub commits: Vec<ContextCommit>,
    /// The instructions filling the delay slot, whose p-code is part of this instruction's
    pub delay_slots: Vec<DisassembledTable<'sleigh>>,
}

impl DisassembledInstruction<'_> {
    /// Where execution continues when the instruction does not branch: past its delay slot.
    /// This is `inst_next` in its p-code, while its disassembly sees its own end.
    pub fn fallthrough(&self) -> u64 {
        self.delay_slots
            .last()
            .map_or(self.inst_next, |slot| slot.inst_next)
    }

    /// Where control can go after the instruction: `fallthrough` first if it can fall
    /// through, then the destinations of its direct branches and calls in the default space.
    /// Without p-code it is taken to fall through.
    pub fn flows(&self) -> Vec<u64> {
        let Ok(ops) = &self.pcode else {
            return vec![self.fallthrough()];
        };
        let mut targets = vec![];
        // Follow the branches between the ops to see if a path reaches the end
        let mut reached = vec![false; ops.len() + 1];
        let mut pending = vec![0];
        while let Some(index) = pending.pop() {
            if index > ops.len() || std::mem::replace(&mut reached[index], true) {
                continue;
            }
            let Some(op) = ops.get(index) else {
                continue;
            };
            match op.branch_target() {
                Some(BranchTarget::Relative(offset)) => {
                    pending.extend(index.checked_add_signed(offset as isize));
                }
                Some(BranchTarget::Address(space, address))
                    if space == VarnodeSpace::Space(self.default_space()) =>
                {
                    targets.push(address);
                }
                _ => {}
            }
            if !matches!(
                op.opcode,
                OpCode::Branch | OpCode::BranchInd | OpCode::Return
            ) {
                pending.push(index + 1);
            }
        }
        let mut flows = match reached[ops.len()] {
            true => vec![self.fallthrough()],
            false => vec![],
        };
        for target in targets {
            if !flows.contains(&target) {
                flows.push(target);
            }
        }
        flows
    }
}

impl<'sleigh> std::ops::Deref for DisassembledInstruction<'sleigh> {
    type Target = DisassembledTable<'sleigh>;

    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl<'sleigh> std::fmt::Display for DisassembledInstruction<'sleigh> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.table, f)
    }
}

/// Where a `globalset` commits to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitTarget {
    /// An address known when the `globalset` runs
    Address(u64),
    /// The address an operand table exports, known once the whole instruction is decoded
    Table(TableId),
}

#[derive(Debug, Clone)]
pub struct DisassembledTable<'sleigh> {
    pub disassembler: &'sleigh Disassembler<'sleigh>,
    pub inst_start: u64,
    pub inst_next: u64,
    pub table: &'sleigh Table,
    pub constructor: &'sleigh Constructor,
    pub token_fields: HashMap<TokenFieldId, i64>,
    pub tables: HashMap<TableId, DisassembledTable<'sleigh>>,
    pub variables: HashMap<VariableId, i64>,
    /// The context once the whole instruction is decoded
    pub context: Context,
    /// The `globalset`s of this constructor, the values are taken from the final context
    pub globalsets: Vec<(CommitTarget, ContextId)>,
    pub len: usize,
    pub bytes: Vec<u8>,
}

impl<'sleigh> std::ops::Deref for DisassembledTable<'sleigh> {
    type Target = Disassembler<'sleigh>;

    fn deref(&self) -> &Self::Target {
        self.disassembler
    }
}

pub fn bitconstraint_to_string(constraints: &[BitConstraint]) -> String {
    constraints
        .iter()
        .map(|constraint| match constraint {
            BitConstraint::Unrestrained => "x",
            BitConstraint::Defined(false) => "0",
            BitConstraint::Defined(true) => "1",
            BitConstraint::Restrained => "r",
        })
        .collect::<Vec<_>>()
        .join("")
}

impl<'sleigh> DisassembledTable<'sleigh> {
    pub fn disassemble(
        disassembler: &'sleigh Disassembler<'sleigh>,
        inst_start: u64,
        table: &'sleigh Table,
        context: &mut Context,
        bytes: &[u8],
    ) -> Result<Self> {
        log::debug!("Disassembling {} table", table.name());

        'match_loop: for matcher in table.matcher_order() {
            let mut bytes = bytes;
            let constructor = table.constructor(matcher.constructor);

            log::trace!(
                "Trying to match {:?} table to bytes {:02x?}",
                table.name(),
                bytes
            );
            if bytes.len()
                < constructor
                    .pattern
                    .len
                    .single_len()
                    .unwrap_or(constructor.pattern.len.min()) as usize
            {
                log::trace!(
                    "{}: too few bytes to match constructor, continuing to next matcher",
                    table.name()
                );
                continue 'match_loop;
            }

            let (context_constraints, data_constraints) = constructor.variant(matcher.variant_id);

            if log::log_enabled!(log::Level::Trace) {
                log::trace!(
                    "Constext constraint: {}",
                    bitconstraint_to_string(context_constraints)
                );
                log::trace!(
                    "Data constraint: {}",
                    bitconstraint_to_string(data_constraints)
                );
            }

            if !context.matches(context_constraints) {
                log::trace!(
                    "{}: context bitconstraint failed to match, continuing to next matcher",
                    table.name()
                );
                continue 'match_loop;
            }

            for (byte_constraint, byte) in data_constraints.chunks(8).zip(bytes) {
                for (bit, constraint) in byte_constraint.iter().enumerate() {
                    if let Some(expected_bit) = constraint.value() {
                        if expected_bit != ((byte >> bit) & 1 == 1) {
                            log::trace!("{}: Token bitconstraint failed to match, continuing to next matcher", table.name());
                            continue 'match_loop;
                        }
                    }
                }
            }

            if let Some(mneumonic) = &constructor.display.mneumonic {
                log::debug!("Constructor {:?} matched token bitconstraint", mneumonic);
            }

            let mut disasm_table = DisassembledTable {
                disassembler,
                table,
                constructor,
                inst_start,
                inst_next: inst_start,
                token_fields: HashMap::new(),
                tables: HashMap::new(),
                variables: HashMap::new(),
                context: context.clone(),
                globalsets: Vec::new(),
                len: 0,
                bytes: Vec::new(),
            };
            // The constructor's context changes are seen by the subtables built after them,
            // and are dropped if it does not match
            let mut matched_context = context.clone();

            for block in constructor.pattern.blocks() {
                let mut block_len = block.len().single_len().unwrap_or(block.len().min()) as usize;
                if bytes.len() < block_len {
                    log::trace!(
                        "{}: too few bytes to match block, continuing to next matcher",
                        table.name()
                    );
                    continue 'match_loop;
                }

                for produced_token_field in block.token_fields() {
                    let token_field_value =
                        disassembler.extract_token_field(produced_token_field.field, bytes);
                    disasm_table
                        .token_fields
                        .insert(produced_token_field.field, token_field_value);
                }

                if let Err(err) = disasm_table.apply_assertions(
                    block.pre_disassembler(),
                    &mut matched_context,
                    bytes,
                ) {
                    log::trace!("{}, continuing to next matcher", err);
                    continue 'match_loop;
                }

                for produced_table in block.tables() {
                    let subtable = disassembler.table(produced_table.table);
                    if let Ok(subtable_value) = DisassembledTable::disassemble(
                        disassembler,
                        inst_start,
                        subtable,
                        &mut matched_context,
                        bytes,
                    ) {
                        block_len = block_len.max(subtable_value.len);
                        disasm_table
                            .tables
                            .insert(produced_table.table, subtable_value);
                    } else {
                        log::trace!(
                            "{}: failed to disassemble subtable {}, continuing to next matcher",
                            table.name(),
                            subtable.name()
                        );
                        continue 'match_loop;
                    }
                }

                if !disasm_table.verify_block(block, context, bytes) {
                    log::trace!(
                        "{}: failed verification, continuing to next matcher",
                        table.name()
                    );
                    continue 'match_loop;
                }
                if bytes.len() < block_len {
                    log::trace!(
                        "{}: too few bytes for block, continuing to next matcher",
                        table.name()
                    );
                    continue 'match_loop;
                }
                if let Err(err) = disasm_table.apply_assertions(
                    block.post_disassembler(),
                    &mut matched_context,
                    bytes,
                ) {
                    log::trace!("{}, continuing to next matcher", err);
                    continue 'match_loop;
                }
                disasm_table.bytes.extend_from_slice(&bytes[..block_len]);
                disasm_table.len += block_len;
                bytes = &bytes[block_len..];
            }
            // A placeholder until `resolve` knows the length of the whole instruction
            disasm_table.inst_next = inst_start + disasm_table.len as u64;
            *context = matched_context;
            return Ok(disasm_table);
        }
        bail!("{}: Failed to disassemble table", table.name());
    }

    /// Whether the bytes at the start of a block and the context pass its verifications: all
    /// of them in an AND block, any branch in an OR block
    fn verify_block(&self, block: &Block, context: &Context, bytes: &[u8]) -> bool {
        match block {
            Block::And { verifications, .. } => verifications
                .iter()
                .all(|verification| self.verify(verification, context, bytes)),
            Block::Or { branches, .. } => branches
                .iter()
                .any(|branch| self.verify(branch, context, bytes)),
        }
    }

    fn verify(&self, verification: &Verification, context: &Context, bytes: &[u8]) -> bool {
        match verification {
            Verification::ContextCheck {
                context: context_id,
                op,
                value,
            } => {
                let name = self.context(*context_id).name();
                let context_value = context.get(self, *context_id);
                self.check(name, context_value, *op, value, context, bytes)
            }
            // Every subtable is decoded whichever OR branch matches, as in Ghidra
            Verification::TableBuild {
                produced_table: _,
                verification: _,
            } => true,
            Verification::TokenFieldCheck { field, op, value } => {
                let name = self.token_field(*field).name();
                let field_value = self.extract_token_field(*field, bytes);
                self.check(name, field_value, *op, value, context, bytes)
            }
            // A parenthesised pattern, verified like a block of the constructor's pattern
            Verification::SubPattern {
                location: _,
                pattern,
            } => match pattern.blocks() {
                [block] => self.verify_block(block, context, bytes),
                // The fields and subtables of later blocks would be read from the wrong offset,
                // see `warn_unsupported`
                _ => false,
            },
        }
    }

    /// Compare the value of `name` with a constraint value, tracing a failure
    fn check(
        &self,
        name: &str,
        value: i64,
        op: CmpOp,
        check: &ConstraintValue,
        context: &Context,
        bytes: &[u8],
    ) -> bool {
        let Some(check_value) = self.evaluate_expr(check.expr(), context, bytes) else {
            log::trace!("{}: division by zero checking {}", self.table.name(), name);
            return false;
        };
        let passed = compare(op, value, check_value);
        if !passed {
            log::trace!(
                "{}: failed verification {}={} {:?} {}",
                self.table.name(),
                name,
                value,
                op,
                check_value
            );
        }
        passed
    }

    /// Run the actions that need `inst_next`, once the whole instruction is matched. Unlike
    /// Ghidra, subtables run theirs before the parent; only context written by both differs.
    fn resolve(&mut self, inst_next: u64, context: &mut Context) -> Result<()> {
        self.inst_next = inst_next;
        for subtable in self.tables.values_mut() {
            subtable.resolve(inst_next, context)?;
        }
        let bytes = std::mem::take(&mut self.bytes);
        let result = self.apply_assertions(
            self.constructor.pattern.disassembly_pos_match(),
            context,
            &bytes,
        );
        self.bytes = bytes;
        result
    }

    fn set_context(&mut self, context: &Context) {
        self.context = context.clone();
        for subtable in self.tables.values_mut() {
            subtable.set_context(context);
        }
    }

    /// The bytes the delay slot of this table and its subtables needs, 0 without one. Ghidra
    /// takes the last constructor with a delay slot it resolves; specs have at most one.
    pub fn delay_slot_len(&self) -> u64 {
        self.tables
            .values()
            .map(|subtable| subtable.delay_slot_len())
            .fold(delay_slot_len(self.constructor), u64::max)
    }

    /// The `globalset`s of this table and its subtables
    fn commits(&self, commits: &mut Vec<ContextCommit>) {
        for (target, context) in self.globalsets.iter() {
            match self.commit_address(*target) {
                Ok(address) => commits.push(ContextCommit {
                    address,
                    context: *context,
                    value: self.context.get(self, *context),
                }),
                Err(err) => log::warn!("{}: globalset skipped: {}", self.table.name(), err),
            }
        }
        for subtable in self.tables.values() {
            subtable.commits(commits);
        }
    }

    /// Where a `globalset` commits to. A table names the address it exports, as libsla does,
    /// which may depend on `inst_next`, so it is only known once the instruction is decoded.
    fn commit_address(&self, target: CommitTarget) -> Result<u64, LiftError> {
        match target {
            CommitTarget::Address(address) => Ok(address),
            CommitTarget::Table(table_id) => {
                let table = self.tables.get(&table_id).ok_or_else(|| {
                    LiftError::Invalid(format!("table {} is not an operand", table_id.0))
                })?;
                crate::pcode::export_address(table)
            }
        }
    }

    /// Fails when an assignment divides by zero
    fn apply_assertions(
        &mut self,
        assertions: &[Assertation],
        context: &mut Context,
        bytes: &[u8],
    ) -> Result<()> {
        for assertion in assertions {
            match assertion {
                Assertation::GlobalSet(global_set) => {
                    let target = match global_set.address {
                        AddrScope::Integer(address) => CommitTarget::Address(address),
                        AddrScope::InstStart(_) => CommitTarget::Address(self.inst_start),
                        AddrScope::InstNext(_) => CommitTarget::Address(self.inst_next),
                        AddrScope::Local(variable_id) => {
                            CommitTarget::Address(self.variables[&variable_id] as u64)
                        }
                        AddrScope::Table(table_id) => CommitTarget::Table(table_id),
                    };
                    self.globalsets.push((target, global_set.context));
                }
                Assertation::Assignment(assignment) => {
                    let Some(value) = self.evaluate_expr(&assignment.right, context, bytes) else {
                        bail!("{}: division by zero in an assignment", self.table.name());
                    };
                    match assignment.left {
                        WriteScope::Context(context_id) => context.set(self, context_id, value),
                        WriteScope::Local(variable_id) => {
                            self.variables.insert(variable_id, value);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The value of a pattern expression, wrapping like Ghidra's 64-bit arithmetic, `None` when
    /// it divides by zero
    pub fn evaluate_expr(&self, expr: &Expr, context: &Context, bytes: &[u8]) -> Option<i64> {
        let value = match expr {
            Expr::Value(expr_element) => match expr_element {
                ExprElement::Value { value, location: _ } => match *value {
                    ReadScope::Integer(number) => match number {
                        sleigh_rs::Number::Positive(x) => x as i64,
                        sleigh_rs::Number::Negative(x) => (x as i64).wrapping_neg(),
                    },
                    ReadScope::Context(context_id) => context.get(self, context_id),
                    ReadScope::TokenField(token_field_id) => self
                        .token_fields
                        .get(&token_field_id)
                        .copied()
                        .unwrap_or_else(|| {
                            self.disassembler.extract_token_field(token_field_id, bytes)
                        }),
                    ReadScope::InstStart(_inst_start) => self.inst_start as i64,
                    ReadScope::InstNext(_inst_next) => self.inst_next as i64,
                    ReadScope::Local(variable_id) => *self.variables.get(&variable_id).unwrap(),
                },
                ExprElement::Op(_, op_unary, expr) => {
                    let expr = self.evaluate_expr(expr, context, bytes)?;
                    match op_unary {
                        OpUnary::Negation => !expr,
                        OpUnary::Negative => expr.wrapping_neg(),
                    }
                }
            },
            Expr::Op(_span, op, expr, expr1) => {
                let l = self.evaluate_expr(expr, context, bytes)?;
                let r = self.evaluate_expr(expr1, context, bytes)?;
                // Shift amounts are taken modulo 64, as Java does
                match op {
                    Op::Add => l.wrapping_add(r),
                    Op::Sub => l.wrapping_sub(r),
                    Op::Mul => l.wrapping_mul(r),
                    Op::Div if r == 0 => return None,
                    Op::Div => l.wrapping_div(r),
                    Op::And => l & r,
                    Op::Or => l | r,
                    Op::Xor => l ^ r,
                    Op::Asr => l.wrapping_shr(r as u32),
                    Op::Lsl => l.wrapping_shl(r as u32),
                }
            }
        };
        Some(value)
    }
}

impl<'sleigh> std::fmt::Display for DisassembledTable<'sleigh> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();

        if let Some(mneumonic) = &self.constructor.display.mneumonic {
            parts.push(mneumonic.to_string())
        }

        for display_element in self.constructor.display.elements() {
            parts.push(match display_element {
                DisplayElement::Varnode(varnode_id) => {
                    self.disassembler.varnode(*varnode_id).name().to_string()
                }
                DisplayElement::Context(context_id) => {
                    let value = self.context.get(self, *context_id);
                    self.fmt_meaning(self.disassembler.context(*context_id).meaning(), value)
                }
                DisplayElement::TokenField(token_field_id) => {
                    let token_field = self.disassembler.token_field(*token_field_id);
                    match self.token_fields.get(token_field_id) {
                        Some(value) => self.fmt_meaning(token_field.meaning(), *value),
                        None => format!(
                            "<UNDEFINED FIELD {}/{}>",
                            token_field.name(),
                            token_field_id.0
                        ),
                    }
                }
                DisplayElement::InstStart(_inst_start) => format!("{:x}", self.inst_start),
                DisplayElement::InstNext(_inst_next) => format!("{:x}", self.inst_next),
                DisplayElement::Table(table_id) => {
                    format!("{}", self.tables.get(table_id).unwrap())
                }
                DisplayElement::Disassembly(variable_id) => {
                    if let Some(variable) = self.variables.get(variable_id) {
                        fmt_hex(*variable)
                    } else {
                        format!("<UNDEFINED DISASSEMBLY VAR {}>", variable_id.0)
                    }
                }
                DisplayElement::Literal(lit) => lit.to_string(),
                DisplayElement::Space => " ".to_string(),
            })
        }

        write!(f, "{}", parts.join(""))
    }
}

impl<'sleigh> DisassembledTable<'sleigh> {
    /// How a token field or context value is displayed
    fn fmt_meaning(&self, meaning: Meaning, value: i64) -> String {
        match meaning {
            Meaning::NoAttach(value_fmt) => match value_fmt.base {
                PrintBase::Dec => format!("{}", value),
                PrintBase::Hex => fmt_hex(value),
            },
            Meaning::Varnode(attach_varnode_id) => {
                match self
                    .attach_varnode(attach_varnode_id)
                    .find_value(value as usize)
                {
                    Some(varnode_id) => self.varnode(varnode_id).name().to_string(),
                    None => format!("<UNDEFINED ATTACHED VARNODE {}>", value),
                }
            }
            Meaning::Literal(attach_literal_id) => {
                match self
                    .attach_literal(attach_literal_id)
                    .find_value(value as usize)
                {
                    Some(literal) => literal.to_string(),
                    None => format!("<UNDEFINED ATTACHED LITERAL {}>", value),
                }
            }
            Meaning::Number(print_base, attach_number_id) => {
                match self
                    .attach_number(attach_number_id)
                    .find_value(value as usize)
                {
                    Some(number) => match (print_base, number) {
                        (PrintBase::Dec, Number::Positive(x)) => format!("{}", x),
                        (PrintBase::Dec, Number::Negative(x)) => format!("-{}", x),
                        (PrintBase::Hex, Number::Positive(x)) => format!("{:#x}", x),
                        (PrintBase::Hex, Number::Negative(x)) => format!("-{:#x}", x),
                    },
                    None => format!("<UNDEFINED ATTACHED NUMBER {}>", value),
                }
            }
        }
    }
}

/// The bytes a constructor's `delayslot` needs, 0 without one
pub fn delay_slot_len(constructor: &Constructor) -> u64 {
    constructor
        .execution
        .iter()
        .flat_map(|execution| execution.blocks())
        .flat_map(|block| block.statements.iter())
        .filter_map(|statement| match statement {
            Statement::Delayslot(len) => Some(*len),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// Warn about the parts of a spec that never match: sub-patterns spanning several blocks
/// with `;`
pub fn warn_unsupported(sleigh: &Sleigh) {
    fn visit(blocks: &[Block]) {
        for verification in blocks.iter().flat_map(Block::verifications) {
            if let Verification::SubPattern { location, pattern } = verification {
                if pattern.blocks().len() > 1 {
                    let start = match location {
                        Span::File(span) => &span.start,
                        Span::Macro(span) => &span.start.expansion.start,
                    };
                    log::warn!(
                        "{}:{}: sub-patterns spanning several blocks with `;` are not supported",
                        start.file.display(),
                        start.line + 1
                    );
                }
                visit(pattern.blocks());
            }
        }
    }
    for table in sleigh.tables() {
        for constructor in table.constructors() {
            visit(constructor.pattern.blocks());
        }
    }
}

fn compare(op: CmpOp, l: i64, r: i64) -> bool {
    match op {
        CmpOp::Eq => l == r,
        CmpOp::Ne => l != r,
        CmpOp::Lt => l < r,
        CmpOp::Gt => l > r,
        CmpOp::Le => l <= r,
        CmpOp::Ge => l >= r,
    }
}

fn fmt_hex(value: i64) -> String {
    if value < 0 {
        format!("-{:#x}", value.unsigned_abs())
    } else {
        format!("{:#x}", value)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::path::Path;

    fn load(slaspec_path: impl AsRef<Path>) -> Sleigh {
        let _ = env_logger::try_init();
        log::info!("Loading slaspec: {:?}", slaspec_path.as_ref());
        sleigh_rs::file_to_sleigh(slaspec_path.as_ref())
            .unwrap_or_else(|_| panic!("Could not load slaspec: {:?}", slaspec_path.as_ref()))
    }

    fn run_tests(slaspec_path: impl AsRef<Path>, tests: &[(&str, Vec<u8>)]) {
        run_tests_in_context(slaspec_path, &[], tests)
    }

    /// Like `run_tests`, starting from the context `values`
    fn run_tests_in_context(
        slaspec_path: impl AsRef<Path>,
        values: &[(&str, i64)],
        tests: &[(&str, Vec<u8>)],
    ) {
        let slaspec = load(slaspec_path);
        let disasm = Disassembler::new(&slaspec);
        let context = Context::from_values(&slaspec, values).unwrap();
        for (expected_output, input_code) in tests.iter() {
            log::info!(
                "Disassembling {:02x?} expecting {:?}",
                input_code,
                expected_output
            );
            let instruction = disasm
                .disassemble(0x00000000, &context, input_code)
                .expect("Could not disassemble code");
            let actual_output = format!("{}", instruction);
            log::info!("Produced disassembly: {:?}", actual_output);
            assert_eq!(&actual_output, expected_output);
        }
    }

    #[test]
    fn test_risc_disassemble() {
        #[rustfmt::skip]
        run_tests("examples/risc.slaspec", &[
            ("xor r2, r15, 0xffff", vec![0x2f, 0x90, 0xff, 0xff]),
            ("add r4, r5, 0x1234", vec![0x02, 0xa0, 0x12, 0x34]),
            ("add r4, r5, 0x1234", vec![0x02, 0xa0, 0x12, 0x34]),
            ("xor r4, r5, 0x23450000", vec![0x2a, 0xa2, 0x23, 0x45]),
            ("and r1, r2, r3", vec![0xf9, 0x09, 0x80, 0x03]),
        ]);
    }

    #[test]
    fn test_vliw_disassemble() {
        #[rustfmt::skip]
        run_tests("examples/vliw.slaspec", &[
            ("{ unk.0x0 r1, r2, 0x1234 ; unk.0xa r5, r1, 0x1234 ; unk.0xb r10, r11, 0 }", vec![0x50, 0x04, 0x4a, 0x28, 0x56, 0xa5, 0x92, 0x34]),
            ("{ unk.0x0 r1, r2, -0x789abcdf }", vec![0xc0, 0x04, 0x40, 0x00, 0x87, 0x65, 0x43, 0x21])
        ]);
    }

    #[test]
    fn test_cisc_disassemble() {
        #[rustfmt::skip]
        run_tests("examples/cisc.slaspec", &[
            ("nop", vec![0x00]),
            ("mov r1, #0x10", vec![0x01, 0xc8, 0x00, 0x00, 0x00, 0x10]),
            ("add r1, r2", vec![0x02, 0x0a]),
            ("sub r0, [r3]", vec![0x03, 0x43]),
            ("xor r2, [sp+-0x4]", vec![0x06, 0x97, 0xfc]),
            ("mov [r4+0x8], r1", vec![0x08, 0x8c, 0x08]),
            ("jmp 0x1234", vec![0x10, 0x00, 0x00, 0x12, 0x34]),
            ("jz 0x4", vec![0x11, 0x02]),
            ("jnz 0x0", vec![0x12, 0xfe]),
            ("out r3", vec![0x20, 0x18]),
            ("in r1", vec![0x21, 0x08]),
        ]);
    }

    #[test]
    fn test_belt_disassemble() {
        #[rustfmt::skip]
        run_tests("examples/belt.slaspec", &[
            ("con -0x1", vec![0x04, 0x03, 0xff, 0xff]),
            ("conw 0xedb88320", vec![0x08, 0x00, 0x00, 0x00, 0xed, 0xb8, 0x83, 0x20]),
            ("add b0, b1", vec![0x0c, 0x04, 0x00, 0x00]),
            ("conform b3, b0, b7, b2, b1", vec![0x54, 0xc1, 0xc8, 0x45]),
            ("br b1, 0x14", vec![0x5c, 0x40, 0x00, 0x04]),
            ("jmp 0x0", vec![0x64, 0x03, 0xff, 0xff]),
        ]);
    }

    #[test]
    fn test_layout_disassemble() {
        #[rustfmt::skip]
        run_tests("examples/layout.slaspec", &[
            ("ldi 0x5, r2", vec![0x01, 0x02, 0x05]),
            ("mov r1, r2", vec![0x02, 0x01, 0x02]),
            ("mov r1, #0x7", vec![0x02, 0x01, 0x80, 0x07]),
            ("add r2, r3", vec![0x03, 0x02, 0x03]),
            ("add #0x7, r3", vec![0x03, 0x80, 0x07, 0x03]),
        ]);
    }

    #[test]
    fn test_8051_disassemble() {
        #[rustfmt::skip]
        run_tests("examples/8051.slaspec", &[
            ("mov A, R2", vec![0xea]),
            ("mov ACC, R2", vec![0x8a, 0xe0]),
            ("mov A, @R1", vec![0xe7]),
            ("mov 0x30, 0x4", vec![0x85, 0x04, 0x30]),
            ("mov SP, #0x2f", vec![0x75, 0x81, 0x2f]),
            ("mov DPTR, #0x1234", vec![0x90, 0x12, 0x34]),
            ("xrl 0x90, #0xed", vec![0x63, 0x90, 0xed]),
            ("ajmp 0x734", vec![0xe1, 0x34]),
            ("lcall 0x1234", vec![0x12, 0x12, 0x34]),
            ("djnz R3, 0x0", vec![0xdb, 0xfe]),
            ("cjne A, #0x5, 0x13", vec![0xb4, 0x05, 0x10]),
        ]);
    }

    #[test]
    fn test_context_disassemble() {
        let path = "examples/context.slaspec";
        #[rustfmt::skip]
        run_tests(path, &[
            ("add r1, r2, 0x5", vec![0x05, 0x00, 0x12, 0x01]),
            ("mode1", vec![0x00, 0x00, 0x00, 0x02]),
            ("pfx 0x2", vec![0x02, 0x00, 0x00, 0x04]),
            ("shl r1, r2, 0x0", vec![0x00, 0x00, 0x12, 0x05]),
            ("mov r1, r2", vec![0x00, 0x00, 0x12, 0x06]),
            ("mov r1, #0x7", vec![0x07, 0x00, 0x18, 0x06]),
            ("lo", vec![0x00, 0x00, 0x00, 0x07]),
        ]);
        #[rustfmt::skip]
        run_tests_in_context(path, &[("mode", 1), ("shift", 2)], &[
            ("sub r1, r2, 0x5", vec![0x05, 0x00, 0x12, 0x01]),
            ("shl r1, r2, 0x2", vec![0x00, 0x00, 0x12, 0x05]),
            ("hi", vec![0x00, 0x00, 0x00, 0x07]),
        ]);
        // mov's own assignment decides which SRC its subtable decodes
        #[rustfmt::skip]
        run_tests_in_context(path, &[("width", 1)], &[
            ("mov r1, r2", vec![0x00, 0x00, 0x12, 0x06]),
        ]);
    }

    #[test]
    fn test_subpattern_disassemble() {
        let path = "examples/subpattern.slaspec";
        #[rustfmt::skip]
        run_tests(path, &[
            ("nop", vec![0x00, 0x00]),
            ("nop", vec![0xf0, 0x00]),
            ("nop", vec![0xf1, 0x23]),
            ("mov r1, r2", vec![0x11, 0x20]),
            ("mov r1, r2", vec![0x21, 0x2f]),
            ("inc r3", vec![0x33, 0x00]),
            ("lim r1, 0x80", vec![0x61, 0x80]),
            ("lim r1, 0xf", vec![0x71, 0x0f]),
            ("ld r1, r2", vec![0x51, 0x20]),
            ("ld r1, #0x21", vec![0x51, 0x21]),
            ("pair r1", vec![0x91, 0x00, 0x01, 0x02]),
            ("pair r1", vec![0x91, 0x00, 0x03, 0xff]),
            ("opt", vec![0x80, 0x00]),
        ]);
        #[rustfmt::skip]
        run_tests_in_context(path, &[("level", 1)], &[
            ("inc r3", vec![0x33, 0x00]),
        ]);
        #[rustfmt::skip]
        run_tests_in_context(path, &[("level", 2)], &[
            ("inc r3", vec![0x43, 0x00]),
        ]);

        // Bytes that match no branch of the sub-patterns
        let sleigh = load(path);
        let disasm = Disassembler::new(&sleigh);
        let level3 = Context::from_values(&sleigh, &[("level", 3)]).unwrap();
        #[rustfmt::skip]
        let rejected = [
            (Context::new(&sleigh), vec![0x00, 0x01]),
            (Context::new(&sleigh), vec![0x11, 0x2f]),
            (Context::new(&sleigh), vec![0x21, 0x20]),
            (Context::new(&sleigh), vec![0x43, 0x00]),
            (Context::new(&sleigh), vec![0x61, 0x7f]),
            (Context::new(&sleigh), vec![0x71, 0x10]),
            (Context::new(&sleigh), vec![0x91, 0x00, 0x01, 0x03]),
            // OPT has to decode whichever branch matches
            (Context::new(&sleigh), vec![0x80, 0x0f]),
            (Context::new(&sleigh), vec![0x80, 0x01]),
            (level3, vec![0x33, 0x00]),
        ];
        for (context, bytes) in rejected {
            assert!(
                disasm.disassemble(0, &context, &bytes).is_err(),
                "{:02x?}",
                bytes
            );
        }
    }

    #[test]
    fn test_solver_undefined_expressions() {
        let path = "examples/solver.slaspec";
        #[rustfmt::skip]
        run_tests(path, &[
            ("div r1, 0x80", vec![0xdb, 0x00, 0x00, 0x02]),
            ("divc r1", vec![0xdb, 0x20, 0x00, 0x01]),
            // Shifts are modulo 64 and the product wraps
            ("bit r1, 0x8", vec![0xdb, 0x30, 0x00, 0x41]),
            ("bit r1, 0x0", vec![0xdb, 0x30, 0x00, 0x3f]),
        ]);
        // Dividing by zero, in an action and in a check, fails instead of panicking
        let sleigh = load(path);
        let disasm = Disassembler::new(&sleigh);
        for bytes in [[0xdb, 0x00, 0x00, 0x00], [0xdb, 0x10, 0x00, 0x04]] {
            assert!(
                disasm
                    .disassemble(0, &Context::new(&sleigh), &bytes)
                    .is_err(),
                "{:02x?}",
                bytes
            );
        }
    }

    #[test]
    fn test_context_commits() {
        let sleigh = load("examples/context.slaspec");
        let disasm = Disassembler::new(&sleigh);
        let id = |name| Context::id(&sleigh, name).unwrap();
        let decode = |bytes: &[u8]| {
            disasm
                .disassemble(0x100, &Context::new(&sleigh), bytes)
                .unwrap()
        };

        #[rustfmt::skip]
        let tests = [
            (vec![0x05, 0x00, 0x12, 0x01], vec![]),
            (vec![0x00, 0x00, 0x00, 0x02], vec![ContextCommit { address: 0x104, context: id("mode"), value: 1 }]),
            (vec![0x02, 0x00, 0x00, 0x04], vec![ContextCommit { address: 0x104, context: id("shift"), value: 2 }]),
            (vec![0x40, 0x12, 0x00, 0x0a], vec![ContextCommit { address: 0x1240, context: id("mode"), value: 1 }]),
            // To the address a table exports, which may depend on inst_next
            (vec![0x40, 0x12, 0x00, 0x0b], vec![ContextCommit { address: 0x1240, context: id("mode"), value: 1 }]),
            (vec![0x10, 0x00, 0x00, 0x0c], vec![ContextCommit { address: 0x114, context: id("mode"), value: 1 }]),
            // An exported constant is a code address, an exported register is not
            (vec![0x00, 0x02, 0x08, 0x0d], vec![ContextCommit { address: 0x200, context: id("mode"), value: 1 }]),
            (vec![0x00, 0x00, 0x01, 0x0d], vec![]),
            // A local is taken when the globalset runs
            (vec![0x34, 0x12, 0x00, 0x0e], vec![ContextCommit { address: 0x1234, context: id("mode"), value: 1 }]),
        ];
        for (bytes, commits) in tests {
            assert_eq!(decode(&bytes).commits, commits, "{:02x?}", bytes);
        }

        // The instruction keeps the context its constructors left behind
        let mov = decode(&[0x07, 0x00, 0x18, 0x06]);
        assert_eq!(mov.context.get(&sleigh, id("width")), 1);
        assert_eq!(mov.context.get(&sleigh, id("mode")), 0);
    }

    /// `jm1` at the end commits mode=1 to its destination, which a linear sweep has passed
    #[test]
    fn test_context_flow() {
        let sleigh = load("examples/context.slaspec");
        let disasm = Disassembler::new(&sleigh);
        #[rustfmt::skip]
        let code = [
            0x0c, 0x01, 0x00, 0x09, // 0x100: jmp 0x10c
            0x05, 0x00, 0x12, 0x01, // 0x104: add/sub r1, r2, 0x5
            0x05, 0x00, 0x12, 0x01, // 0x108: add/sub r1, r2, 0x5
            0x04, 0x01, 0x00, 0x0a, // 0x10c: jm1 0x104
            0x05, 0x00, 0x12, 0x01, // 0x110: not reached
        ];
        let context = Context::new(&sleigh);
        assert_eq!(
            disasm
                .disassemble(0x104, &context, &code[4..])
                .unwrap()
                .to_string(),
            "add r1, r2, 0x5"
        );

        let instructions = disasm
            .disassemble_flow(0x100, &code, 0x100, &context)
            .unwrap();
        let listing = instructions
            .iter()
            .map(|(address, instruction)| (*address, instruction.to_string()))
            .collect::<Vec<_>>();
        #[rustfmt::skip]
        assert_eq!(listing, vec![
            (0x100, "jmp 0x10c".to_string()),
            (0x104, "sub r1, r2, 0x5".to_string()),
            (0x108, "sub r1, r2, 0x5".to_string()),
            (0x10c, "jm1 0x104".to_string()),
        ]);

        // jt1 commits to its destination through the Dest table
        let mut code = code;
        code[15] = 0x0b;
        let instructions = disasm
            .disassemble_flow(0x100, &code, 0x100, &context)
            .unwrap();
        assert_eq!(instructions[&0x10c].to_string(), "jt1 0x104");
        assert_eq!(instructions[&0x104].to_string(), "sub r1, r2, 0x5");
    }

    /// A little endian instruction of the flow example
    fn flow_instruction(op: u8, imm16: u16) -> [u8; 4] {
        (((op as u32) << 24) | imm16 as u32).to_le_bytes()
    }

    #[test]
    fn test_flows() {
        let sleigh = load("examples/flow.slaspec");
        let disasm = Disassembler::new(&sleigh);
        let decode = |op, imm16| {
            disasm
                .disassemble(0x100, &Context::new(&sleigh), &flow_instruction(op, imm16))
                .unwrap()
        };
        #[rustfmt::skip]
        let tests: &[(u8, &str, &[u64])] = &[
            (0, "nop", &[0x104]),
            (1, "jmp 0x200", &[0x200]),
            (2, "call 0x200", &[0x104, 0x200]),
            (3, "ret", &[]),
            (4, "jr", &[]),
            (5, "callr", &[0x104]),
            // Other spaces are not code
            (6, "jio 0x200", &[]),
            (7, "jz 0x200", &[0x104, 0x200]),
            (8, "retz", &[0x104]),
            (9, "callz 0x200", &[0x104, 0x200]),
            (10, "loop", &[]),
            (11, "skipz", &[0x104]),
            // Through memory at a dynamic address, not to where the address is kept
            (13, "jd [r0]", &[]),
            (14, "calld [r0]", &[0x104]),
            (15, "jv [vec]", &[]),
        ];
        for (op, text, flows) in tests {
            let instruction = decode(*op, 0x200);
            assert_eq!(instruction.to_string(), *text);
            assert_eq!(instruction.flows(), *flows, "{}", text);
        }

        // Without p-code it falls through
        let bad = disasm
            .disassemble(0x100, &Context::new(&sleigh), &0x0c010000u32.to_le_bytes())
            .unwrap();
        assert_eq!(bad.to_string(), "bad");
        assert!(bad.pcode.is_err());
        assert_eq!(bad.flows(), vec![0x104]);
    }

    #[test]
    fn test_disassemble_flow() {
        let sleigh = load("examples/flow.slaspec");
        let disasm = Disassembler::new(&sleigh);
        let code = [
            flow_instruction(2, 0x10c), // 0x100: call 0x10c
            flow_instruction(1, 0x999), // 0x104: jmp 0x999, outside the code
            flow_instruction(0, 0),     // 0x108: nop, not reached
            flow_instruction(7, 0x100), // 0x10c: jz 0x100, back to the entry
        ]
        .concat();
        let instructions = disasm
            .disassemble_flow(0x100, &code, 0x100, &Context::new(&sleigh))
            .unwrap();
        assert_eq!(
            instructions.keys().copied().collect::<Vec<_>>(),
            vec![0x100, 0x104, 0x10c]
        );
    }

    /// Delay slots are decoded with their branch, which falls through past them
    #[test]
    fn test_disassemble_flow_delay_slots() {
        let sleigh = load("examples/delay.slaspec");
        let disasm = Disassembler::new(&sleigh);
        let code = [
            0x08000004u32, // 0x00: j 0x10
            0x04010000,    // 0x04: _slot r1, 0x1
            0x00000000,    // 0x08: nop, not reached
            0x10000001,    // 0x0c: beq r0, r0, 0x14
            0x18e00000,    // 0x10: jr ra
            0x00000000,    // 0x14: _nop
        ]
        .map(u32::to_be_bytes)
        .concat();
        let instructions = disasm
            .disassemble_flow(0x0, &code, 0x0, &Context::new(&sleigh))
            .unwrap();
        assert_eq!(
            instructions.keys().copied().collect::<Vec<_>>(),
            vec![0x0, 0x10]
        );
        assert_eq!(
            instructions[&0x0].delay_slots[0].to_string(),
            "slot r1, 0x1"
        );

        let beq = disasm
            .disassemble(0xc, &Context::new(&sleigh), &code[0xc..])
            .unwrap();
        assert_eq!(beq.to_string(), "beq r0, r0, 0x14");
        // Taken or not, it goes past its delay slot
        assert_eq!(beq.flows(), vec![0x14]);
    }
}
