use std::collections::HashMap;

use anyhow::{bail, Result};

use sleigh_rs::disassembly::{Assertation, Expr, ExprElement, Op, OpUnary, ReadScope, VariableId};
use sleigh_rs::display::DisplayElement;
use sleigh_rs::pattern::{BitConstraint, Verification};
use sleigh_rs::table::{Constructor, Table};
use sleigh_rs::token::TokenFieldAttach;
use sleigh_rs::{Endian, Sleigh, TableId, TokenFieldId};

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

pub struct Context;

impl<'sleigh> Disassembler<'sleigh> {
    pub fn new(sleigh: &'sleigh Sleigh) -> Self {
        Disassembler { sleigh }
    }

    pub fn disassemble(
        &'sleigh self,
        inst_start: u64,
        context: Context,
        bytes: &[u8],
    ) -> Result<DisassembledInstruction<'sleigh>> {
        let table = self.disassemble_table(
            inst_start,
            self.table(self.instruction_table()),
            context,
            bytes,
        )?;
        Ok(DisassembledInstruction { table })
    }

    pub fn disassemble_table(
        &'sleigh self,
        inst_start: u64,
        table: &'sleigh Table,
        context: Context,
        bytes: &[u8],
    ) -> Result<DisassembledTable<'sleigh>> {
        DisassembledTable::disassemble(self, inst_start, table, bytes)
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

        let token_field_value = if token_field.raw_value_is_signed()
            && (token_field_raw & (1 << (token_field.bits.len().get() - 1)) != 0)
        {
            (token_field_raw as i64) - (1i64 << token_field.bits.len().get())
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
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        (self.inst_next - self.inst_start) as usize
    }

    pub fn disassemble(
        disassembler: &'sleigh Disassembler<'sleigh>,
        inst_start: u64,
        table: &'sleigh Table,
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
                bail!("too few bytes to match constructor")
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
                inst_start: inst_start,
                inst_next: inst_start,
                token_fields: HashMap::new(),
                tables: HashMap::new(),
                variables: HashMap::new(),
            };

            for block in constructor.pattern.blocks() {
                let block_len = block.len().single_len().unwrap_or(block.len().min());
                if bytes.len() < block_len as usize {
                    bail!("too few bytes to match block")
                }
                disasm_table.inst_next += block_len;

                for produced_token_field in block.token_fields() {
                    let token_field_value =
                        disassembler.extract_token_field(produced_token_field.field, bytes);
                    disasm_table
                        .token_fields
                        .insert(produced_token_field.field, token_field_value);
                }

                for produced_table in block.tables() {
                    let subtable = disassembler.table(produced_table.table);
                    if let Ok(subtable_value) =
                        DisassembledTable::disassemble(disassembler, inst_start, subtable, bytes)
                    {
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

                for assertion in block.pre_disassembler() {
                    match assertion {
                        Assertation::GlobalSet(global_set) => todo!(),
                        Assertation::Assignment(assignment) => {
                            let value = disasm_table.evaluate_expr(&assignment.right, bytes);
                            match assignment.left {
                                sleigh_rs::disassembly::WriteScope::Context(context_id) => todo!(),
                                sleigh_rs::disassembly::WriteScope::Local(variable_id) => {
                                    disasm_table.variables.insert(variable_id, value);
                                }
                            }
                        }
                    }
                }

                for verification in block.verifications() {
                    match verification {
                        Verification::ContextCheck { context, op, value } => todo!(),
                        Verification::TableBuild {
                            produced_table,
                            verification,
                        } => {}
                        Verification::TokenFieldCheck { field, op, value } => {
                            let field_value = disassembler.extract_token_field(*field, bytes);
                            let check_value = disasm_table.evaluate_expr(value.expr(), bytes);
                            let check_ok = match op {
                                sleigh_rs::pattern::CmpOp::Eq => field_value == check_value,
                                sleigh_rs::pattern::CmpOp::Ne => field_value != check_value,
                                sleigh_rs::pattern::CmpOp::Lt => field_value < check_value,
                                sleigh_rs::pattern::CmpOp::Gt => field_value > check_value,
                                sleigh_rs::pattern::CmpOp::Le => field_value <= check_value,
                                sleigh_rs::pattern::CmpOp::Ge => field_value >= check_value,
                            };
                            if !check_ok {
                                log::trace!("{}: failed verification {}={} {:?} {}, continuing to next matcher", table.name(), disassembler.token_field(*field).name(), field_value, op, check_value);
                                continue 'match_loop;
                            }
                        }
                        Verification::SubPattern { location, pattern } => todo!(),
                    }
                }
                bytes = &bytes[block_len as usize..];
            }
            return Ok(disasm_table);
        }
        bail!("{}: Failed to disassemble table", table.name());
    }

    pub fn evaluate_expr(&self, expr: &Expr, bytes: &[u8]) -> i64 {
        match expr {
            Expr::Value(expr_element) => match expr_element {
                ExprElement::Value { value, location } => match *value {
                    ReadScope::Integer(number) => match number {
                        sleigh_rs::Number::Positive(x) => x as i64,
                        sleigh_rs::Number::Negative(x) => -(x as i64),
                    },
                    ReadScope::Context(context_id) => todo!(),
                    ReadScope::TokenField(token_field_id) => {
                        self.disassembler.extract_token_field(token_field_id, bytes)
                    }
                    ReadScope::InstStart(_inst_start) => self.inst_start as i64,
                    ReadScope::InstNext(_inst_next) => self.inst_next as i64,
                    ReadScope::Local(variable_id) => {
                        *self.variables.get(&variable_id).unwrap()
                    }
                },
                ExprElement::Op(span, op_unary, expr) => {
                    let expr = self.evaluate_expr(expr, bytes);
                    match op_unary {
                        OpUnary::Negation => !expr,
                        OpUnary::Negative => -expr,
                    }
                }
            },
            Expr::Op(_span, op, expr, expr1) => {
                let l = self.evaluate_expr(expr, bytes);
                let r = self.evaluate_expr(expr1, bytes);
                match op {
                    Op::Add => l + r,
                    Op::Sub => l - r,
                    Op::Mul => l * r,
                    Op::Div => l / r,
                    Op::And => l & r,
                    Op::Or => l | r,
                    Op::Xor => l ^ r,
                    Op::Asr => l >> r,
                    Op::Lsl => l << r,
                }
            }
        }
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
                    self.disassembler.context(*context_id).name().to_string()
                }
                DisplayElement::TokenField(token_field_id) => {
                    let token_field = self.disassembler.token_field(*token_field_id);
                    if let Some(token_field_value) = self.token_fields.get(token_field_id).cloned() {
                        match token_field.attach {
                            TokenFieldAttach::NoAttach(value_fmt) => match value_fmt.base {
                                sleigh_rs::PrintBase::Dec => format!("{}", token_field_value),
                                sleigh_rs::PrintBase::Hex => format!("{:#x}", token_field_value),
                            },
                            TokenFieldAttach::Varnode(attach_varnode_id) => {
                                let attach_varnode =
                                    self.disassembler.attach_varnode(attach_varnode_id);
                                if let Some(varnode_id) =
                                    attach_varnode.find_value(token_field_value as usize)
                                {
                                    let varnode = self.disassembler.varnode(varnode_id);
                                    varnode.name().to_string()
                                } else {
                                    format!("<UNDEFINED ATTACHED VARNODE {}>", token_field_value)
                                }
                            }
                            TokenFieldAttach::Literal(attach_literal_id) => {
                                let attach_literal =
                                    self.disassembler.attach_literal(attach_literal_id);
                                if let Some(literal_str) =
                                    attach_literal.find_value(token_field_value as usize)
                                {
                                    literal_str.to_string()
                                } else {
                                    format!("<UNDEFINED ATTACHED LITERAL {}>", token_field_value)
                                }
                            }
                            TokenFieldAttach::Number(print_base, attach_number_id) => {
                                let attach_number =
                                    self.disassembler.attach_number(attach_number_id);
                                if let Some(number) =
                                    attach_number.find_value(token_field_value as usize)
                                {
                                    match (print_base, number) {
                                        (
                                            sleigh_rs::PrintBase::Dec,
                                            sleigh_rs::Number::Positive(x),
                                        ) => format!("{}", x),
                                        (
                                            sleigh_rs::PrintBase::Dec,
                                            sleigh_rs::Number::Negative(x),
                                        ) => format!("-{}", x),
                                        (
                                            sleigh_rs::PrintBase::Hex,
                                            sleigh_rs::Number::Positive(x),
                                        ) => format!("{:#x}", x),
                                        (
                                            sleigh_rs::PrintBase::Hex,
                                            sleigh_rs::Number::Negative(x),
                                        ) => format!("-{:#x}", x),
                                    }
                                } else {
                                    format!("<UNDEFINED ATTACHED NUMBER {}>", token_field_value)
                                }
                            }
                        }
                    } else {
                        format!(
                            "<UNDEFINED FIELD {}/{}>",
                            token_field.name(),
                            token_field_id.0
                        )
                    }
                }
                DisplayElement::InstStart(_inst_start) => format!("{:x}", self.inst_start),
                DisplayElement::InstNext(_inst_next) => format!("{:x}", self.inst_next),
                DisplayElement::Table(table_id) => {
                    format!("{}", self.tables.get(table_id).unwrap())
                }
                DisplayElement::Disassembly(variable_id) => {
                    if let Some(variable) = self.variables.get(variable_id) {
                        format!("{:#x}", variable)
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

#[cfg(test)]
mod test {
    use super::*;
    use std::path::Path;

    fn run_tests(slaspec_path: impl AsRef<Path>, tests: &[(&str, Vec<u8>)]) {
        let _ = env_logger::try_init();
        log::info!("Loading slaspec: {:?}", slaspec_path.as_ref());
        let slaspec = sleigh_rs::file_to_sleigh(slaspec_path.as_ref()).expect(&format!(
            "Could not load slaspec: {:?}",
            slaspec_path.as_ref()
        ));
        let disasm = Disassembler::new(&slaspec);
        for (expected_output, input_code) in tests.iter() {
            log::info!(
                "Disassembling {:02x?} expecting {:?}",
                input_code,
                expected_output
            );
            let instruction = disasm
                .disassemble(0x00000000, Context, &input_code)
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
            ("{ unk.0x0 r1, r2, 0xffffffff87654321 }", vec![0xc0, 0x04, 0x40, 0x00, 0x87, 0x65, 0x43, 0x21])
        ]);
    }
}
