use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use sleigh_rs::disassembly::{
    Assertation, Expr, ExprElement, Op, OpUnary, ReadScope, VariableId, WriteScope,
};
use sleigh_rs::display::DisplayElement;
use sleigh_rs::meaning::AttachVarnode;
use sleigh_rs::pattern::{CmpOp, Verification};
use sleigh_rs::table::{Constructor, Table};
use sleigh_rs::{token::TokenFieldAttach, Endian, Number, Sleigh, TokenFieldId, TokenId};
use z3::ast::{Ast, Bool, BV};

use crate::disassembler::{Context, Disassembler};

/// Every way a parser matched a prefix of the input, each with the remaining input.
/// An empty list means no match.
pub type Parses<'a, T> = Vec<(&'a str, T)>;

/// One of several distinct encodings of an ambiguous instruction
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub bytes: Vec<u8>,
    pub disassembly: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsmError {
    NoMatch,
    Ambiguous(Vec<Candidate>),
}

impl std::fmt::Display for AsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsmError::NoMatch => write!(f, "no constructor matches the input"),
            AsmError::Ambiguous(candidates) => {
                write!(f, "ambiguous, {} encodings:", candidates.len())?;
                for candidate in candidates {
                    write!(f, "\n  {:02x?} {}", candidate.bytes, candidate.disassembly)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for AsmError {}

fn parse_literal<'a>(lit: &str, s: &'a str) -> Option<&'a str> {
    s.strip_prefix(lit)
}

fn parse_space1(s: &str) -> Option<&str> {
    let rest = s.trim_start_matches([' ', '\t']);
    (rest.len() < s.len()).then_some(rest)
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

#[derive(Debug, Clone)]
pub struct Constraints<'asm> {
    pub asm: &'asm InstructionAssembler,

    pub token_order: Vec<TokenId>,

    pub tokens: HashMap<TokenId, BV<'asm>>,
    pub fields: HashMap<TokenFieldId, BV<'asm>>,

    pub eqs: HashSet<Bool<'asm>>,
}

impl<'asm> Constraints<'asm> {
    pub fn new(asm: &'asm InstructionAssembler) -> Self {
        Self {
            asm,
            token_order: Vec::new(),
            tokens: HashMap::new(),
            fields: HashMap::new(),
            eqs: HashSet::new(),
        }
    }

    pub fn token(&mut self, token_id: TokenId) -> BV<'asm> {
        let token = self.asm.token(token_id);
        let bv = self.tokens.entry(token_id).or_insert_with(|| {
            self.token_order.push(token_id);
            BV::fresh_const(
                &self.asm.ctx,
                token.name(),
                8 * (token.len_bytes().get() as u32),
            )
        });
        bv.clone()
    }

    pub fn token_field(&mut self, token_field_id: TokenFieldId, sz: Option<u32>) -> BV<'asm> {
        let token_field = self.asm.sleigh.token_field(token_field_id);
        let field_bv = self
            .fields
            .entry(token_field_id)
            .or_insert_with(|| {
                let field_bv = BV::fresh_const(
                    &self.asm.ctx,
                    token_field.name(),
                    token_field.bits.len().get() as u32,
                );
                field_bv
            })
            .clone();
        let token_bv = self.token(token_field.token);
        self.eq(token_bv
            .extract(
                (token_field.bits.end().get() - 1) as u32,
                token_field.bits.start() as u32,
            )
            ._eq(&field_bv));

        #[allow(clippy::comparison_chain)]
        if let Some(sz) = sz {
            if field_bv.get_size() < sz {
                if token_field.raw_value_is_signed() {
                    field_bv.sign_ext(sz - field_bv.get_size())
                } else {
                    field_bv.zero_ext(sz - field_bv.get_size())
                }
            } else if field_bv.get_size() > sz {
                field_bv.extract(sz - 1, 0)
            } else {
                field_bv
            }
        } else {
            field_bv
        }
    }

    pub fn eq(&mut self, eq: Bool<'asm>) {
        self.eqs.insert(eq);
    }

    pub fn merge(&mut self, other: Constraints<'asm>) {
        for (field_id, field_bv) in other.fields.into_iter() {
            if let Some(self_field_bv) = self.fields.get(&field_id) {
                self.eqs.insert(field_bv._eq(self_field_bv));
            } else {
                self.fields.insert(field_id, field_bv);
            }
        }
        self.eqs.extend(other.eqs);
        self.tokens.extend(other.tokens);
    }

    pub fn solver(&self) -> z3::Solver<'asm> {
        let solver = z3::Solver::new(&self.asm.ctx);
        for eq in self.eqs.iter() {
            solver.assert(eq)
        }
        solver
    }

    pub fn check(&self) -> bool {
        match self.solver().check() {
            z3::SatResult::Unsat | z3::SatResult::Unknown => false,
            z3::SatResult::Sat => true,
        }
    }

    pub fn model(&self) -> Option<z3::Model<'asm>> {
        let solver = self.solver();
        match solver.check() {
            z3::SatResult::Unsat | z3::SatResult::Unknown => None,
            z3::SatResult::Sat => solver.get_model(),
        }
    }

    pub fn build_u64_const(&self, u: u64, sz: u32) -> BV<'asm> {
        BV::from_u64(&self.asm.ctx, u, sz)
    }

    pub fn build_i64_const(&self, i: i64, sz: u32) -> BV<'asm> {
        BV::from_i64(&self.asm.ctx, i, sz)
    }

    /// Whether some instruction bytes satisfy both `self` and `other`
    pub fn same_encoding(&self, other: &Constraints<'asm>) -> bool {
        if self.token_order.len() != other.token_order.len() {
            return false;
        }
        let solver = self.solver();
        for eq in other.eqs.iter() {
            solver.assert(eq);
        }
        for (self_id, other_id) in self.token_order.iter().zip(other.token_order.iter()) {
            let (self_token, other_token) = (self.asm.token(*self_id), self.asm.token(*other_id));
            if self_token.len_bytes() != other_token.len_bytes()
                || self_token.endian() != other_token.endian()
            {
                return false;
            }
            solver.assert(&self.tokens[self_id]._eq(&other.tokens[other_id]));
        }
        solver.check() == z3::SatResult::Sat
    }

    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        log::debug!("Generating instruction bytes");
        let model = self.model()?;
        let mut instruction_bytes = vec![];
        for token_id in self.token_order.iter() {
            let token = self.asm.token(*token_id);
            let token_bv = self.tokens.get(token_id)?;
            let token_value = model.eval(token_bv, true)?.as_u64()?;
            let token_length = token.len_bytes().get() as usize;

            log::debug!("{}: {:#010X}/{}", token.name(), token_value, token_length);

            instruction_bytes.extend(match token.endian() {
                Endian::Little => token_value.to_le_bytes()[..token_length].to_vec(),
                Endian::Big => token_value.to_be_bytes()[8 - token_length..].to_vec(),
            });
        }
        Some(instruction_bytes)
    }
}

#[derive(Debug, Clone)]
struct Variables<'asm> {
    constructor: &'asm Constructor,
    constraints: Constraints<'asm>,
    variables: HashMap<VariableId, BV<'asm>>,
}

impl<'asm> Deref for Variables<'asm> {
    type Target = Constraints<'asm>;

    fn deref(&self) -> &Self::Target {
        &self.constraints
    }
}

impl<'asm> DerefMut for Variables<'asm> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.constraints
    }
}

impl<'asm> Variables<'asm> {
    pub fn new(constraints: Constraints<'asm>, constructor: &'asm Constructor) -> Self {
        Self {
            constraints,
            constructor,
            variables: HashMap::new(),
        }
    }

    pub fn variable(&mut self, variable_id: VariableId) -> BV<'asm> {
        self.variables
            .entry(variable_id)
            .or_insert_with(|| {
                let variable = self.constructor.pattern.disassembly_var(variable_id);
                BV::fresh_const(&self.constraints.asm.ctx, variable.name(), 64)
            })
            .clone()
    }

    pub fn build_expr_bv(&mut self, expr: &Expr, sz: u32) -> BV<'asm> {
        let expr_bv = match expr {
            Expr::Value(expr_element) => match expr_element {
                ExprElement::Value { value, location: _ } => match value {
                    ReadScope::Integer(number) => match number {
                        Number::Positive(x) => self.build_u64_const(*x, sz),
                        Number::Negative(x) => self.build_i64_const(-(*x as i64), sz),
                    },
                    ReadScope::Context(_context_id) => todo!(),
                    ReadScope::TokenField(token_field_id) => {
                        self.token_field(*token_field_id, Some(sz))
                    }
                    ReadScope::InstStart(_inst_start) => {
                        self.build_u64_const(0x1234567890abcdef, sz)
                    }
                    ReadScope::InstNext(_inst_next) => todo!(),
                    ReadScope::Local(variable_id) => self.variable(*variable_id),
                },
                ExprElement::Op(_span, op_unary, expr) => {
                    let expr_bv = self.build_expr_bv(expr, 64);
                    match op_unary {
                        OpUnary::Negation => expr_bv.bvnot(),
                        OpUnary::Negative => expr_bv.bvneg(),
                    }
                }
            },
            Expr::Op(_span, op, expr, expr1) => {
                let expr_r = self.build_expr_bv(expr, sz);
                let expr_l = self.build_expr_bv(expr1, sz);
                match op {
                    Op::Add => expr_r + expr_l,
                    Op::Sub => expr_r - expr_l,
                    Op::Mul => expr_r * expr_l,
                    Op::Div => expr_r.bvudiv(&expr_l),
                    Op::And => expr_r & expr_l,
                    Op::Or => expr_r | expr_l,
                    Op::Xor => expr_r ^ expr_l,
                    Op::Asr => expr_r.bvashr(&expr_l),
                    Op::Lsl => expr_r << expr_l,
                }
            }
        };
        #[allow(clippy::comparison_chain)]
        if expr_bv.get_size() < sz {
            expr_bv.sign_ext(sz - expr_bv.get_size())
        } else if expr_bv.get_size() > sz {
            expr_bv.extract(sz - 1, 0)
        } else {
            expr_bv
        }
    }
}

#[derive(Debug)]
pub struct InstructionAssembler {
    sleigh: Sleigh,
    ctx: z3::Context,
}

impl Deref for InstructionAssembler {
    type Target = Sleigh;

    fn deref(&self) -> &Self::Target {
        &self.sleigh
    }
}

impl InstructionAssembler {
    pub fn new(sleigh: Sleigh) -> Self {
        Self {
            sleigh,
            ctx: z3::Context::new(&z3::Config::new()),
        }
    }

    /// Assemble one instruction that must consume all of `s` (up to trailing whitespace) and
    /// have exactly one encoding. Parses that can produce the same bytes count as one encoding.
    pub fn assemble_instruction<'asm>(&'asm self, s: &str) -> Result<Constraints<'asm>, AsmError> {
        let mut encodings: Vec<Constraints<'asm>> = vec![];
        for (rest, candidate) in self.assemble_candidates(s) {
            if !rest.trim_end().is_empty() {
                continue;
            }
            if !encodings
                .iter()
                .any(|encoding| encoding.same_encoding(&candidate))
            {
                encodings.push(candidate);
            }
        }
        match encodings.len() {
            0 => Err(AsmError::NoMatch),
            1 => Ok(encodings.pop().unwrap()),
            _ => Err(AsmError::Ambiguous(
                encodings
                    .iter()
                    .map(|encoding| self.candidate(encoding))
                    .collect(),
            )),
        }
    }

    fn candidate(&self, constraints: &Constraints) -> Candidate {
        let bytes = constraints.to_bytes().unwrap_or_default();
        let disassembly = Disassembler::new(&self.sleigh)
            .disassemble(0, Context, &bytes)
            .map(|instruction| instruction.to_string())
            .unwrap_or_else(|err| format!("<{}>", err));
        Candidate { bytes, disassembly }
    }

    /// Every way the instruction table matches a prefix of `s`
    pub fn assemble_candidates<'a, 'asm>(&'asm self, s: &'a str) -> Parses<'a, Constraints<'asm>> {
        let constraints = Constraints::new(self);
        self.assemble_table(self.table(self.instruction_table()), constraints, s)
    }

    pub fn assemble_table<'a, 'asm>(
        &'asm self,
        table: &'asm Table,
        constraints: Constraints<'asm>,
        s: &'a str,
    ) -> Parses<'a, Constraints<'asm>> {
        table
            .constructors()
            .iter()
            .flat_map(|constructor| self.assemble_constructor(constructor, constraints.clone(), s))
            .collect()
    }

    pub fn assemble_constructor<'a, 'asm>(
        &'asm self,
        constructor: &'asm Constructor,
        constraints: Constraints<'asm>,
        s: &'a str,
    ) -> Parses<'a, Constraints<'asm>> {
        let s = match constructor.display.mneumonic.as_ref() {
            Some(mneumonic) => match parse_literal(mneumonic, s) {
                Some(s) => {
                    log::trace!("MNEUMONIC: {}", mneumonic);
                    s
                }
                None => return vec![],
            },
            None => s,
        };

        let mut variables = Variables::new(constraints, constructor);

        for block in constructor.pattern.blocks() {
            for verification in block.verifications() {
                match verification {
                    Verification::ContextCheck {
                        context: _,
                        op: _,
                        value: _,
                    } => todo!(),
                    Verification::TableBuild {
                        produced_table: _,
                        verification: _,
                    } => {
                        continue;
                    }
                    Verification::TokenFieldCheck { field, op, value } => {
                        let field_bv = variables.token_field(*field, None);
                        let value_bv = variables.build_expr_bv(value.expr(), field_bv.get_size());
                        let assertion = match op {
                            CmpOp::Eq => field_bv._eq(&value_bv),
                            CmpOp::Ne => field_bv._eq(&value_bv).not(),
                            CmpOp::Lt => field_bv.bvult(&value_bv),
                            CmpOp::Gt => field_bv.bvugt(&value_bv),
                            CmpOp::Le => field_bv.bvule(&value_bv),
                            CmpOp::Ge => field_bv.bvuge(&value_bv),
                        };
                        variables.eq(assertion);
                    }
                    Verification::SubPattern {
                        location: _,
                        pattern: _,
                    } => todo!(),
                }
            }
            for assertion in block.pre_disassembler() {
                match assertion {
                    Assertation::GlobalSet(_global_set) => todo!(),
                    Assertation::Assignment(assignment) => {
                        let value = variables.build_expr_bv(&assignment.right, 64);
                        match assignment.left {
                            WriteScope::Context(_context_id) => todo!(),
                            WriteScope::Local(variable_id) => {
                                let var = variables.variable(variable_id);
                                variables.eq(var._eq(&value))
                            }
                        }
                    }
                }
            }
        }

        let mut states = vec![(s, variables)];
        for elem in constructor.display.elements() {
            states = states
                .into_iter()
                .flat_map(|(s, variables)| {
                    self.assemble_display_element(constructor, elem, variables, s)
                })
                .collect();
            if states.is_empty() {
                return vec![];
            }
        }

        states
            .into_iter()
            .filter_map(|(s, variables)| {
                if variables.check() {
                    Some((s, variables.constraints))
                } else {
                    log::trace!("Constraint check failed, Eqs: {:#?}", variables.eqs);
                    None
                }
            })
            .collect()
    }

    fn assemble_display_element<'a, 'asm>(
        &'asm self,
        constructor: &'asm Constructor,
        elem: &'asm DisplayElement,
        mut variables: Variables<'asm>,
        s: &'a str,
    ) -> Parses<'a, Variables<'asm>> {
        match elem {
            DisplayElement::Varnode(_varnode_id) => todo!(),
            DisplayElement::Context(_context_id) => todo!(),
            DisplayElement::TokenField(token_field_id) => {
                let token_field = self.token_field(*token_field_id);
                log::trace!("TOKEN_FIELD: {:?} {:?}", token_field.name(), s);
                let token_field_bv = variables.token_field(*token_field_id, None);
                let values = match token_field.attach {
                    TokenFieldAttach::NoAttach(value_fmt) => {
                        self.parse_value(value_fmt.signed, s).into_iter().collect()
                    }
                    TokenFieldAttach::Varnode(attach_varnode_id) => {
                        let attach_varnode = self.attach_varnode(attach_varnode_id);
                        self.parse_attach_varnode(attach_varnode, s)
                    }
                    TokenFieldAttach::Literal(_attach_literal_id) => todo!(),
                    TokenFieldAttach::Number(_print_base, _attach_number_id) => todo!(),
                };
                let size = token_field_bv.get_size();
                values
                    .into_iter()
                    .filter_map(|(s, value)| {
                        let in_range = if token_field.raw_value_is_signed() {
                            let high = value.checked_shr(size - 1).unwrap_or(value >> 63);
                            high == 0 || high == -1
                        } else {
                            (value as u64).checked_shr(size).unwrap_or(0) == 0
                        };
                        if !in_range {
                            log::trace!("Immidiate out of range {} {}", size, value);
                            return None;
                        }
                        let mut variables = variables.clone();
                        let const_bv = variables.build_u64_const(value as u64, size);
                        variables.eq(token_field_bv._eq(&const_bv));
                        Some((s, variables))
                    })
                    .collect()
            }
            DisplayElement::InstStart(_inst_start) => todo!(),
            DisplayElement::InstNext(_inst_next) => todo!(),
            DisplayElement::Table(table_id) => {
                let table = self.table(*table_id);
                log::trace!("TABLE: {:?}/{:?} {:?}", table.name(), table_id, s);
                self.assemble_table(table, variables.constraints.clone(), s)
                    .into_iter()
                    .map(|(s, table_constraints)| {
                        let mut variables = variables.clone();
                        variables.merge(table_constraints);
                        (s, variables)
                    })
                    .collect()
            }
            DisplayElement::Disassembly(variable_id) => {
                let variable = constructor.pattern.disassembly_var(*variable_id);
                log::trace!("DISASSEMBLY: {:?} {:?}", variable.name(), s);
                let Some((s, value)) = self.parse_value(true, s) else {
                    return vec![];
                };
                let var = variables.variable(*variable_id);
                let const_bv = variables.build_u64_const(value as u64, var.get_size());
                variables.eq(var._eq(&const_bv));
                vec![(s, variables)]
            }
            DisplayElement::Literal(lit) => {
                log::trace!("LITERAL: {:?} {:?}", lit, s);
                parse_literal(lit, s)
                    .map(|s| (s, variables))
                    .into_iter()
                    .collect()
            }
            DisplayElement::Space => {
                log::trace!("SPACE: {:?}", s);
                parse_space1(s)
                    .map(|s| (s, variables))
                    .into_iter()
                    .collect()
            }
        }
    }

    pub fn parse_value<'a>(&self, signed: bool, s: &'a str) -> Option<(&'a str, i64)> {
        //TODO: labels?
        let (s, sign) = match s.strip_prefix('-') {
            Some(s) if signed => (s, true),
            _ => (s, false),
        };
        let (s, value) = parse_hex(s).or_else(|| parse_dec(s))?;
        let value = if sign { -(value as i64) } else { value as i64 };
        Some((s, value))
    }

    /// Every register name in the attach list that prefixes `s`, e.g. both `r1` and `r10`,
    /// longest match first
    pub fn parse_attach_varnode<'a>(
        &self,
        attach_varnode: &AttachVarnode,
        s: &'a str,
    ) -> Parses<'a, i64> {
        let mut parses = attach_varnode
            .0
            .iter()
            .filter_map(|(value, id)| {
                let s = s.strip_prefix(self.varnode(*id).name())?;
                Some((s, *value as i64))
            })
            .collect::<Parses<'a, i64>>();
        parses.sort_by_key(|(rest, _)| rest.len());
        parses
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
        let assembler = InstructionAssembler::new(slaspec);

        for (input, expected_bytes) in tests.iter() {
            log::info!("Assembling {:?} expecting {:02x?}", input, expected_bytes);

            let constraints = assembler
                .assemble_instruction(input)
                .unwrap_or_else(|err| panic!("{:?}: {}", input, err));

            let bytes = constraints
                .to_bytes()
                .expect("Constraints failed to produce bytes");
            log::info!("Produced bytes: {:02x?}", bytes);

            assert_eq!(bytes, *expected_bytes);
        }
    }

    #[test]
    pub fn test_vliw_assemble() {
        #[rustfmt::skip]
        run_tests("examples/vliw.slaspec", &[
            ("{ unk.0x0 r1, r2, 0x1234 ; unk.0xa r5, r1, 0x1234 ; unk.0xb r10, r11, 0 }", vec![0x50, 0x04, 0x4a, 0x28, 0x56, 0xa5, 0x92, 0x34]),
            ("{ unk.0x0 r1, r2, 0xffffffff87654321 }", vec![0xc0, 0x04, 0x40, 0x00, 0x87, 0x65, 0x43, 0x21]),
        ]);
    }

    #[test]
    pub fn test_risc_assemble() {
        #[rustfmt::skip]
        run_tests("examples/risc.slaspec", &[
            ("xor r2, r15, 0xffff", vec![0x2f, 0x90, 0xff, 0xff]),
            ("xor r4, r5, 0x23450000", vec![0x2a, 0xa2, 0x23, 0x45]),
            ("and r1, r2, r3", vec![0xf9, 0x09, 0x80, 0x03]),
        ]);
    }

    fn load(slaspec_path: impl AsRef<Path>) -> InstructionAssembler {
        let _ = env_logger::try_init();
        let sleigh = sleigh_rs::file_to_sleigh(slaspec_path.as_ref()).unwrap_or_else(|_| {
            panic!("Could not load slaspec: {:?}", slaspec_path.as_ref());
        });
        InstructionAssembler::new(sleigh)
    }

    fn assemble(assembler: &InstructionAssembler, input: &str) -> Result<Vec<u8>, AsmError> {
        let constraints = assembler.assemble_instruction(input)?;
        Ok(constraints
            .to_bytes()
            .expect("Constraints failed to produce bytes"))
    }

    fn disassemble(assembler: &InstructionAssembler, bytes: &[u8]) -> Option<String> {
        let disassembler = Disassembler::new(assembler);
        let instruction = disassembler.disassemble(0, Context, bytes).ok()?;
        Some(instruction.to_string())
    }

    fn assert_encodes(assembler: &InstructionAssembler, tests: &[(&str, Vec<u8>)]) {
        let mut failures = vec![];
        for (input, expected_bytes) in tests.iter() {
            match assemble(assembler, input) {
                Ok(bytes) if bytes == *expected_bytes => {}
                result => failures.push(format!(
                    "{:?}: expected {:02x?}, got {:02x?}",
                    input, expected_bytes, result
                )),
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    fn assert_rejects(assembler: &InstructionAssembler, inputs: &[&str]) {
        let mut failures = vec![];
        for input in inputs.iter() {
            match assemble(assembler, input) {
                Err(AsmError::NoMatch) => {}
                result => failures.push(format!(
                    "{:?}: expected rejection, got {:02x?}",
                    input, result
                )),
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    fn assert_ambiguous(assembler: &InstructionAssembler, tests: &[(&str, usize)]) {
        let mut failures = vec![];
        for (input, expected_count) in tests.iter() {
            match assemble(assembler, input) {
                Err(AsmError::Ambiguous(candidates)) if candidates.len() == *expected_count => {}
                result => failures.push(format!(
                    "{:?}: expected {} candidates, got {:02x?}",
                    input, expected_count, result
                )),
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    fn assert_roundtrips(assembler: &InstructionAssembler, len: usize, count: usize) {
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut failures = vec![];
        let mut checked = 0;
        for _ in 0..count {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let bytes = &seed.to_be_bytes()[..len];
            let Some(text) = disassemble(assembler, bytes) else {
                continue;
            };
            checked += 1;
            // Either one encoding that disassembles to the same text, or an ambiguity whose
            // candidates include one that does
            let reassembled = assemble(assembler, &text);
            let ok = match &reassembled {
                Ok(bytes) => disassemble(assembler, bytes).as_deref() == Some(text.as_str()),
                Err(AsmError::Ambiguous(candidates)) => candidates
                    .iter()
                    .any(|candidate| candidate.disassembly == text),
                Err(AsmError::NoMatch) => false,
            };
            if !ok {
                failures.push(format!("{:02x?} {:?} -> {:02x?}", bytes, text, reassembled));
            }
        }
        assert!(checked > 0, "no generated word disassembled");
        assert!(
            failures.is_empty(),
            "{} of {} round trips failed, first 10:\n{}",
            failures.len(),
            checked,
            failures[..failures.len().min(10)].join("\n")
        );
    }

    #[test]
    fn risc_unsigned_immediate() {
        let asm = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("or r15, r15, 0xffff", vec![0x17, 0xf8, 0xff, 0xff]),
        ]);
    }

    #[test]
    fn risc_signed_immediate() {
        let asm = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("add r1, r2, -1", vec![0x01, 0x0c, 0xff, 0xff]),
        ]);
    }

    #[test]
    fn risc_shifted_immediate() {
        let asm = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("add r1, r2, 0xffff0000", vec![0x01, 0x0a, 0xff, 0xff]),
            ("add r1, r2, 0x123400", vec![0x01, 0x0e, 0x12, 0x34]),
        ]);
    }

    #[test]
    fn risc_negative_shifted_immediate() {
        let asm = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("add r1, r2, -0x10000", vec![0x01, 0x0e, 0xff, 0x00]),
        ]);
    }

    #[test]
    fn risc_all_ones_immediate() {
        let asm = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("add r1, r2, 0xffffffffffffffff", vec![0x01, 0x0c, 0xff, 0xff]),
        ]);
    }

    #[test]
    fn risc_ambiguous_immediates() {
        let asm = load("examples/risc.slaspec");
        // The IMM16 forms overlap: uimm16, simm16, uimm16 << 16 and simm16 << 8
        #[rustfmt::skip]
        assert_ambiguous(&asm, &[
            ("add r4, r5, 0x1234", 2),
            ("add r1, r2, 12", 2),
            ("sub r0, r0, 0x0", 4),
            ("add r1, r2, -0x100", 2),
            ("add r1, r2, -0x8000", 2),
            ("add r1, r2, 0x10000", 2),
            ("unk.0x7 r3, r4, 0x1", 2),
        ]);
    }

    #[test]
    fn risc_register_operands() {
        let asm = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("add r0, r0, r0", vec![0xf8, 0x00, 0x00, 0x00]),
            ("xor r15, r14, r13", vec![0xff, 0x7e, 0x80, 0x04]),
            ("sub r1, r10, r15", vec![0xfd, 0x0f, 0x80, 0x01]),
        ]);
    }

    #[test]
    fn risc_unknown_opcodes() {
        let asm = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("unk.0x20 r1, r2, r3", vec![0xf9, 0x09, 0x80, 0x20]),
        ]);
    }

    #[test]
    fn risc_rejects_unencodable() {
        let asm = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_rejects(&asm, &[
            "",
            "mul r1, r2, r3",
            "add r16, r1, r2",
            "add r1, r2",
            "add r1, r2,",
            "add r1, r2, foo",
            "add r1, r2, 0x10001",
            "add r1, r2, 0x100000000",
            "unk.0x20 r1, r2, 0x5",
        ]);
    }

    #[test]
    fn risc_rejects_signed_out_of_range() {
        let asm = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_rejects(&asm, &[
            "add r1, r2, -0x8001",
        ]);
    }

    #[test]
    fn risc_returns_remaining_input() {
        let asm = load("examples/risc.slaspec");
        let prefix = |input| {
            asm.assemble_candidates(input)
                .into_iter()
                .map(|(rest, constraints)| (rest, constraints.to_bytes().unwrap()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            prefix("and r1, r2, r3 ; next"),
            vec![(" ; next", vec![0xf9, 0x09, 0x80, 0x03])]
        );
        assert!(prefix("add r4, r5, 0x1234\n").contains(&("\n", vec![0x02, 0xa0, 0x12, 0x34])));

        // A complete instruction may be followed by whitespace, but nothing else
        assert_eq!(
            assemble(&asm, "and r1, r2, r3 \n"),
            Ok(vec![0xf9, 0x09, 0x80, 0x03])
        );
        assert_eq!(
            assemble(&asm, "and r1, r2, r3 ; next"),
            Err(AsmError::NoMatch)
        );
    }

    #[test]
    fn risc_roundtrip() {
        let asm = load("examples/risc.slaspec");
        assert_roundtrips(&asm, 4, 500);
    }

    #[test]
    fn vliw_ambiguous_zero_constants() {
        let asm = load("examples/vliw.slaspec");
        // A 0 in slots 1 and 2 is either the shared constant or the op1/op2_const_zero literal
        #[rustfmt::skip]
        assert_ambiguous(&asm, &[
            ("{ unk.0x1 r2, r3, 0 ; unk.0x2 r4, r5, 0 ; unk.0x3 r6, r7, 0 ; unk.0x4 r8, r9 }", 4),
        ]);
    }

    #[test]
    fn vliw_rejects_unencodable() {
        let asm = load("examples/vliw.slaspec");
        #[rustfmt::skip]
        assert_rejects(&asm, &[
            "{ unk.0x1 r2, r3, 0x5 ; unk.0x2 r4, r5, 0 ; unk.0x3 r6, r7, 0 ; unk.0x4 r8, r9 }",
            "{ unk.0x0 r32, r1, 0 }",
            "{ unk.0x20 r1, r2, 0 }",
            "{ unk.0x0 r1, r2, 0x1 ",
        ]);
    }

    #[test]
    fn vliw_roundtrip() {
        let asm = load("examples/vliw.slaspec");
        assert_roundtrips(&asm, 8, 300);
    }
}
