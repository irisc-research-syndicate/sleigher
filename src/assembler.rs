use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Debug,
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use sleigh_rs::disassembly::{
    Assertation, Expr, ExprElement, Op, OpUnary, ReadScope, VariableId, WriteScope,
};
use sleigh_rs::display::DisplayElement;
use sleigh_rs::meaning::{AttachNumber, AttachVarnode, Meaning};
use sleigh_rs::pattern::{CmpOp, Verification};
use sleigh_rs::table::{Constructor, Table};
use sleigh_rs::{ContextId, Endian, Number, Sleigh, TableId, TokenFieldId, TokenId};
use z3::ast::{Ast, Bool, BV};

use anyhow::{anyhow, bail};

use crate::context::{Context, ContextFlow};
use crate::disassembler::Disassembler;
use crate::value::{parse_number, parse_number_exact};

/// Labels a program defines: resolved to an address, or `None` while their address is unknown
pub type Labels = BTreeMap<String, Option<u64>>;

const NO_LABELS: &Labels = &BTreeMap::new();

/// Passes over a program before label addresses have to stop changing
const MAX_PASSES: usize = 16;

/// One source line of an assembled program that holds an instruction
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    /// 1-based line number in the source
    pub line_no: usize,
    pub address: u64,
    pub bytes: Vec<u8>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub base: u64,
    pub bytes: Vec<u8>,
    pub labels: BTreeMap<String, u64>,
    pub lines: Vec<Line>,
}

/// A source line split into its label definitions and instruction text
struct SourceLine<'s> {
    line_no: usize,
    labels: Vec<&'s str>,
    instruction: Option<&'s str>,
}

/// Strip a `//` comment and peel off leading `name:` label definitions
fn parse_source_line(line_no: usize, line: &str) -> SourceLine<'_> {
    let mut rest = line.split("//").next().unwrap_or_default().trim();
    let mut labels = vec![];
    while let Some((after, name)) = parse_identifier(rest) {
        let Some(after) = after.strip_prefix(':') else {
            break;
        };
        labels.push(name);
        rest = after.trim_start();
    }
    SourceLine {
        line_no,
        labels,
        instruction: (!rest.is_empty()).then_some(rest),
    }
}

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

/// Every index whose name prefixes `s`, e.g. both `r1` and `r10`, longest match first.
/// Indices that share a name each give a parse.
fn parse_attach_names<'a, 'n>(
    names: impl Iterator<Item = (usize, &'n str)>,
    s: &'a str,
) -> Parses<'a, i64> {
    let mut parses = names
        .filter_map(|(index, name)| Some((s.strip_prefix(name)?, index as i64)))
        .collect::<Parses<'a, i64>>();
    parses.sort_by_key(|(rest, _)| rest.len());
    parses
}

/// Every index whose number in the attach list is the decimal or hex number that starts `s`.
/// Indices that share a number each give a parse.
fn parse_attach_number<'a>(attach_number: &AttachNumber, s: &'a str) -> Parses<'a, i64> {
    let Some((s, operand)) = parse_number_exact(true, s) else {
        return vec![];
    };
    attach_number
        .0
        .iter()
        .filter(|(_, number)| number.signed_super() == operand)
        .map(|(index, _)| (s, *index as i64))
        .collect()
}

fn parse_space1(s: &str) -> Option<&str> {
    let rest = s.trim_start_matches([' ', '\t']);
    (rest.len() < s.len()).then_some(rest)
}

/// `[A-Za-z_.][A-Za-z0-9_.]*`
pub fn parse_identifier(s: &str) -> Option<(&str, &str)> {
    let is_start = |c: char| c.is_ascii_alphabetic() || c == '_' || c == '.';
    if !s.starts_with(is_start) {
        return None;
    }
    let len = s
        .find(|c: char| !(is_start(c) || c.is_ascii_digit()))
        .unwrap_or(s.len());
    Some((&s[len..], &s[..len]))
}

/// A constructor matched while assembling
#[derive(Debug, Clone)]
pub struct Instance<'asm> {
    pub constructor: &'asm Constructor,
    /// The parent instance and the index of its pattern block that holds this subtable
    pub parent: Option<(usize, usize)>,
}

/// A token occurrence: constructor instance, pattern block in that constructor, token
pub type TokenKey = (usize, usize, TokenId);

static NEXT_INSTANCE: AtomicUsize = AtomicUsize::new(0);

/// `l op r`, comparing as the disassembler does: signed values sign extended
fn compare<'asm>(op: CmpOp, signed: bool, l: &BV<'asm>, r: &BV<'asm>) -> Bool<'asm> {
    match (op, signed) {
        (CmpOp::Eq, _) => l._eq(r),
        (CmpOp::Ne, _) => l._eq(r).not(),
        (CmpOp::Lt, false) => l.bvult(r),
        (CmpOp::Gt, false) => l.bvugt(r),
        (CmpOp::Le, false) => l.bvule(r),
        (CmpOp::Ge, false) => l.bvuge(r),
        (CmpOp::Lt, true) => l.bvslt(r),
        (CmpOp::Gt, true) => l.bvsgt(r),
        (CmpOp::Le, true) => l.bvsle(r),
        (CmpOp::Ge, true) => l.bvsge(r),
    }
}

/// All constraints are quantifier free bit-vector formulas
fn new_solver(ctx: &z3::Context) -> z3::Solver<'_> {
    z3::Solver::new_for_logic(ctx, "QF_BV").unwrap()
}

#[derive(Debug, Clone)]
pub struct Constraints<'asm> {
    pub asm: &'asm InstructionAssembler,

    pub instances: BTreeMap<usize, Instance<'asm>>,
    pub tokens: BTreeMap<TokenKey, BV<'asm>>,
    pub fields: HashMap<(usize, TokenFieldId), BV<'asm>>,

    pub eqs: HashSet<Bool<'asm>>,

    /// The value of each context variable at this point of the parse, by `ContextId`, 64 bits
    /// wide. Context writes are applied before the constructor's display, so its subtables
    /// see all of them, as in Ghidra; subtables pass theirs on in display order.
    pub context: Vec<BV<'asm>>,

    pub inst_start: u64,
    /// Address after the instruction, tied to its length once the whole instruction is parsed
    pub inst_next: BV<'asm>,

    pub labels: &'asm Labels,

    /// One solver for every check of an instruction, much cheaper than a new solver per check
    checker: Rc<z3::Solver<'asm>>,
}

impl<'asm> Constraints<'asm> {
    pub fn new(
        asm: &'asm InstructionAssembler,
        inst_start: u64,
        context: &Context,
        labels: &'asm Labels,
    ) -> Self {
        let context = (0..asm.contexts().len())
            .map(|id| BV::from_i64(&asm.ctx, context.get(asm, ContextId(id)), 64))
            .collect();
        Self {
            asm,
            instances: BTreeMap::new(),
            tokens: BTreeMap::new(),
            fields: HashMap::new(),
            eqs: HashSet::new(),
            context,
            inst_start,
            inst_next: BV::fresh_const(&asm.ctx, "inst_next", 64),
            checker: Rc::new(new_solver(&asm.ctx)),
            labels,
        }
    }

    /// A number or a defined label, as a 64-bit value. Unresolved labels are symbolic.
    pub fn parse_operand<'a>(&self, signed: bool, s: &'a str) -> Option<(&'a str, BV<'asm>)> {
        if let Some((s, value)) = parse_number(signed, s) {
            return Some((s, self.build_u64_const(value as u64, 64)));
        }
        let (s, name) = parse_identifier(s)?;
        let value = match self.labels.get(name)? {
            Some(address) => self.build_u64_const(*address, 64),
            None => BV::new_const(&self.asm.ctx, name, 64),
        };
        Some((s, value))
    }

    pub fn new_instance(
        &mut self,
        constructor: &'asm Constructor,
        parent: Option<(usize, usize)>,
    ) -> usize {
        let id = NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed);
        self.instances.insert(
            id,
            Instance {
                constructor,
                parent,
            },
        );
        id
    }

    pub fn token(&mut self, key: TokenKey) -> BV<'asm> {
        if let Some(bv) = self.tokens.get(&key) {
            return bv.clone();
        }
        let token_id = key.2;
        let token = self.asm.token(token_id);
        let bv = BV::fresh_const(
            &self.asm.ctx,
            token.name(),
            8 * (token.len_bytes().get() as u32),
        );

        // Tie bytes shared with tokens already placed, when the offsets are known before the
        // whole instruction is parsed. That keeps constraint checks while parsing precise;
        // finalize ties the rest.
        if let Some(offset) = self.token_offset(key) {
            // The same token at the same offset is the same value
            let same = self.tokens.iter().find(|(other_key, _)| {
                other_key.2 == key.2 && self.token_offset(**other_key) == Some(offset)
            });
            if let Some((_, same_bv)) = same {
                let same_bv = same_bv.clone();
                self.tokens.insert(key, same_bv.clone());
                return same_bv;
            }
            let len = token.len_bytes().get();
            let overlapping = self
                .tokens
                .iter()
                .filter_map(|(other_key, other_bv)| {
                    Some((self.token_offset(*other_key)?, other_key.2, other_bv))
                })
                .collect::<Vec<_>>();
            for (other_offset, other_id, other_bv) in overlapping {
                let other_len = self.asm.token(other_id).len_bytes().get();
                for position in
                    offset.max(other_offset)..(offset + len).min(other_offset + other_len)
                {
                    let byte = self.token_byte(token_id, &bv, position - offset);
                    let other_byte = self.token_byte(other_id, other_bv, position - other_offset);
                    self.eqs.insert(byte._eq(&other_byte));
                }
            }
        }

        self.tokens.insert(key, bv.clone());
        bv
    }

    /// Offset of an instance in the instruction, if every block before it has a fixed length
    fn instance_offset(&self, id: usize) -> Option<u64> {
        match self.instances[&id].parent {
            None => Some(0),
            Some((parent, block)) => {
                Some(self.instance_offset(parent)? + self.fixed_offset(parent, block)?)
            }
        }
    }

    /// Offset of a block in its instance, if every block before it has a fixed length
    fn fixed_offset(&self, id: usize, block: usize) -> Option<u64> {
        self.instances[&id].constructor.pattern.blocks()[..block]
            .iter()
            .map(|block| block.len().single_len())
            .sum()
    }

    fn token_offset(&self, key: TokenKey) -> Option<u64> {
        Some(self.instance_offset(key.0)? + self.fixed_offset(key.0, key.1)?)
    }

    /// A token field read from the token in `block` of constructor `instance`
    pub fn token_field_at(
        &mut self,
        instance: usize,
        block: usize,
        token_field_id: TokenFieldId,
        sz: Option<u32>,
    ) -> BV<'asm> {
        let token_field = self.asm.sleigh.token_field(token_field_id);
        let field_bv = self
            .fields
            .entry((instance, token_field_id))
            .or_insert_with(|| {
                let field_bv = BV::fresh_const(
                    &self.asm.ctx,
                    token_field.name(),
                    token_field.bits.len().get() as u32,
                );
                field_bv
            })
            .clone();
        let token_bv = self.token((instance, block, token_field.token));
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

    /// Store `value` in a context variable, which keeps as many bits as the variable has
    pub fn set_context(&mut self, id: ContextId, value: BV<'asm>) {
        let context = self.asm.context(id);
        let bits = context.bitrange.bits.len().get() as u32;
        let value = value.extract(bits - 1, 0);
        let value = if context.is_signed() {
            value.sign_ext(64 - bits)
        } else {
            value.zero_ext(64 - bits)
        };
        self.context[id.0] = value;
    }

    /// Take over the constraints of a subtable parsed from a clone of `self`
    pub fn merge(&mut self, other: Constraints<'asm>) {
        self.instances.extend(other.instances);
        self.tokens.extend(other.tokens);
        self.fields.extend(other.fields);
        self.eqs.extend(other.eqs);
        self.context = other.context;
    }

    fn root(&self) -> Option<usize> {
        self.instances
            .iter()
            .find(|(_, instance)| instance.parent.is_none())
            .map(|(id, _)| *id)
    }

    fn children(&self, id: usize, block: usize) -> Vec<usize> {
        self.instances
            .iter()
            .filter(|(_, instance)| instance.parent == Some((id, block)))
            .map(|(child, _)| *child)
            .collect()
    }

    fn block_tokens(&self, id: usize, block: usize) -> Vec<(TokenId, &BV<'asm>)> {
        self.tokens
            .iter()
            .filter(|((instance, token_block, _), _)| *instance == id && *token_block == block)
            .map(|((_, _, token_id), bv)| (*token_id, bv))
            .collect()
    }

    fn block_len(&self, id: usize, block: usize) -> u64 {
        let pattern_block = &self.instances[&id].constructor.pattern.blocks()[block];
        if let Some(len) = pattern_block.len().single_len() {
            return len;
        }
        // Variable length: the longest of the block's tokens and the subtables chosen for it
        let tokens = self
            .block_tokens(id, block)
            .into_iter()
            .map(|(token_id, _)| self.asm.token(token_id).len_bytes().get());
        let children = self
            .children(id, block)
            .into_iter()
            .map(|child| self.instance_len(child));
        tokens
            .chain(children)
            .max()
            .unwrap_or(pattern_block.len().min())
    }

    fn instance_len(&self, id: usize) -> u64 {
        let blocks = self.instances[&id].constructor.pattern.blocks().len();
        (0..blocks).map(|block| self.block_len(id, block)).sum()
    }

    /// Instruction length in bytes
    pub fn len_bytes(&self) -> u64 {
        self.root().map_or(0, |root| self.instance_len(root))
    }

    /// Byte `index` of a token, in instruction order
    fn token_byte(&self, token_id: TokenId, bv: &BV<'asm>, index: u64) -> BV<'asm> {
        let token = self.asm.token(token_id);
        let len = token.len_bytes().get();
        let low = match token.endian() {
            Endian::Big => 8 * (len - 1 - index),
            Endian::Little => 8 * index,
        } as u32;
        bv.extract(low + 7, low)
    }

    /// Every token byte placed at each offset of the instruction
    pub fn layout(&self) -> Vec<Vec<BV<'asm>>> {
        let mut bytes = vec![vec![]; self.len_bytes() as usize];
        if let Some(root) = self.root() {
            self.place(root, 0, &mut bytes);
        }
        bytes
    }

    fn place(&self, id: usize, offset: u64, bytes: &mut Vec<Vec<BV<'asm>>>) {
        let mut block_offset = offset;
        let blocks = self.instances[&id].constructor.pattern.blocks().len();
        for block in 0..blocks {
            for (token_id, bv) in self.block_tokens(id, block) {
                for index in 0..self.asm.token(token_id).len_bytes().get() {
                    let position = (block_offset + index) as usize;
                    if bytes.len() <= position {
                        bytes.resize(position + 1, vec![]);
                    }
                    bytes[position].push(self.token_byte(token_id, bv, index));
                }
            }
            for child in self.children(id, block) {
                self.place(child, block_offset, bytes);
            }
            block_offset += self.block_len(id, block);
        }
    }

    /// Once the whole instruction is parsed: token bytes at the same offset are the same
    /// byte, and inst_next follows the instruction
    pub fn finalize(&mut self) {
        for byte in self.layout() {
            for other in byte.iter().skip(1) {
                self.eq(byte[0]._eq(other));
            }
        }
        let inst_next = self.build_u64_const(self.inst_start + self.len_bytes(), 64);
        let eq = self.inst_next._eq(&inst_next);
        self.eq(eq);
    }

    pub fn solver(&self) -> z3::Solver<'asm> {
        let solver = new_solver(&self.asm.ctx);
        for eq in self.eqs.iter() {
            solver.assert(eq)
        }
        solver
    }

    pub fn check(&self) -> bool {
        self.checker.push();
        for eq in self.eqs.iter() {
            self.checker.assert(eq);
        }
        let result = self.checker.check();
        self.checker.pop(1);
        match result {
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
        let (self_layout, other_layout) = (self.layout(), other.layout());
        if self_layout.len() != other_layout.len() {
            return false;
        }
        let solver = self.solver();
        for eq in other.eqs.iter() {
            solver.assert(eq);
        }
        for (self_byte, other_byte) in self_layout.iter().zip(other_layout.iter()) {
            if let (Some(self_byte), Some(other_byte)) = (self_byte.first(), other_byte.first()) {
                solver.assert(&self_byte._eq(other_byte));
            }
        }
        solver.check() == z3::SatResult::Sat
    }

    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        log::debug!("Generating instruction bytes");
        let model = self.model()?;
        self.layout()
            .iter()
            .map(|byte| match byte.first() {
                Some(bv) => Some(model.eval(bv, true)?.as_u64()? as u8),
                None => Some(0),
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct Variables<'asm> {
    constructor: &'asm Constructor,
    instance: usize,
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
    pub fn new(
        mut constraints: Constraints<'asm>,
        constructor: &'asm Constructor,
        parent: Option<(usize, usize)>,
    ) -> Self {
        let instance = constraints.new_instance(constructor, parent);
        Self {
            constraints,
            constructor,
            instance,
            variables: HashMap::new(),
        }
    }

    /// The pattern block of this constructor whose token holds the field
    fn field_block(&self, token_field_id: TokenFieldId) -> usize {
        let token_of = |field: TokenFieldId| self.asm.sleigh.token_field(field).token;
        let token = token_of(token_field_id);
        self.constructor
            .pattern
            .blocks()
            .iter()
            .position(|block| {
                block
                    .token_fields()
                    .iter()
                    .any(|produced| token_of(produced.field) == token)
                    || block.verifications().iter().any(|verification| {
                        matches!(verification, Verification::TokenFieldCheck { field, .. }
                            if token_of(*field) == token)
                    })
            })
            .unwrap_or(0)
    }

    pub fn token_field(&mut self, token_field_id: TokenFieldId, sz: Option<u32>) -> BV<'asm> {
        let block = self.field_block(token_field_id);
        let instance = self.instance;
        self.constraints
            .token_field_at(instance, block, token_field_id, sz)
    }

    /// The pattern block of this constructor that holds the subtable
    fn table_block(&self, table_id: TableId) -> usize {
        self.constructor
            .pattern
            .blocks()
            .iter()
            .position(|block| block.tables().iter().any(|table| table.table == table_id))
            .unwrap_or(0)
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

    /// Add disassembly action assignments as constraints
    pub fn assert_all(&mut self, assertions: &[Assertation]) {
        for assertion in assertions {
            match assertion {
                // Commits do not constrain the encoding, program assembly reads them back
                // from the disassembled bytes
                Assertation::GlobalSet(_) => {}
                Assertation::Assignment(assignment) => {
                    let value = self.build_expr_bv(&assignment.right, 64);
                    match assignment.left {
                        WriteScope::Context(context_id) => self.set_context(context_id, value),
                        WriteScope::Local(variable_id) => {
                            let var = self.variable(variable_id);
                            self.eq(var._eq(&value))
                        }
                    }
                }
            }
        }
    }

    pub fn build_expr_bv(&mut self, expr: &Expr, sz: u32) -> BV<'asm> {
        let expr_bv = match expr {
            Expr::Value(expr_element) => match expr_element {
                ExprElement::Value { value, location: _ } => match value {
                    ReadScope::Integer(number) => match number {
                        Number::Positive(x) => self.build_u64_const(*x, sz),
                        Number::Negative(x) => self.build_i64_const(-(*x as i64), sz),
                    },
                    ReadScope::Context(context_id) => self.context[context_id.0].clone(),
                    ReadScope::TokenField(token_field_id) => {
                        self.token_field(*token_field_id, Some(sz))
                    }
                    ReadScope::InstStart(_inst_start) => self.build_u64_const(self.inst_start, sz),
                    ReadScope::InstNext(_inst_next) => self.inst_next.clone(),
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
    /// Whether any constructor has a `globalset`, so program assembly has to follow commits
    has_globalset: bool,
}

impl Deref for InstructionAssembler {
    type Target = Sleigh;

    fn deref(&self) -> &Self::Target {
        &self.sleigh
    }
}

impl InstructionAssembler {
    pub fn new(sleigh: Sleigh) -> Self {
        let has_globalset = sleigh
            .tables()
            .iter()
            .flat_map(|table| table.constructors().iter())
            .flat_map(|constructor| {
                let pattern = &constructor.pattern;
                pattern
                    .blocks()
                    .iter()
                    .flat_map(|block| {
                        block
                            .pre_disassembler()
                            .iter()
                            .chain(block.post_disassembler())
                    })
                    .chain(pattern.disassembly_pos_match())
            })
            .any(|assertion| matches!(assertion, Assertation::GlobalSet(_)));
        Self {
            sleigh,
            ctx: z3::Context::new(&z3::Config::new()),
            has_globalset,
        }
    }

    /// Assemble one instruction at address 0 in the zero context, see `assemble_instruction_at`
    pub fn assemble_instruction<'asm>(&'asm self, s: &str) -> Result<Constraints<'asm>, AsmError> {
        self.assemble_instruction_at(s, 0, &Context::new(self), NO_LABELS)
    }

    /// Assemble one instruction at `inst_start`, decoded in `context`, that must consume all of
    /// `s` (up to trailing whitespace) and have exactly one encoding. Parses that can produce the
    /// same bytes count as one encoding. Operands may name any of `labels`.
    pub fn assemble_instruction_at<'asm>(
        &'asm self,
        s: &str,
        inst_start: u64,
        context: &Context,
        labels: &'asm Labels,
    ) -> Result<Constraints<'asm>, AsmError> {
        let mut encodings: Vec<Constraints<'asm>> = vec![];
        for (rest, candidate) in self.assemble_candidates_at(s, inst_start, context, labels) {
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
                    .map(|encoding| self.candidate(encoding, context))
                    .collect(),
            )),
        }
    }

    /// Assemble a program of one instruction per line, starting at `base`. Lines may start with
    /// `name:` label definitions, `//` starts a comment, and operands may name any label.
    ///
    /// The first pass treats every label as unknown to find instruction lengths. Later passes
    /// resolve the labels, require exactly one encoding per instruction and repeat until the
    /// label addresses stop changing.
    pub fn assemble_program(&self, source: &str, base: u64) -> anyhow::Result<Program> {
        self.assemble_program_in_context(source, base, &Context::new(self))
    }

    /// Like `assemble_program`, starting from `context`. The context flows from line to line,
    /// with the values `globalset`s commit, as the emulator would execute the lines in order:
    /// commits to earlier lines are not seen. While labels are unresolved, commits to a label
    /// may go astray.
    pub fn assemble_program_in_context(
        &self,
        source: &str,
        base: u64,
        context: &Context,
    ) -> anyhow::Result<Program> {
        let source_lines = source
            .lines()
            .enumerate()
            .map(|(index, line)| parse_source_line(index + 1, line))
            .collect::<Vec<_>>();

        let mut labels = Labels::new();
        for line in source_lines.iter() {
            for name in line.labels.iter() {
                if self
                    .varnodes()
                    .iter()
                    .any(|varnode| varnode.name() == *name)
                {
                    bail!("line {}: label {:?} is a register name", line.line_no, name);
                }
                if labels.insert(name.to_string(), None).is_some() {
                    bail!("line {}: duplicate label {:?}", line.line_no, name);
                }
            }
        }

        for pass in 0..MAX_PASSES {
            let resolved = pass > 0;
            let mut address = base;
            let mut addresses = Labels::new();
            let mut lines = vec![];
            let mut flow = ContextFlow::new(context.clone());
            for line in source_lines.iter() {
                for name in line.labels.iter() {
                    addresses.insert(name.to_string(), Some(address));
                }
                let Some(text) = line.instruction else {
                    continue;
                };
                let error = |err: &dyn std::fmt::Display| {
                    anyhow!("line {}: {:?}: {}", line.line_no, text, err)
                };
                let context = flow.at(self, address);
                let bytes = if resolved {
                    let constraints = self
                        .assemble_instruction_at(text, address, &context, &labels)
                        .map_err(|err| error(&err))?;
                    constraints
                        .to_bytes()
                        .ok_or_else(|| error(&"constraints produced no bytes"))?
                } else {
                    // Labels are unknown, so only take the shortest candidate
                    let shortest = self
                        .assemble_candidates_at(text, address, &context, &labels)
                        .into_iter()
                        .filter(|(rest, _)| rest.trim_end().is_empty())
                        .map(|(_, candidate)| candidate)
                        .min_by_key(|candidate| candidate.len_bytes())
                        .ok_or_else(|| error(&AsmError::NoMatch))?;
                    match self.has_globalset {
                        true => shortest
                            .to_bytes()
                            .ok_or_else(|| error(&"constraints produced no bytes"))?,
                        false => vec![0; shortest.len_bytes() as usize],
                    }
                };
                // What the instruction commits is read back from its encoding
                let commits = match self.has_globalset {
                    true => Disassembler::new(self)
                        .disassemble(address, &context, &bytes)
                        .map(|instruction| instruction.commits)
                        .map_err(|err| error(&err))?,
                    false => vec![],
                };
                flow.advance(self, context, &commits);
                lines.push(Line {
                    line_no: line.line_no,
                    address,
                    bytes,
                    source: text.to_string(),
                });
                address += lines.last().unwrap().bytes.len() as u64;
            }

            if resolved && addresses == labels {
                return Ok(Program {
                    base,
                    bytes: lines.iter().flat_map(|line| line.bytes.clone()).collect(),
                    labels: labels
                        .into_iter()
                        .map(|(name, address)| (name, address.unwrap()))
                        .collect(),
                    lines,
                });
            }
            labels = addresses;
        }
        bail!("label addresses did not settle after {} passes", MAX_PASSES)
    }

    fn candidate(&self, constraints: &Constraints, context: &Context) -> Candidate {
        let bytes = constraints.to_bytes().unwrap_or_default();
        let disassembly = Disassembler::new(&self.sleigh)
            .disassemble(constraints.inst_start, context, &bytes)
            .map(|instruction| instruction.to_string())
            .unwrap_or_else(|err| format!("<{}>", err));
        Candidate { bytes, disassembly }
    }

    /// Every way the instruction table matches a prefix of `s` at address 0 in the zero context
    pub fn assemble_candidates<'a, 'asm>(&'asm self, s: &'a str) -> Parses<'a, Constraints<'asm>> {
        self.assemble_candidates_at(s, 0, &Context::new(self), NO_LABELS)
    }

    /// Every way the instruction table matches a prefix of `s` at `inst_start` in `context`.
    /// Candidates whose constraints fail once `inst_next` is tied to their length (e.g. a branch
    /// out of range) are dropped.
    pub fn assemble_candidates_at<'a, 'asm>(
        &'asm self,
        s: &'a str,
        inst_start: u64,
        context: &Context,
        labels: &'asm Labels,
    ) -> Parses<'a, Constraints<'asm>> {
        let constraints = Constraints::new(self, inst_start, context, labels);
        self.assemble_table(self.table(self.instruction_table()), constraints, s, None)
            .into_iter()
            .filter_map(|(rest, mut candidate)| {
                candidate.finalize();
                candidate.check().then_some((rest, candidate))
            })
            .collect()
    }

    pub fn assemble_table<'a, 'asm>(
        &'asm self,
        table: &'asm Table,
        constraints: Constraints<'asm>,
        s: &'a str,
        parent: Option<(usize, usize)>,
    ) -> Parses<'a, Constraints<'asm>> {
        table
            .constructors()
            .iter()
            .flat_map(|constructor| {
                self.assemble_constructor(constructor, constraints.clone(), s, parent)
            })
            .collect()
    }

    pub fn assemble_constructor<'a, 'asm>(
        &'asm self,
        constructor: &'asm Constructor,
        constraints: Constraints<'asm>,
        s: &'a str,
        parent: Option<(usize, usize)>,
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

        let mut variables = Variables::new(constraints, constructor, parent);
        // The constructor's own context changes come after its pattern is matched
        let context = variables.context.clone();

        for block in constructor.pattern.blocks() {
            for verification in block.verifications() {
                match verification {
                    Verification::ContextCheck {
                        context: context_id,
                        op,
                        value,
                    } => {
                        let value_bv = variables.build_expr_bv(value.expr(), 64);
                        let signed = self.context(*context_id).is_signed();
                        variables.eq(compare(*op, signed, &context[context_id.0], &value_bv));
                    }
                    Verification::TableBuild {
                        produced_table: _,
                        verification: _,
                    } => {
                        continue;
                    }
                    Verification::TokenFieldCheck { field, op, value } => {
                        let field_bv = variables.token_field(*field, None);
                        let value_bv = variables.build_expr_bv(value.expr(), field_bv.get_size());
                        let signed = self.token_field(*field).raw_value_is_signed();
                        variables.eq(compare(*op, signed, &field_bv, &value_bv));
                    }
                    Verification::SubPattern {
                        location: _,
                        pattern: _,
                    } => todo!(),
                }
            }
            variables.assert_all(block.pre_disassembler());
            variables.assert_all(block.post_disassembler());
        }
        // Post-disassembler and post-match assertions are the ones that need inst_next; as
        // constraints the order they are evaluated in does not matter
        variables.assert_all(constructor.pattern.disassembly_pos_match());

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
            DisplayElement::Context(context_id) => {
                let context = self.context(*context_id);
                log::trace!("CONTEXT: {:?} {:?}", context.name(), s);
                let value = variables.context[context_id.0].clone();
                self.assemble_value(context.meaning(), value, variables, s)
            }
            DisplayElement::TokenField(token_field_id) => {
                let token_field = self.token_field(*token_field_id);
                log::trace!("TOKEN_FIELD: {:?} {:?}", token_field.name(), s);
                let token_field_bv = variables.token_field(*token_field_id, None);
                let size = token_field_bv.get_size();
                let value = if size == 64 {
                    token_field_bv
                } else if token_field.raw_value_is_signed() {
                    token_field_bv.sign_ext(64 - size)
                } else {
                    token_field_bv.zero_ext(64 - size)
                };
                self.assemble_value(token_field.meaning(), value, variables, s)
            }
            DisplayElement::InstStart(_) | DisplayElement::InstNext(_) => {
                let Some((s, value)) = variables.parse_operand(false, s) else {
                    return vec![];
                };
                let address = match elem {
                    DisplayElement::InstStart(_) => {
                        variables.build_u64_const(variables.inst_start, 64)
                    }
                    _ => variables.inst_next.clone(),
                };
                variables.eq(address._eq(&value));
                vec![(s, variables)]
            }
            DisplayElement::Table(table_id) => {
                let table = self.table(*table_id);
                log::trace!("TABLE: {:?}/{:?} {:?}", table.name(), table_id, s);
                let parent = Some((variables.instance, variables.table_block(*table_id)));
                self.assemble_table(table, variables.constraints.clone(), s, parent)
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
                let Some((s, value)) = variables.parse_operand(true, s) else {
                    return vec![];
                };
                let var = variables.variable(*variable_id);
                variables.eq(var._eq(&value));
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

    /// Parse an operand that displays the 64-bit `value` with `meaning`, a token field or a
    /// context variable
    fn assemble_value<'a, 'asm>(
        &'asm self,
        meaning: Meaning,
        value: BV<'asm>,
        mut variables: Variables<'asm>,
        s: &'a str,
    ) -> Parses<'a, Variables<'asm>> {
        let indices = match meaning {
            Meaning::NoAttach(value_fmt) => {
                let Some((s, operand)) = variables.parse_operand(value_fmt.signed, s) else {
                    return vec![];
                };
                // This rejects operands that do not fit the value
                variables.eq(value._eq(&operand));
                return vec![(s, variables)];
            }
            Meaning::Varnode(attach_varnode_id) => {
                self.parse_attach_varnode(self.attach_varnode(attach_varnode_id), s)
            }
            Meaning::Literal(attach_literal_id) => {
                let names = self.attach_literal(attach_literal_id).0.iter();
                parse_attach_names(names.map(|(index, name)| (*index, name.as_str())), s)
            }
            Meaning::Number(_, attach_number_id) => {
                parse_attach_number(self.attach_number(attach_number_id), s)
            }
        };
        indices
            .into_iter()
            .map(|(s, index)| {
                let mut variables = variables.clone();
                let index = variables.build_i64_const(index, 64);
                variables.eq(value._eq(&index));
                (s, variables)
            })
            .collect()
    }

    /// Every register name in the attach list that prefixes `s`, e.g. both `r1` and `r10`,
    /// longest match first
    pub fn parse_attach_varnode<'a>(
        &self,
        attach_varnode: &AttachVarnode,
        s: &'a str,
    ) -> Parses<'a, i64> {
        let names = attach_varnode
            .0
            .iter()
            .map(|(index, id)| (*index, self.varnode(*id).name()));
        parse_attach_names(names, s)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::path::Path;

    fn run_tests(slaspec_path: impl AsRef<Path>, tests: &[(&str, Vec<u8>)]) {
        let _ = env_logger::try_init();
        log::info!("Loading slaspec: {:?}", slaspec_path.as_ref());
        let slaspec = sleigh_rs::file_to_sleigh(slaspec_path.as_ref())
            .unwrap_or_else(|_| panic!("Could not load slaspec: {:?}", slaspec_path.as_ref()));
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
        assemble_at(assembler, input, 0)
    }

    fn assemble_at(
        assembler: &InstructionAssembler,
        input: &str,
        address: u64,
    ) -> Result<Vec<u8>, AsmError> {
        assemble_with(assembler, input, address, NO_LABELS)
    }

    fn assemble_with(
        assembler: &InstructionAssembler,
        input: &str,
        address: u64,
        labels: &Labels,
    ) -> Result<Vec<u8>, AsmError> {
        let constraints =
            assembler.assemble_instruction_at(input, address, &Context::new(assembler), labels)?;
        Ok(constraints
            .to_bytes()
            .expect("Constraints failed to produce bytes"))
    }

    fn disassemble(assembler: &InstructionAssembler, bytes: &[u8]) -> Option<String> {
        let disassembler = Disassembler::new(assembler);
        let instruction = disassembler
            .disassemble(0, &Context::new(assembler), bytes)
            .ok()?;
        Some(instruction.to_string())
    }

    fn assert_encodes(assembler: &InstructionAssembler, tests: &[(&str, Vec<u8>)]) {
        assert_encodes_at(assembler, 0, tests)
    }

    fn assert_encodes_at(
        assembler: &InstructionAssembler,
        address: u64,
        tests: &[(&str, Vec<u8>)],
    ) {
        let mut failures = vec![];
        for (input, expected_bytes) in tests.iter() {
            match assemble_at(assembler, input, address) {
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
    fn risc_label_operands() {
        let asm = load("examples/risc.slaspec");
        let labels = Labels::from([("data".to_string(), Some(0x12340000))]);
        // Only uimm16 << 16 can produce 0x12340000
        assert_eq!(
            assemble_with(&asm, "add r1, r0, data", 0, &labels),
            Ok(vec![0x00, 0x0a, 0x12, 0x34])
        );
        // Register names stay registers when labels are defined
        assert_eq!(
            assemble_with(&asm, "add r1, r2, r3", 0, &labels),
            Ok(vec![0xf9, 0x09, 0x80, 0x00])
        );
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
    fn cisc_operands() {
        let asm = load("examples/cisc.slaspec");
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("nop", vec![0x00]),
            ("mov r1, r2", vec![0x01, 0x0a]),
            ("mov r1, [r2]", vec![0x01, 0x4a]),
            ("mov r1, [r2+0x4]", vec![0x01, 0x8a, 0x04]),
            ("mov r1, #0x12345678", vec![0x01, 0xc8, 0x12, 0x34, 0x56, 0x78]),
            ("cmp r1, #0x2a", vec![0x07, 0xc8, 0x00, 0x00, 0x00, 0x2a]),
            ("mov [r2], r1", vec![0x08, 0x4a]),
            ("mov [r2+-0x4], r1", vec![0x08, 0x8a, 0xfc]),
            ("jmp 0x2000", vec![0x10, 0x00, 0x00, 0x20, 0x00]),
        ]);
    }

    #[test]
    fn cisc_relative_branches() {
        let asm = load("examples/cisc.slaspec");
        // dest = inst_next + simm8, with inst_next = 0x1002
        #[rustfmt::skip]
        assert_encodes_at(&asm, 0x1000, &[
            ("jz 0x1012", vec![0x11, 0x10]),
            ("jz 0x1000", vec![0x11, 0xfe]),
            ("jnz 0x1081", vec![0x12, 0x7f]),
            ("jnz 0xf82", vec![0x12, 0x80]),
        ]);
        assert_eq!(
            assemble_at(&asm, "jz 0x1082", 0x1000),
            Err(AsmError::NoMatch)
        );
        assert_eq!(
            assemble_at(&asm, "jz 0xf81", 0x1000),
            Err(AsmError::NoMatch)
        );
    }

    #[test]
    fn cisc_label_operands() {
        let asm = load("examples/cisc.slaspec");
        let labels = Labels::from([
            ("target".to_string(), Some(0x1012)),
            ("unknown".to_string(), None),
        ]);
        let assemble = |input| assemble_with(&asm, input, 0x1000, &labels);
        assert_eq!(assemble("jz target"), Ok(vec![0x11, 0x10]));
        assert_eq!(
            assemble("jmp target"),
            Ok(vec![0x10, 0x00, 0x00, 0x10, 0x12])
        );
        assert_eq!(
            assemble("mov r1, #target"),
            Ok(vec![0x01, 0xc8, 0x00, 0x00, 0x10, 0x12])
        );
        // An unresolved label can take any value, so only the instruction's shape is known
        assert_eq!(assemble("jz unknown").map(|bytes| bytes.len()), Ok(2));
        // Only defined labels are operands
        assert_eq!(assemble("jz nowhere"), Err(AsmError::NoMatch));
    }

    #[test]
    fn cisc_program_counting_loop() {
        let asm = load("examples/cisc.slaspec");
        let source = "
            // r1 = r2 + (r2 - 1) + ... + 1
            start:  xor r1, r1
            loop:   add r1, r2
                    sub r2, r3  // r3 = 1
                    jnz loop
            done:   nop
        ";
        let program = asm.assemble_program(source, 0x1000).unwrap();
        // Same bytes as the hand encoded emulator test cisc_counting_loop
        assert_eq!(
            program.bytes,
            vec![0x06, 0x09, 0x02, 0x0a, 0x03, 0x13, 0x12, 0xfa, 0x00]
        );
        assert_eq!(
            program.labels,
            BTreeMap::from([
                ("start".to_string(), 0x1000),
                ("loop".to_string(), 0x1002),
                ("done".to_string(), 0x1008),
            ])
        );
        let lines = program
            .lines
            .iter()
            .map(|line| (line.line_no, line.address))
            .collect::<Vec<_>>();
        assert_eq!(
            lines,
            vec![
                (3, 0x1000),
                (4, 0x1002),
                (5, 0x1004),
                (6, 0x1006),
                (7, 0x1008)
            ]
        );
    }

    #[test]
    fn cisc_program_forward_references() {
        let asm = load("examples/cisc.slaspec");
        let source = "
            jz end
            jmp end

            mov r1, #end
            end:
            nop
        ";
        let program = asm.assemble_program(source, 0x1000).unwrap();
        #[rustfmt::skip]
        assert_eq!(program.bytes, vec![
            0x11, 0x0b,                         // 0x1000: jz end
            0x10, 0x00, 0x00, 0x10, 0x0d,       // 0x1002: jmp end
            0x01, 0xc8, 0x00, 0x00, 0x10, 0x0d, // 0x1007: mov r1, #end
            0x00,                               // 0x100d: end: nop
        ]);
    }

    #[test]
    fn program_errors() {
        let cisc = load("examples/cisc.slaspec");
        let risc = load("examples/risc.slaspec");
        let error = |asm: &InstructionAssembler, source: &str| {
            asm.assemble_program(source, 0x1000)
                .unwrap_err()
                .to_string()
        };
        let far = format!("jz end\n{}end: nop", "nop\n".repeat(130));
        let errors = [
            (
                error(&cisc, &far),
                "line 1: \"jz end\": no constructor matches the input",
            ),
            (
                error(&cisc, "a: nop\na: nop"),
                "line 2: duplicate label \"a\"",
            ),
            (
                error(&cisc, "r1: nop"),
                "line 1: label \"r1\" is a register name",
            ),
            (
                error(&cisc, "nop\nmul r1, r2"),
                "line 2: \"mul r1, r2\": no constructor matches the input",
            ),
            (
                error(&risc, "add r1, r2, r3\nadd r1, r2, 0x5"),
                "line 2: \"add r1, r2, 0x5\": ambiguous, 2 encodings:",
            ),
        ];
        for (actual, expected) in errors.iter() {
            assert!(
                actual.starts_with(expected),
                "{:?} does not start with {:?}",
                actual,
                expected
            );
        }
    }

    #[test]
    fn risc_program_label_operand() {
        let asm = load("examples/risc.slaspec");
        // data lands at 0x12340000, which only the uimm16 << 16 form can produce
        let program = asm
            .assemble_program("add r1, r0, data\ndata:", 0x1233fffc)
            .unwrap();
        assert_eq!(program.bytes, vec![0x00, 0x0a, 0x12, 0x34]);
        assert_eq!(program.labels["data"], 0x12340000);
    }

    #[test]
    fn cisc_roundtrip() {
        let asm = load("examples/cisc.slaspec");
        assert_roundtrips(&asm, 6, 2000);
    }

    #[test]
    fn layout_operands_out_of_byte_order() {
        let asm = load("examples/layout.slaspec");
        // imm8 is displayed before reg, but its byte comes after
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("ldi 0x5, r2", vec![0x01, 0x02, 0x05]),
        ]);
    }

    #[test]
    fn layout_repeated_token() {
        let asm = load("examples/layout.slaspec");
        // REG and OPND both use regbyte, at different offsets
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("mov r1, r2", vec![0x02, 0x01, 0x02]),
            ("mov r1, #0x7", vec![0x02, 0x01, 0x80, 0x07]),
            ("add r2, r3", vec![0x03, 0x02, 0x03]),
            ("add #0x7, r3", vec![0x03, 0x80, 0x07, 0x03]),
        ]);
    }

    #[test]
    fn layout_roundtrip() {
        let asm = load("examples/layout.slaspec");
        assert_roundtrips(&asm, 4, 5000);
    }

    #[test]
    fn solver_split_and_scaled_immediates() {
        let asm = load("examples/solver.slaspec");
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("addi r1, r2, 0x5", vec![0x93, 0x00, 0x51, 0x00]),
            ("addi r1, r2, -0x1", vec![0x93, 0x00, 0xf1, 0xff]),
            ("addi r1, r2, -0x800", vec![0x93, 0x00, 0x01, 0x80]),
            // off = (imm_hi << 5) | imm_lo
            ("sw r2, -0x4(r1)", vec![0x23, 0xae, 0x20, 0xfe]),
            ("sw r3, 0x7ff(r4)", vec![0xa3, 0x2f, 0x32, 0x7e]),
            ("sw r3, -0x800(r4)", vec![0x23, 0x20, 0x32, 0x80]),
            // disp = d8 * 4 + 8
            ("ldx r1, 0x8(r2)", vec![0x83, 0x30, 0x01, 0x00]),
            ("ldx r1, -0x8(r2)", vec![0x83, 0x30, 0xc1, 0x0f]),
            ("ldx r1, 0x204(r2)", vec![0x83, 0x30, 0xf1, 0x07]),
            ("ldx r1, -0x1f8(r2)", vec![0x83, 0x30, 0x01, 0x08]),
            // val = imm8 << (bpos * 8)
            ("movb r1, 0x12000000", vec![0xb7, 0x10, 0x30, 0x12]),
            ("movb r5, 0xff", vec![0xb7, 0x12, 0x00, 0xff]),
            ("movb r1, 0x3400", vec![0xb7, 0x10, 0x10, 0x34]),
        ]);
        #[rustfmt::skip]
        assert_rejects(&asm, &[
            "addi r1, r2, 0x800",
            "sw r3, 0x800(r4)",
            // not d8 * 4 + 8: misaligned, then out of range
            "ldx r1, 0x9(r2)",
            "ldx r1, 0x208(r2)",
            // not one byte shifted by whole bytes
            "movb r1, 0x1234",
            "movb r1, 0x100000000",
        ]);
    }

    #[test]
    fn solver_rotated_immediates() {
        let asm = load("examples/solver.slaspec");
        // val = imm8 rotated right by 2 * rot; these have a single rot/imm8
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("orri r1, r2, 0xab000000", vec![0x8b, 0x60, 0x41, 0xab]),
            ("orri r1, r2, 0x3fc", vec![0x8b, 0x60, 0xf1, 0xff]),
            ("orri r1, r2, 0xfc000003", vec![0x8b, 0x60, 0x31, 0xff]),
            ("orri r1, r2, 0xff", vec![0x8b, 0x60, 0x01, 0xff]),
        ]);
        // 0x4 has four rot/imm8 pairs and 0x0 sixteen; whichever is found must decode back
        for input in ["orri r1, r2, 0x4", "orri r1, r2, 0x0"] {
            let bytes = assemble(&asm, input).unwrap();
            assert_eq!(disassemble(&asm, &bytes).as_deref(), Some(input));
        }
        // Set bits spread over more than 8 rotated bits
        assert_rejects(&asm, &["orri r1, r2, 0x101", "orri r1, r2, 0x1fe00"]);
    }

    #[test]
    fn solver_branch_offsets() {
        let asm = load("examples/solver.slaspec");
        // target = inst_start + sext(b12:b11:b10_5:b4_1:0)
        #[rustfmt::skip]
        assert_encodes_at(&asm, 0x1000, &[
            ("beq r1, r2, 0x1010", vec![0x63, 0x88, 0x20, 0x00]),
            ("beq r1, r2, 0xff0", vec![0xe3, 0x88, 0x20, 0xfe]),
            ("beq r1, r2, 0x1ffe", vec![0xe3, 0x8f, 0x20, 0x7e]),
            ("beq r1, r2, 0x0", vec![0x63, 0x80, 0x20, 0x80]),
        ]);
        // Out of range forwards (+4096), odd, and out of range backwards (-4098)
        #[rustfmt::skip]
        let rejects = [
            ("beq r1, r2, 0x2000", 0x1000),
            ("beq r1, r2, 0x1011", 0x1000),
            ("beq r1, r2, 0x1ffe", 0x3000),
        ];
        for (input, address) in rejects {
            assert_eq!(
                assemble_at(&asm, input, address),
                Err(AsmError::NoMatch),
                "{}",
                input
            );
        }
    }

    #[test]
    fn solver_program_with_labels() {
        let asm = load("examples/solver.slaspec");
        let source = "
            loop:   addi r1, r1, -0x1
                    beq r1, r0, done
                    beq r0, r0, loop
            done:   movb r2, 0xff00
        ";
        let program = asm.assemble_program(source, 0x1000).unwrap();
        #[rustfmt::skip]
        assert_eq!(program.bytes, vec![
            0x93, 0x80, 0xf0, 0xff, // addi r1, r1, -0x1
            0x63, 0x84, 0x00, 0x00, // beq r1, r0, 0x100c
            0xe3, 0x0c, 0x00, 0xfe, // beq r0, r0, 0x1000
            0x37, 0x11, 0x10, 0xff, // movb r2, 0xff00
        ]);
    }

    #[test]
    fn solver_roundtrip() {
        let asm = load("examples/solver.slaspec");
        assert_roundtrips(&asm, 4, 4000);
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

    #[test]
    fn belt_encodings() {
        let asm = load("examples/belt.slaspec");
        #[rustfmt::skip]
        assert_encodes_at(&asm, 0x1000, &[
            ("con 0x5", vec![0x04, 0x00, 0x00, 0x05]),
            ("con -0x1", vec![0x04, 0x03, 0xff, 0xff]),
            ("conw 0xedb88320", vec![0x08, 0x00, 0x00, 0x00, 0xed, 0xb8, 0x83, 0x20]),
            ("mul b2, b3", vec![0x14, 0x8c, 0x00, 0x00]),
            ("addi b1, -0x1", vec![0x30, 0x43, 0xff, 0xff]),
            ("mov b15", vec![0x53, 0xc0, 0x00, 0x00]),
            ("conform b2, b0", vec![0x54, 0x80, 0x00, 0x02]),
            ("conform b1, b2, b0", vec![0x54, 0x48, 0x00, 0x03]),
            ("conform b3, b0, b7, b2, b1", vec![0x54, 0xc1, 0xc8, 0x45]),
            ("st b1, b0", vec![0x48, 0x40, 0x00, 0x00]),
            ("br b0, 0x1014", vec![0x5c, 0x00, 0x00, 0x04]),
            ("brz b0, 0x1014", vec![0x60, 0x00, 0x00, 0x04]),
            ("jmp 0xffc", vec![0x64, 0x03, 0xff, 0xfe]),
            ("out b0", vec![0x68, 0x00, 0x00, 0x00]),
            ("nop", vec![0x00, 0x00, 0x00, 0x00]),
        ]);
        #[rustfmt::skip]
        assert_rejects(&asm, &[
            "add b16, b0",
            "con 0x20000",
            "mov b0, b1",
            "conform b0",
            "conform b0, b1, b2, b3, b4, b5",
            "br b0, 0x1012",
        ]);
    }

    #[test]
    fn belt_roundtrip() {
        let asm = load("examples/belt.slaspec");
        assert_roundtrips(&asm, 8, 400);
    }

    /// (input, context values, expected encoding)
    type ContextCase<'a> = (&'a str, &'a [(&'a str, i64)], Result<Vec<u8>, AsmError>);

    /// Assemble at 0 starting from the context `values`
    fn assemble_in(
        assembler: &InstructionAssembler,
        input: &str,
        values: &[(&str, i64)],
    ) -> Result<Vec<u8>, AsmError> {
        let context = Context::from_values(assembler, values).unwrap();
        let constraints = assembler.assemble_instruction_at(input, 0, &context, NO_LABELS)?;
        Ok(constraints
            .to_bytes()
            .expect("Constraints failed to produce bytes"))
    }

    fn assert_context_cases(assembler: &InstructionAssembler, tests: &[ContextCase]) {
        let mut failures = vec![];
        for (input, values, expected) in tests {
            let result = assemble_in(assembler, input, values);
            if result != *expected {
                failures.push(format!(
                    "{:?} in {:?}: expected {:?}, got {:?}",
                    input, values, expected, result
                ));
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    #[test]
    fn context_assemble() {
        let asm = load("examples/context.slaspec");
        let none: &[(&str, i64)] = &[];
        #[rustfmt::skip]
        let tests: &[ContextCase] = &[
            ("add r1, r2, 0x5", none, Ok(vec![0x05, 0x00, 0x12, 0x01])),
            ("sub r1, r2, 0x5", none, Err(AsmError::NoMatch)),
            ("sub r1, r2, 0x5", &[("mode", 1)], Ok(vec![0x05, 0x00, 0x12, 0x01])),
            ("add r1, r2, 0x5", &[("mode", 1)], Err(AsmError::NoMatch)),
            ("mode1", none, Ok(vec![0x00, 0x00, 0x00, 0x02])),
            ("pfx 0x2", none, Ok(vec![0x02, 0x00, 0x00, 0x04])),
            // A context operand has to be the context's value
            ("shl r1, r2, 0x2", &[("shift", 2)], Ok(vec![0x00, 0x00, 0x12, 0x05])),
            ("shl r1, r2, 0x2", none, Err(AsmError::NoMatch)),
            ("lo", none, Ok(vec![0x00, 0x00, 0x00, 0x07])),
            ("hi", none, Err(AsmError::NoMatch)),
            ("hi", &[("shift", 3)], Ok(vec![0x00, 0x00, 0x00, 0x07])),
            // mov's assignment picks the SRC its operand assembles as, whatever came in
            ("mov r1, r2", none, Ok(vec![0x00, 0x00, 0x12, 0x06])),
            ("mov r1, #0x7", none, Ok(vec![0x07, 0x00, 0x18, 0x06])),
            ("mov r1, r2", &[("width", 1)], Ok(vec![0x00, 0x00, 0x12, 0x06])),
            ("inc r6", none, Ok(vec![0x00, 0x00, 0x00, 0x08])),
            ("inc r7", &[("bank", 1)], Ok(vec![0x00, 0x00, 0x00, 0x08])),
            ("inc r6", &[("bank", 1)], Err(AsmError::NoMatch)),
        ];
        assert_context_cases(&asm, tests);
    }

    /// Program assembly follows the context from line to line like the emulator does
    #[test]
    fn context_program() {
        let asm = load("examples/context.slaspec");
        let source = "
            add r1, r1, 0x5
            mode1
            sub r1, r1, 0x2   // mode 1 from here on
            pfx 0x2
            shl r1, r1, 0x2   // shift only here
            shl r1, r1, 0x0
        ";
        let program = asm.assemble_program(source, 0x1000).unwrap();
        #[rustfmt::skip]
        assert_eq!(program.bytes, vec![
            0x05, 0x00, 0x11, 0x01,
            0x00, 0x00, 0x00, 0x02,
            0x02, 0x00, 0x11, 0x01,
            0x02, 0x00, 0x00, 0x04,
            0x00, 0x00, 0x11, 0x05,
            0x00, 0x00, 0x11, 0x05,
        ]);
        assert!(asm.assemble_program("sub r1, r1, 0x2", 0x1000).is_err());

        let mode1 = Context::from_values(&asm, &[("mode", 1)]).unwrap();
        let program = asm
            .assemble_program_in_context("sub r1, r1, 0x2", 0x1000, &mode1)
            .unwrap();
        assert_eq!(program.bytes, vec![0x02, 0x00, 0x11, 0x01]);
    }

    #[test]
    fn attach_names_and_values() {
        let asm = load("examples/attach.slaspec");
        #[rustfmt::skip]
        assert_encodes(&asm, &[
            ("b.eq 0x5", vec![0x10, 0x05]),
            ("b.le 0x5", vec![0x13, 0x05]),
            ("b.l 0x5", vec![0x16, 0x05]),
            ("shl r1, -0x2", vec![0x24, 0x80]),
            ("shl r3, -1", vec![0x2d, 0x00]),
            ("shl r0, 0x4", vec![0x23, 0x00]),
            ("shl r0, 0", vec![0x21, 0x80]),
            ("lds r0, 0x10", vec![0x30, 0x40]),
            ("lds r2, 8", vec![0x38, 0x30]),
        ]);
        // Indices that share a name or number are distinct encodings
        #[rustfmt::skip]
        assert_ambiguous(&asm, &[
            ("b.al 0x5", 2),
            ("lds r0, 0x1", 2),
        ]);
        #[rustfmt::skip]
        assert_rejects(&asm, &[
            "b.xx 0x5",
            "b.lee 0x5",
            "shl r0, 0x3",
            "shl r0, -0x3",
            "lds r0, 0x3",
            "lds r0, -0x1",
        ]);
    }

    #[test]
    fn attach_roundtrip() {
        let asm = load("examples/attach.slaspec");
        assert_roundtrips(&asm, 2, 300);
    }

    #[test]
    fn attach_names_on_context() {
        let asm = load("examples/attach.slaspec");
        let none: &[(&str, i64)] = &[];
        #[rustfmt::skip]
        let tests: &[ContextCase] = &[
            ("ld.b r1", none, Ok(vec![0x64, 0x00])),
            ("ld.w r1", &[("size", 2)], Ok(vec![0x64, 0x00])),
            ("ld.w r1", none, Err(AsmError::NoMatch)),
            ("ld.x r1", none, Err(AsmError::NoMatch)),
        ];
        assert_context_cases(&asm, tests);
    }
}
