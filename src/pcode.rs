//! Ghidra style p-code, lifted from disassembled instructions

use std::collections::{HashMap, HashSet};

use sleigh_rs::execution::{
    Assignment, AssignmentOp, AssignmentWrite, AssignmentWriteVariable, Binary, BlockId,
    BranchCall, CpuBranch, DynamicValueType, Export, Expr, ExprElement, ExprValue, ReferencedValue,
    Statement, Unary, UserCall, VariableId,
};
use sleigh_rs::token::TokenFieldAttach;
use sleigh_rs::{
    AttachNumberId, AttachVarnodeId, Endian, Sleigh, SpaceId, TableId, TokenFieldId, VarnodeId,
};

use crate::disassembler::DisassembledTable;
use crate::value::sign_extend;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VarnodeSpace {
    Space(SpaceId),
    Const,
    Unique,
}

/// `size` bytes at `offset` in `space`. A constant's value is its offset, kept with full 64-bit
/// precision so that values known at disassembly time keep their sign when resized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Varnode {
    pub space: VarnodeSpace,
    pub offset: u64,
    pub size: u32,
}

impl Varnode {
    pub fn constant(value: u64, size: u32) -> Self {
        Self {
            space: VarnodeSpace::Const,
            offset: value,
            size,
        }
    }

    pub fn is_const(&self) -> bool {
        self.space == VarnodeSpace::Const
    }
}

/// Mask with the low `size` bytes set
pub fn size_mask(size: u32) -> u64 {
    if size >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * size)) - 1
    }
}

/// Ghidra's p-code opcodes, as in its p-code reference (current Ghidra calls EXTRACT ZPULL and
/// adds SPULL, which are not here). The lifter emits a subset; the rest, like the decompiler's
/// analysis ops, are for p-code built elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCode {
    Copy,
    Load,
    Store,
    Branch,
    CBranch,
    BranchInd,
    Call,
    CallInd,
    Return,
    CallOther,
    IntEqual,
    IntNotEqual,
    IntSLess,
    IntSLessEqual,
    IntLess,
    IntLessEqual,
    IntZExt,
    IntSExt,
    IntAdd,
    IntSub,
    IntCarry,
    IntSCarry,
    IntSBorrow,
    Int2Comp,
    IntNegate,
    IntXor,
    IntAnd,
    IntOr,
    IntLeft,
    IntRight,
    IntSRight,
    IntMult,
    IntDiv,
    IntSDiv,
    IntRem,
    IntSRem,
    BoolNegate,
    BoolXor,
    BoolAnd,
    BoolOr,
    FloatEqual,
    FloatNotEqual,
    FloatLess,
    FloatLessEqual,
    FloatNan,
    FloatAdd,
    FloatDiv,
    FloatMult,
    FloatSub,
    FloatNeg,
    FloatAbs,
    FloatSqrt,
    FloatInt2Float,
    FloatFloat2Float,
    FloatTrunc,
    FloatCeil,
    FloatFloor,
    FloatRound,
    MultiEqual,
    Indirect,
    Piece,
    Subpiece,
    Cast,
    PtrAdd,
    PtrSub,
    SegmentOp,
    CPoolRef,
    New,
    Insert,
    Extract,
    Popcount,
    Lzcount,
}

impl OpCode {
    /// Ghidra's name for the opcode
    pub fn name(&self) -> &'static str {
        match self {
            OpCode::Copy => "COPY",
            OpCode::Load => "LOAD",
            OpCode::Store => "STORE",
            OpCode::Branch => "BRANCH",
            OpCode::CBranch => "CBRANCH",
            OpCode::BranchInd => "BRANCHIND",
            OpCode::Call => "CALL",
            OpCode::CallInd => "CALLIND",
            OpCode::Return => "RETURN",
            OpCode::CallOther => "CALLOTHER",
            OpCode::IntEqual => "INT_EQUAL",
            OpCode::IntNotEqual => "INT_NOTEQUAL",
            OpCode::IntSLess => "INT_SLESS",
            OpCode::IntSLessEqual => "INT_SLESSEQUAL",
            OpCode::IntLess => "INT_LESS",
            OpCode::IntLessEqual => "INT_LESSEQUAL",
            OpCode::IntZExt => "INT_ZEXT",
            OpCode::IntSExt => "INT_SEXT",
            OpCode::IntAdd => "INT_ADD",
            OpCode::IntSub => "INT_SUB",
            OpCode::IntCarry => "INT_CARRY",
            OpCode::IntSCarry => "INT_SCARRY",
            OpCode::IntSBorrow => "INT_SBORROW",
            OpCode::Int2Comp => "INT_2COMP",
            OpCode::IntNegate => "INT_NEGATE",
            OpCode::IntXor => "INT_XOR",
            OpCode::IntAnd => "INT_AND",
            OpCode::IntOr => "INT_OR",
            OpCode::IntLeft => "INT_LEFT",
            OpCode::IntRight => "INT_RIGHT",
            OpCode::IntSRight => "INT_SRIGHT",
            OpCode::IntMult => "INT_MULT",
            OpCode::IntDiv => "INT_DIV",
            OpCode::IntSDiv => "INT_SDIV",
            OpCode::IntRem => "INT_REM",
            OpCode::IntSRem => "INT_SREM",
            OpCode::BoolNegate => "BOOL_NEGATE",
            OpCode::BoolXor => "BOOL_XOR",
            OpCode::BoolAnd => "BOOL_AND",
            OpCode::BoolOr => "BOOL_OR",
            OpCode::FloatEqual => "FLOAT_EQUAL",
            OpCode::FloatNotEqual => "FLOAT_NOTEQUAL",
            OpCode::FloatLess => "FLOAT_LESS",
            OpCode::FloatLessEqual => "FLOAT_LESSEQUAL",
            OpCode::FloatNan => "FLOAT_NAN",
            OpCode::FloatAdd => "FLOAT_ADD",
            OpCode::FloatDiv => "FLOAT_DIV",
            OpCode::FloatMult => "FLOAT_MULT",
            OpCode::FloatSub => "FLOAT_SUB",
            OpCode::FloatNeg => "FLOAT_NEG",
            OpCode::FloatAbs => "FLOAT_ABS",
            OpCode::FloatSqrt => "FLOAT_SQRT",
            OpCode::FloatInt2Float => "INT2FLOAT",
            OpCode::FloatFloat2Float => "FLOAT2FLOAT",
            OpCode::FloatTrunc => "TRUNC",
            OpCode::FloatCeil => "CEIL",
            OpCode::FloatFloor => "FLOOR",
            OpCode::FloatRound => "ROUND",
            OpCode::MultiEqual => "MULTIEQUAL",
            OpCode::Indirect => "INDIRECT",
            OpCode::Piece => "PIECE",
            OpCode::Subpiece => "SUBPIECE",
            OpCode::Cast => "CAST",
            OpCode::PtrAdd => "PTRADD",
            OpCode::PtrSub => "PTRSUB",
            OpCode::SegmentOp => "SEGMENTOP",
            OpCode::CPoolRef => "CPOOLREF",
            OpCode::New => "NEW",
            OpCode::Insert => "INSERT",
            OpCode::Extract => "EXTRACT",
            OpCode::Popcount => "POPCOUNT",
            OpCode::Lzcount => "LZCOUNT",
        }
    }
}

/// One p-code operation. Following Ghidra:
/// - `LOAD`/`STORE` take the space as a constant in input 0
/// - a branch target in a space is an address; a constant target is an op offset relative to
///   the branch, used for jumps inside the instruction
/// - `CALLOTHER` takes the user op index as a constant in input 0
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcodeOp {
    pub opcode: OpCode,
    pub output: Option<Varnode>,
    pub inputs: Vec<Varnode>,
}

/// Where a direct branch op goes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchTarget {
    /// Another op of the same instruction, this many ops on
    Relative(i64),
    /// The instruction at an address
    Address(VarnodeSpace, u64),
}

impl PcodeOp {
    pub fn display<'a>(&'a self, sleigh: &'a Sleigh) -> DisplayPcodeOp<'a> {
        DisplayPcodeOp { op: self, sleigh }
    }

    /// The target of a `BRANCH`, `CBRANCH` or `CALL`: a constant is relative to the op,
    /// anything else names an address
    pub fn branch_target(&self) -> Option<BranchTarget> {
        let target = match self.opcode {
            OpCode::Branch | OpCode::CBranch | OpCode::Call => self.inputs.first()?,
            _ => return None,
        };
        Some(match target.space {
            VarnodeSpace::Const => BranchTarget::Relative(target.offset as i64),
            space => BranchTarget::Address(space, target.offset),
        })
    }
}

pub fn space_name(sleigh: &Sleigh, space: SpaceId) -> &str {
    &sleigh.space(space).name
}

pub struct DisplayVarnode<'a> {
    pub varnode: &'a Varnode,
    pub sleigh: &'a Sleigh,
}

impl std::fmt::Display for DisplayVarnode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let varnode = self.varnode;
        match varnode.space {
            VarnodeSpace::Const => write!(
                f,
                "{:#x}:{}",
                varnode.offset & size_mask(varnode.size),
                varnode.size
            ),
            VarnodeSpace::Unique => write!(f, "$U{:x}:{}", varnode.offset, varnode.size),
            VarnodeSpace::Space(space) => {
                let register = self.sleigh.varnodes().iter().find(|register| {
                    register.space == space
                        && register.address == varnode.offset
                        && register.len_bytes.get() == varnode.size as u64
                });
                match register {
                    Some(register) => write!(f, "{}", register.name()),
                    None => write!(
                        f,
                        "{}[{:#x}]:{}",
                        space_name(self.sleigh, space),
                        varnode.offset,
                        varnode.size
                    ),
                }
            }
        }
    }
}

pub struct DisplayPcodeOp<'a> {
    op: &'a PcodeOp,
    sleigh: &'a Sleigh,
}

impl std::fmt::Display for DisplayPcodeOp<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = self.op;
        if let Some(output) = &op.output {
            let output = DisplayVarnode {
                varnode: output,
                sleigh: self.sleigh,
            };
            write!(f, "{} = ", output)?;
        }
        write!(f, "{}", op.opcode.name())?;
        for (index, input) in op.inputs.iter().enumerate() {
            write!(f, "{}", if index == 0 { " " } else { ", " })?;
            match (op.opcode, index) {
                (OpCode::Load | OpCode::Store, 0) => write!(
                    f,
                    "{}",
                    space_name(self.sleigh, SpaceId(input.offset as usize))
                )?,
                (OpCode::CallOther, 0) => write!(
                    f,
                    "{:?}",
                    self.sleigh
                        .user_function(sleigh_rs::UserFunctionId(input.offset as usize))
                        .name()
                )?,
                (_, 0) if matches!(op.branch_target(), Some(BranchTarget::Relative(_))) => {
                    write!(f, "{:+}", input.offset as i64)?
                }
                _ => write!(
                    f,
                    "{}",
                    DisplayVarnode {
                        varnode: input,
                        sleigh: self.sleigh,
                    }
                )?,
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiftError {
    /// A construct the lifter does not handle. The lifter handles everything sleigh-rs
    /// produces now, the variant is kept for API stability.
    Unsupported(String),
    /// The disassembled instruction is inconsistent with its constructor
    Invalid(String),
}

impl std::fmt::Display for LiftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiftError::Unsupported(what) => write!(f, "unsupported: {}", what),
            LiftError::Invalid(what) => write!(f, "invalid: {}", what),
        }
    }
}

impl std::error::Error for LiftError {}

type LiftResult<T> = Result<T, LiftError>;

/// The bits of a `bits` wide value a partial write op selects, all of them without one
fn assignment_bits(op: &Option<AssignmentOp>, bits: u64) -> LiftResult<std::ops::Range<u64>> {
    let range = match op {
        None => 0..bits,
        Some(AssignmentOp::TakeLsb(bytes)) => 0..8 * bytes.get(),
        Some(AssignmentOp::TrunkLsb(bytes)) => 8 * bytes..bits,
        Some(AssignmentOp::BitRange(range)) => range.clone(),
    };
    if range.start >= range.end || range.end > bits {
        return Err(LiftError::Invalid(format!(
            "{:?} is outside of {} bits",
            op, bits
        )));
    }
    Ok(range)
}

fn bits_to_bytes(bits: u64) -> u32 {
    bits.div_ceil(8).max(1) as u32
}

/// Where a table's export lives: a varnode, or memory at a dynamic address
#[derive(Debug, Clone, Copy)]
enum Handle {
    Direct(Varnode),
    Pointer {
        space: SpaceId,
        addr: Varnode,
        size: u32,
    },
}

impl Handle {
    /// Where memory at a constant address is, the address cut to the spec's address size
    fn fixed_address(&self, sleigh: &Sleigh) -> Option<(SpaceId, u64)> {
        match self {
            Handle::Pointer { space, addr, .. } if addr.is_const() => Some((
                *space,
                addr.offset & size_mask(sleigh.addr_bytes().get() as u32),
            )),
            _ => None,
        }
    }
}

/// Lifting state of one constructor
struct Scope<'t> {
    table: &'t DisassembledTable<'t>,
    /// `inst_next` in the semantics, past the delay slot of the instruction
    inst_next: u64,
    locals: HashMap<VariableId, Varnode>,
    built: HashSet<TableId>,
    exports: HashMap<TableId, Handle>,
    export: Option<Handle>,
    block_labels: HashMap<BlockId, usize>,
}

impl<'t> Scope<'t> {
    fn new(table: &'t DisassembledTable<'t>, inst_next: u64) -> Self {
        Self {
            table,
            inst_next,
            locals: HashMap::new(),
            built: HashSet::new(),
            exports: HashMap::new(),
            export: None,
            block_labels: HashMap::new(),
        }
    }
}

/// Lift the semantics of a disassembled instruction to p-code. `delayslot` runs the p-code of
/// `delay_slots`, the instructions that follow, and `inst_next` is the address after them.
pub fn lift<'s>(
    table: &'s DisassembledTable<'s>,
    inst_next: u64,
    delay_slots: &'s [DisassembledTable<'s>],
) -> Result<Vec<PcodeOp>, LiftError> {
    let mut lifter = Lifter::new(table.disassembler.sleigh, delay_slots);
    lifter.lift_table(table, inst_next)?;
    lifter.finish()
}

/// The address a table exports, as `globalset` takes it: memory at a constant address in the
/// code space, or a constant. Ghidra's C++ libsla moves commits to a constant into the code
/// space, while those to other spaces, a register say, have no effect on decoding. Ghidra's
/// Java parser differs: it takes the export's offset as a code address whatever its space.
///
/// The table's whole semantics are lifted to find its export, so a table whose semantics
/// cannot be lifted has no address either.
pub fn export_address(table: &DisassembledTable) -> Result<u64, LiftError> {
    let sleigh = table.disassembler.sleigh;
    let export = Lifter::new(sleigh, &[]).lift_table(table, table.inst_next)?;
    match export {
        Some(Handle::Direct(varnode)) if varnode.is_const() => {
            Ok(varnode.offset & size_mask(sleigh.addr_bytes().get() as u32))
        }
        Some(handle) => match handle.fixed_address(sleigh) {
            Some((space, address)) if space == sleigh.default_space() => Ok(address),
            _ => Err(LiftError::Invalid(format!(
                "table {} exports no address in the code space",
                table.table.name()
            ))),
        },
        None => Err(LiftError::Invalid(format!(
            "table {} has no export",
            table.table.name()
        ))),
    }
}

struct Lifter<'s> {
    sleigh: &'s Sleigh,
    delay_slots: &'s [DisassembledTable<'s>],
    ops: Vec<PcodeOp>,
    next_unique: u64,
    /// Op index of each label, once placed
    labels: Vec<Option<usize>>,
    /// Branch ops whose target is a label
    fixups: Vec<(usize, usize)>,
}

impl<'s> Lifter<'s> {
    fn new(sleigh: &'s Sleigh, delay_slots: &'s [DisassembledTable<'s>]) -> Self {
        Self {
            sleigh,
            delay_slots,
            ops: vec![],
            next_unique: 0,
            labels: vec![],
            fixups: vec![],
        }
    }

    fn finish(mut self) -> LiftResult<Vec<PcodeOp>> {
        for (op_index, label) in self.fixups.iter() {
            let target = self.labels[*label]
                .ok_or_else(|| LiftError::Invalid("branch to an unplaced label".to_string()))?;
            let relative = target as i64 - *op_index as i64;
            self.ops[*op_index].inputs[0] = Varnode::constant(relative as u64, 4);
        }
        Ok(self.ops)
    }

    fn emit(&mut self, opcode: OpCode, output: Option<Varnode>, inputs: Vec<Varnode>) {
        self.ops.push(PcodeOp {
            opcode,
            output,
            inputs,
        });
    }

    fn unique(&mut self, size: u32) -> Varnode {
        let varnode = Varnode {
            space: VarnodeSpace::Unique,
            offset: self.next_unique,
            size,
        };
        self.next_unique += (size as u64).max(8);
        varnode
    }

    /// `opcode` with a fresh output of `size` bytes
    fn op(&mut self, opcode: OpCode, size: u32, inputs: Vec<Varnode>) -> Varnode {
        let output = self.unique(size);
        self.emit(opcode, Some(output), inputs);
        output
    }

    fn new_label(&mut self) -> usize {
        self.labels.push(None);
        self.labels.len() - 1
    }

    fn place_label(&mut self, label: usize) {
        self.labels[label] = Some(self.ops.len());
    }

    /// A branch to a label; `cond` makes it conditional
    fn branch_to_label(&mut self, label: usize, cond: Option<Varnode>) {
        self.fixups.push((self.ops.len(), label));
        let target = Varnode::constant(label as u64, 4);
        match cond {
            Some(cond) => self.emit(OpCode::CBranch, None, vec![target, cond]),
            None => self.emit(OpCode::Branch, None, vec![target]),
        }
    }

    fn register(&self, varnode_id: VarnodeId) -> Varnode {
        let varnode = self.sleigh.varnode(varnode_id);
        Varnode {
            space: VarnodeSpace::Space(varnode.space),
            offset: varnode.address,
            size: varnode.len_bytes.get() as u32,
        }
    }

    fn token_field(&self, scope: &Scope, id: TokenFieldId) -> LiftResult<i64> {
        scope
            .table
            .token_fields
            .get(&id)
            .copied()
            .ok_or_else(|| LiftError::Invalid(format!("token field {} not decoded", id.0)))
    }

    /// The value of a token field or context variable that indexes an attach
    fn attach_index(&self, scope: &Scope, value: DynamicValueType) -> LiftResult<i64> {
        Ok(match value {
            DynamicValueType::TokenField(token_field_id) => {
                self.token_field(scope, token_field_id)?
            }
            DynamicValueType::Context(context_id) => {
                scope.table.context.get(self.sleigh, context_id)
            }
        })
    }

    /// The register an attached token field or context variable selects
    fn attached(
        &self,
        scope: &Scope,
        attach_id: AttachVarnodeId,
        value: DynamicValueType,
    ) -> LiftResult<Varnode> {
        let index = self.attach_index(scope, value)?;
        let varnode_id = self
            .sleigh
            .attach_varnode(attach_id)
            .find_value(index as usize)
            .ok_or_else(|| LiftError::Invalid(format!("no register attached to {}", index)))?;
        Ok(self.register(varnode_id))
    }

    /// The number an attached token field selects
    fn attached_number(
        &self,
        scope: &Scope,
        attach_id: AttachNumberId,
        value: DynamicValueType,
    ) -> LiftResult<u64> {
        let index = self.attach_index(scope, value)?;
        let number = self
            .sleigh
            .attach_number(attach_id)
            .find_value(index as usize)
            .ok_or_else(|| LiftError::Invalid(format!("no number attached to {}", index)))?;
        Ok(number.signed_super() as u64)
    }

    /// `varnode` as `size` bytes: constants are resized directly, others zero extended or
    /// truncated
    fn resize(&mut self, varnode: Varnode, size: u32) -> Varnode {
        if varnode.size == size {
            varnode
        } else if varnode.is_const() {
            Varnode::constant(varnode.offset, size)
        } else if varnode.size < size {
            self.op(OpCode::IntZExt, size, vec![varnode])
        } else {
            self.op(
                OpCode::Subpiece,
                size,
                vec![varnode, Varnode::constant(0, 4)],
            )
        }
    }

    /// The bytes of `varnode` from byte `offset` (counted from the least significant byte)
    fn sub_varnode(&self, varnode: Varnode, offset: u32, size: u32) -> LiftResult<Varnode> {
        if varnode.is_const() || offset + size > varnode.size {
            return Err(LiftError::Invalid(format!(
                "cannot take {} bytes at {} of {:?}",
                size, offset, varnode
            )));
        }
        let start = match self.sleigh.endian() {
            Endian::Big => varnode.offset + (varnode.size - offset - size) as u64,
            Endian::Little => varnode.offset + offset as u64,
        };
        Ok(Varnode {
            offset: start,
            size,
            ..varnode
        })
    }

    /// `addr` advanced by `offset` bytes
    fn offset_addr(&mut self, addr: Varnode, offset: u32) -> Varnode {
        if offset == 0 {
            addr
        } else if addr.is_const() {
            Varnode::constant(addr.offset.wrapping_add(offset as u64), addr.size)
        } else {
            let offset = Varnode::constant(offset as u64, addr.size);
            self.op(OpCode::IntAdd, addr.size, vec![addr, offset])
        }
    }

    fn lift_table(
        &mut self,
        table: &DisassembledTable,
        inst_next: u64,
    ) -> LiftResult<Option<Handle>> {
        let Some(execution) = &table.constructor.execution else {
            return Ok(None);
        };
        let mut scope = Scope::new(table, inst_next);

        // Subtables without an explicit `build` are built before the constructor's semantics
        let explicit_builds = execution
            .blocks()
            .iter()
            .flat_map(|block| block.statements.iter())
            .filter_map(|stmt| match stmt {
                Statement::Build(build) => Some(build.table),
                _ => None,
            })
            .collect::<HashSet<_>>();
        for pattern_block in table.constructor.pattern.blocks() {
            for produced_table in pattern_block.tables() {
                if !explicit_builds.contains(&produced_table.table)
                    && table.tables.contains_key(&produced_table.table)
                {
                    self.build(&mut scope, produced_table.table)?;
                }
            }
        }

        // The entry block first, then the others in order
        let order = std::iter::once(execution.entry_block)
            .chain(
                (0..execution.blocks().len())
                    .map(BlockId)
                    .filter(|id| *id != execution.entry_block),
            )
            .collect::<Vec<_>>();
        for block_id in order.iter() {
            let label = self.new_label();
            scope.block_labels.insert(*block_id, label);
        }
        let end = self.new_label();

        for (index, block_id) in order.iter().enumerate() {
            self.place_label(scope.block_labels[block_id]);
            let block = execution.block(*block_id);
            for statement in block.statements.iter() {
                self.statement(&mut scope, statement)?;
            }
            let fallthrough = order.get(index + 1).copied();
            if block.next != fallthrough {
                let target = match block.next {
                    Some(next) => scope.block_labels[&next],
                    None => end,
                };
                if fallthrough.is_some() {
                    self.branch_to_label(target, None);
                }
            }
        }
        self.place_label(end);
        Ok(scope.export)
    }

    fn build(&mut self, scope: &mut Scope, table_id: TableId) -> LiftResult<()> {
        if !scope.built.insert(table_id) {
            return Ok(());
        }
        let table = scope.table.tables.get(&table_id).ok_or_else(|| {
            LiftError::Invalid(format!(
                "build of table {} that is not an operand",
                table_id.0
            ))
        })?;
        if let Some(handle) = self.lift_table(table, scope.inst_next)? {
            scope.exports.insert(table_id, handle);
        }
        Ok(())
    }

    fn table_export(&self, scope: &Scope, table_id: TableId) -> LiftResult<Handle> {
        scope.exports.get(&table_id).copied().ok_or_else(|| {
            LiftError::Invalid(format!(
                "table {} has no export or is not built",
                table_id.0
            ))
        })
    }

    fn read_handle(&mut self, handle: Handle) -> Varnode {
        match handle {
            Handle::Direct(varnode) => varnode,
            Handle::Pointer { space, addr, size } => self.op(
                OpCode::Load,
                size,
                vec![Varnode::constant(space.0 as u64, 4), addr],
            ),
        }
    }

    fn statement(&mut self, scope: &mut Scope, statement: &Statement) -> LiftResult<()> {
        match statement {
            Statement::Delayslot(_) => self.delay_slot(),
            Statement::Export(export) => {
                scope.export = Some(self.export(scope, export)?);
                Ok(())
            }
            Statement::CpuBranch(cpu_branch) => self.cpu_branch(scope, cpu_branch),
            Statement::LocalGoto(local_goto) => {
                let label = *scope.block_labels.get(&local_goto.dst).ok_or_else(|| {
                    LiftError::Invalid(format!("goto unknown block {}", local_goto.dst.0))
                })?;
                let cond = match &local_goto.cond {
                    Some(cond) => Some(self.condition(scope, cond)?),
                    None => None,
                };
                self.branch_to_label(label, cond);
                Ok(())
            }
            Statement::UserCall(user_call) => {
                self.user_call(scope, user_call, None)?;
                Ok(())
            }
            Statement::Build(build) => self.build(scope, build.table),
            Statement::Declare(variable_id) => {
                self.local(scope, *variable_id);
                Ok(())
            }
            Statement::Assignment(assignment) => self.assignment(scope, assignment),
        }
    }

    /// The p-code of the delay slot instructions, each with its own `inst_next`. They have no
    /// delay slots of their own.
    fn delay_slot(&mut self) -> LiftResult<()> {
        let delay_slots = std::mem::take(&mut self.delay_slots);
        if delay_slots.is_empty() {
            return Err(LiftError::Invalid(
                "no instructions for the delay slot".to_string(),
            ));
        }
        for slot in delay_slots.iter() {
            self.lift_table(slot, slot.inst_next)?;
        }
        self.delay_slots = delay_slots;
        Ok(())
    }

    fn local(&mut self, scope: &mut Scope, variable_id: VariableId) -> Varnode {
        if let Some(varnode) = scope.locals.get(&variable_id) {
            return *varnode;
        }
        let execution = scope.table.constructor.execution.as_ref().unwrap();
        let size = bits_to_bytes(execution.variable(variable_id).len_bits.get());
        let varnode = self.unique(size);
        scope.locals.insert(variable_id, varnode);
        varnode
    }

    fn export(&mut self, scope: &mut Scope, export: &Export) -> LiftResult<Handle> {
        Ok(match export {
            Export::Value(expr) => Handle::Direct(self.expr(scope, expr, None)?),
            Export::Reference { addr, memory } => Handle::Pointer {
                space: memory.space,
                addr: self.expr(scope, addr, None)?,
                size: memory.len_bytes.get() as u32,
            },
            Export::AttachVarnode {
                attach_value,
                attach_id,
                ..
            } => Handle::Direct(self.attached(scope, *attach_id, *attach_value)?),
            Export::Table { table_id, .. } => self.table_export(scope, *table_id)?,
        })
    }

    /// A 1 byte condition
    fn condition(&mut self, scope: &mut Scope, expr: &Expr) -> LiftResult<Varnode> {
        let cond = self.expr(scope, expr, Some(1))?;
        if cond.size == 1 {
            return Ok(cond);
        }
        let zero = Varnode::constant(0, cond.size);
        Ok(self.op(OpCode::IntNotEqual, 1, vec![cond, zero]))
    }

    fn cpu_branch(&mut self, scope: &mut Scope, cpu_branch: &CpuBranch) -> LiftResult<()> {
        let addr_size = self.sleigh.addr_bytes().get() as u32;
        // A direct branch goes to the address the destination names, an indirect one to the
        // value it holds
        let (target, direct) = if cpu_branch.direct {
            match &cpu_branch.dst {
                Expr::Value(ExprElement::Value {
                    value: ExprValue::Table(table_id),
                    ..
                }) => {
                    let handle = self.table_export(scope, *table_id)?;
                    match (handle.fixed_address(self.sleigh), handle) {
                        (Some((space, offset)), _) => {
                            let target = Varnode {
                                space: VarnodeSpace::Space(space),
                                offset,
                                size: addr_size,
                            };
                            (target, true)
                        }
                        // The address of memory at a dynamic address is that address, so it is
                        // branched to indirectly. Ghidra loads through the pointer and branches
                        // to the temporary instead, which likewise gives no flow target
                        (None, Handle::Pointer { addr, .. }) => (addr, false),
                        (None, Handle::Direct(varnode)) => (varnode, true),
                    }
                }
                dst => (self.expr(scope, dst, Some(addr_size))?, true),
            }
        } else {
            (self.expr(scope, &cpu_branch.dst, Some(addr_size))?, false)
        };
        let target = if target.is_const() {
            Varnode {
                space: VarnodeSpace::Space(self.sleigh.default_space()),
                offset: target.offset & size_mask(addr_size),
                size: addr_size,
            }
        } else {
            target
        };
        let is_address = matches!(target.space, VarnodeSpace::Space(_)) && direct;

        let opcode = match (cpu_branch.call, is_address) {
            (BranchCall::Goto, true) => OpCode::Branch,
            (BranchCall::Goto, false) => OpCode::BranchInd,
            (BranchCall::Call, true) => OpCode::Call,
            (BranchCall::Call, false) => OpCode::CallInd,
            (BranchCall::Return, _) => OpCode::Return,
        };
        match &cpu_branch.cond {
            None => self.emit(opcode, None, vec![target]),
            Some(cond) => {
                let cond = self.condition(scope, cond)?;
                if opcode == OpCode::Branch {
                    self.emit(OpCode::CBranch, None, vec![target, cond]);
                } else {
                    // Skip the branch unless the condition holds
                    let skip = self.new_label();
                    let not_cond = self.op(OpCode::BoolNegate, 1, vec![cond]);
                    self.branch_to_label(skip, Some(not_cond));
                    self.emit(opcode, None, vec![target]);
                    self.place_label(skip);
                }
            }
        }
        Ok(())
    }

    fn user_call(
        &mut self,
        scope: &mut Scope,
        user_call: &UserCall,
        output_size: Option<u32>,
    ) -> LiftResult<Option<Varnode>> {
        let mut inputs = vec![Varnode::constant(user_call.function.0 as u64, 4)];
        for param in user_call.params.iter() {
            inputs.push(self.expr(scope, param, None)?);
        }
        let output = output_size.map(|size| self.unique(size));
        self.emit(OpCode::CallOther, output, inputs);
        Ok(output)
    }

    /// Where an assignment writes, before any partial write op is applied
    fn write_target(
        &mut self,
        scope: &mut Scope,
        value: &AssignmentWriteVariable,
    ) -> LiftResult<Varnode> {
        Ok(match value {
            AssignmentWriteVariable::Varnode(varnode_id) => self.register(*varnode_id),
            AssignmentWriteVariable::DynVarnode {
                value_id,
                attach_id,
            } => self.attached(scope, *attach_id, *value_id)?,
            AssignmentWriteVariable::Variable(variable_id) => self.local(scope, *variable_id),
            AssignmentWriteVariable::Bitrange(_) => unreachable!("handled by the caller"),
        })
    }

    /// Bits `range` of `input`, as a value of `size` bytes
    fn read_bits(&mut self, input: Varnode, range: std::ops::Range<u64>, size: u32) -> Varnode {
        let len = range.end - range.start;
        let mask = if len >= 64 { u64::MAX } else { (1 << len) - 1 };
        let shifted = self.op(
            OpCode::IntRight,
            input.size,
            vec![input, Varnode::constant(range.start, 4)],
        );
        let masked = self.op(
            OpCode::IntAnd,
            input.size,
            vec![shifted, Varnode::constant(mask, input.size)],
        );
        self.resize(masked, size)
    }

    /// Write bits `range` of `target`, as `assignment_bits` checked them, with the low bits
    /// of `value`
    fn write_bits(&mut self, target: Varnode, range: std::ops::Range<u64>, value: Varnode) {
        let size = target.size;
        let len = range.end - range.start;
        let field_mask = if len >= 64 { u64::MAX } else { (1 << len) - 1 };
        let mask = (field_mask << range.start) & size_mask(size);
        let value = self.resize(value, size);
        let shifted = self.op(
            OpCode::IntLeft,
            size,
            vec![value, Varnode::constant(range.start, 4)],
        );
        let new_bits = self.op(
            OpCode::IntAnd,
            size,
            vec![shifted, Varnode::constant(mask, size)],
        );
        let old_bits = self.op(
            OpCode::IntAnd,
            size,
            vec![target, Varnode::constant(!mask & size_mask(size), size)],
        );
        self.emit(OpCode::IntOr, Some(target), vec![old_bits, new_bits]);
    }

    /// Write `value` to `target`, through a partial write op if any. `first_op` is where the
    /// ops computing `value` start: a temporary they produced last is replaced by the target.
    fn write(
        &mut self,
        target: Varnode,
        op: &Option<AssignmentOp>,
        value: Varnode,
        first_op: usize,
    ) -> LiftResult<()> {
        let bits = assignment_bits(op, 8 * target.size as u64)?;
        if let Some(AssignmentOp::BitRange(_)) = op {
            self.write_bits(target, bits, value);
            return Ok(());
        }
        let (offset, len) = (
            (bits.start / 8) as u32,
            ((bits.end - bits.start) / 8) as u32,
        );
        let target = self.sub_varnode(target, offset, len)?;
        let value = self.resize(value, target.size);
        let computed_here = self.ops.len() > first_op;
        if let Some(last) = self.ops.last_mut() {
            if computed_here && last.output == Some(value) && value.space == VarnodeSpace::Unique {
                last.output = Some(target);
                return Ok(());
            }
        }
        self.emit(OpCode::Copy, Some(target), vec![value]);
        Ok(())
    }

    fn assignment(&mut self, scope: &mut Scope, assignment: &Assignment) -> LiftResult<()> {
        match &assignment.var {
            AssignmentWrite::Variable {
                value: AssignmentWriteVariable::Bitrange(bitrange_id),
                op,
            } => {
                let bitrange = self.sleigh.bitrange(*bitrange_id);
                let target = self.register(bitrange.varnode);
                // A partial write op selects bits inside the bitrange
                let start = bitrange.bits.start();
                let range = assignment_bits(op, bitrange.bits.len().get())?;
                let value = self.expr(scope, &assignment.right, None)?;
                self.write_bits(target, start + range.start..start + range.end, value);
                Ok(())
            }
            AssignmentWrite::Variable { value, op } => {
                let target = self.write_target(scope, value)?;
                let first_op = self.ops.len();
                let value = self.expr(scope, &assignment.right, Some(target.size))?;
                self.write(target, op, value, first_op)
            }
            AssignmentWrite::Memory { mem, addr } => {
                let size = mem.len_bytes.get() as u32;
                let addr = self.expr(scope, addr, None)?;
                let value = self.expr(scope, &assignment.right, Some(size))?;
                let value = self.resize(value, size);
                let space = Varnode::constant(mem.space.0 as u64, 4);
                self.emit(OpCode::Store, None, vec![space, addr, value]);
                Ok(())
            }
            AssignmentWrite::TableExport { table_id, op, .. } => {
                match self.table_export(scope, *table_id)? {
                    Handle::Direct(target) => {
                        let first_op = self.ops.len();
                        let value = self.expr(scope, &assignment.right, Some(target.size))?;
                        self.write(target, op, value, first_op)
                    }
                    Handle::Pointer { space, addr, size } => {
                        let space = Varnode::constant(space.0 as u64, 4);
                        let bits = assignment_bits(op, 8 * size as u64)?;
                        if let Some(AssignmentOp::BitRange(_)) = op {
                            // Load the old value, modify the bits, store it back
                            let value = self.expr(scope, &assignment.right, Some(size))?;
                            let old = self.op(OpCode::Load, size, vec![space, addr]);
                            self.write_bits(old, bits, value);
                            self.emit(OpCode::Store, None, vec![space, addr, old]);
                            return Ok(());
                        }
                        // As in Ghidra, other ops store just the bytes they select
                        let (offset, len) = (
                            (bits.start / 8) as u32,
                            ((bits.end - bits.start) / 8) as u32,
                        );
                        let value = self.expr(scope, &assignment.right, Some(len))?;
                        let value = self.resize(value, len);
                        let offset = match self.sleigh.endian() {
                            Endian::Big => size - offset - len,
                            Endian::Little => offset,
                        };
                        let addr = self.offset_addr(addr, offset);
                        self.emit(OpCode::Store, None, vec![space, addr, value]);
                        Ok(())
                    }
                }
            }
        }
    }

    fn execution_len_bytes(&self, scope: &Scope, expr: &Expr) -> u32 {
        let execution = scope.table.constructor.execution.as_ref().unwrap();
        bits_to_bytes(expr.len_bits(self.sleigh, execution).get())
    }

    /// Lift an expression to a varnode. `size_hint` sizes results sleigh-rs does not size,
    /// such as user op calls.
    fn expr(
        &mut self,
        scope: &mut Scope,
        expr: &Expr,
        size_hint: Option<u32>,
    ) -> LiftResult<Varnode> {
        match expr {
            Expr::Value(ExprElement::Value { value, .. }) => self.expr_value(scope, value),
            Expr::Value(ExprElement::UserCall(user_call)) => {
                let size = size_hint.unwrap_or(self.sleigh.addr_bytes().get() as u32);
                Ok(self.user_call(scope, user_call, Some(size))?.unwrap())
            }
            Expr::Value(ExprElement::Reference(reference)) => {
                let size = bits_to_bytes(reference.len_bits.get());
                let address = match &reference.value {
                    // As in Ghidra, the offset of the varnode the field stands for: the
                    // address of an attached register, an attached number, or else the
                    // field's value
                    ReferencedValue::TokenField(field) => {
                        let value = DynamicValueType::TokenField(field.id);
                        let offset = match self.sleigh.token_field(field.id).attach {
                            TokenFieldAttach::Varnode(attach_id) => {
                                self.attached(scope, attach_id, value)?.offset
                            }
                            TokenFieldAttach::Number(_, attach_id) => {
                                self.attached_number(scope, attach_id, value)?
                            }
                            TokenFieldAttach::NoAttach(_) | TokenFieldAttach::Literal(_) => {
                                self.token_field(scope, field.id)? as u64
                            }
                        };
                        Varnode::constant(offset, size)
                    }
                    ReferencedValue::InstStart(_) => {
                        Varnode::constant(scope.table.inst_start, size)
                    }
                    ReferencedValue::InstNext(_) => Varnode::constant(scope.inst_next, size),
                    ReferencedValue::Table(table) => match self.table_export(scope, table.id)? {
                        Handle::Pointer { addr, .. } => self.resize(addr, size),
                        // A constant's offset is its value
                        Handle::Direct(varnode) => Varnode::constant(varnode.offset, size),
                    },
                };
                // The offset is truncated to the size of the reference
                if address.is_const() {
                    Ok(Varnode::constant(address.offset & size_mask(size), size))
                } else {
                    Ok(address)
                }
            }
            Expr::Value(ExprElement::Op(unary)) => {
                let size = self.execution_len_bytes(scope, expr);
                let input = self.expr(scope, &unary.input, None)?;
                self.unary(&unary.op, input, size)
            }
            Expr::Value(element @ (ExprElement::New(_) | ExprElement::CPool(_))) => {
                let (opcode, params) = match element {
                    ExprElement::New(new) => (
                        OpCode::New,
                        std::iter::once(&*new.first)
                            .chain(new.second.as_deref())
                            .collect::<Vec<_>>(),
                    ),
                    ExprElement::CPool(cpool) => (OpCode::CPoolRef, cpool.params.iter().collect()),
                    _ => unreachable!(),
                };
                let inputs = params
                    .into_iter()
                    .map(|param| self.expr(scope, param, None))
                    .collect::<LiftResult<Vec<_>>>()?;
                let size = size_hint.unwrap_or(self.sleigh.addr_bytes().get() as u32);
                Ok(self.op(opcode, size, inputs))
            }
            Expr::Op(binary) => {
                let size = bits_to_bytes(binary.len_bits.get());
                let left = self.expr(scope, &binary.left, Some(size))?;
                let right = self.expr(scope, &binary.right, Some(size))?;
                self.binary(binary.op, left, right, size)
            }
        }
    }

    fn expr_value(&mut self, scope: &mut Scope, value: &ExprValue) -> LiftResult<Varnode> {
        let execution = scope.table.constructor.execution.as_ref().unwrap();
        let size = || bits_to_bytes(value.len_bits(self.sleigh, execution).get());
        Ok(match value {
            ExprValue::Int(number) => Varnode::constant(
                number.number.signed_super() as u64,
                bits_to_bytes(number.size.get()),
            ),
            ExprValue::TokenField(field) => Varnode::constant(
                self.token_field(scope, field.id)? as u64,
                bits_to_bytes(field.size.get()),
            ),
            ExprValue::InstStart(_) => Varnode::constant(scope.table.inst_start, size()),
            ExprValue::InstNext(_) => Varnode::constant(scope.inst_next, size()),
            ExprValue::Varnode(varnode_id) => self.register(*varnode_id),
            ExprValue::VarnodeDynamic(dynamic) => {
                self.attached(scope, dynamic.attach_id, dynamic.attach_value)?
            }
            ExprValue::Table(table_id) => {
                let handle = self.table_export(scope, *table_id)?;
                self.read_handle(handle)
            }
            ExprValue::DisVar(var) => {
                let value = scope.table.variables.get(&var.id).copied().ok_or_else(|| {
                    LiftError::Invalid(format!("disassembly variable {} undefined", var.id.0))
                })?;
                Varnode::constant(value as u64, bits_to_bytes(var.size.get()))
            }
            ExprValue::ExeVar(variable_id) => self.local(scope, *variable_id),
            ExprValue::Bitrange(bitrange) => {
                let bitrange = self.sleigh.bitrange(bitrange.id);
                let source = self.register(bitrange.varnode);
                let range = bitrange.bits.start()..bitrange.bits.end().get();
                self.read_bits(source, range, size())
            }
            ExprValue::Context(context) => Varnode::constant(
                scope.table.context.get(self.sleigh, context.id) as u64,
                bits_to_bytes(context.size.get()),
            ),
            // A token field attached to numbers reads as the number it selects
            ExprValue::IntDynamic(dynamic) => Varnode::constant(
                self.attached_number(scope, dynamic.attach_id, dynamic.attach_value)?,
                bits_to_bytes(dynamic.bits.get()),
            ),
        })
    }

    fn unary(&mut self, op: &Unary, input: Varnode, size: u32) -> LiftResult<Varnode> {
        // Constant folding keeps disassembly time values constant. Constants hold 64 bits, so
        // wider values are left to the emulator.
        if input.is_const() && input.size <= 8 && size <= 8 {
            let value = input.offset;
            let folded = match op {
                Unary::Zext(_) | Unary::TakeLsb(_) => Some(value),
                Unary::Sext(_) => Some(sign_extend(value, 8 * input.size.min(8)) as u64),
                Unary::TrunkLsb { trunk, .. } => {
                    Some(value.checked_shr(8 * *trunk as u32).unwrap_or(0))
                }
                Unary::BitRange { range, .. } => {
                    let len = range.end - range.start;
                    let mask = if len >= 64 { u64::MAX } else { (1 << len) - 1 };
                    Some(value.checked_shr(range.start as u32).unwrap_or(0) & mask)
                }
                Unary::BitNegation => Some(!value),
                Unary::Negative => Some(value.wrapping_neg()),
                Unary::Negation => Some((value & size_mask(input.size) == 0) as u64),
                _ => None,
            };
            if let Some(value) = folded {
                return Ok(Varnode::constant(value, size));
            }
        }

        Ok(match op {
            Unary::Dereference(memory) => {
                let space = Varnode::constant(memory.space.0 as u64, 4);
                self.op(
                    OpCode::Load,
                    memory.len_bytes.get() as u32,
                    vec![space, input],
                )
            }
            Unary::Zext(_) if input.size >= size => self.resize(input, size),
            Unary::Zext(_) => self.op(OpCode::IntZExt, size, vec![input]),
            Unary::Sext(_) if input.size >= size => self.resize(input, size),
            Unary::Sext(_) => self.op(OpCode::IntSExt, size, vec![input]),
            Unary::TakeLsb(_) => self.resize(input, size),
            Unary::TrunkLsb { trunk, .. } => self.op(
                OpCode::Subpiece,
                size,
                vec![input, Varnode::constant(*trunk, 4)],
            ),
            Unary::BitRange { range, .. } => self.read_bits(input, range.clone(), size),
            Unary::Negation => {
                let input = self.resize(input, 1);
                self.op(OpCode::BoolNegate, 1, vec![input])
            }
            Unary::BitNegation => self.op(OpCode::IntNegate, input.size, vec![input]),
            Unary::Negative => self.op(OpCode::Int2Comp, input.size, vec![input]),
            Unary::Popcount(_) => self.op(OpCode::Popcount, size, vec![input]),
            Unary::Lzcount(_) => self.op(OpCode::Lzcount, size, vec![input]),
            Unary::FloatNan(_) => self.op(OpCode::FloatNan, size, vec![input]),
            Unary::SignTrunc(_) => self.op(OpCode::FloatTrunc, size, vec![input]),
            Unary::Float2Float(_) => self.op(OpCode::FloatFloat2Float, size, vec![input]),
            Unary::Int2Float(_) => self.op(OpCode::FloatInt2Float, size, vec![input]),
            Unary::FloatNegative => self.op(OpCode::FloatNeg, input.size, vec![input]),
            Unary::FloatAbs => self.op(OpCode::FloatAbs, input.size, vec![input]),
            Unary::FloatSqrt => self.op(OpCode::FloatSqrt, input.size, vec![input]),
            Unary::FloatCeil => self.op(OpCode::FloatCeil, input.size, vec![input]),
            Unary::FloatFloor => self.op(OpCode::FloatFloor, input.size, vec![input]),
            Unary::FloatRound => self.op(OpCode::FloatRound, input.size, vec![input]),
        })
    }

    fn binary(
        &mut self,
        op: Binary,
        left: Varnode,
        right: Varnode,
        size: u32,
    ) -> LiftResult<Varnode> {
        use OpCode::*;
        // (opcode, swap operands, operands are booleans, result is a boolean)
        let (opcode, swap, kind) = match op {
            Binary::Mult => (IntMult, false, Kind::Arith),
            Binary::Div => (IntDiv, false, Kind::Arith),
            Binary::SigDiv => (IntSDiv, false, Kind::Arith),
            Binary::Rem => (IntRem, false, Kind::Arith),
            Binary::SigRem => (IntSRem, false, Kind::Arith),
            Binary::Add => (IntAdd, false, Kind::Arith),
            Binary::Sub => (IntSub, false, Kind::Arith),
            Binary::BitAnd => (IntAnd, false, Kind::Arith),
            Binary::BitXor => (IntXor, false, Kind::Arith),
            Binary::BitOr => (IntOr, false, Kind::Arith),
            Binary::Lsl => (IntLeft, false, Kind::Shift),
            Binary::Lsr => (IntRight, false, Kind::Shift),
            Binary::Asr => (IntSRight, false, Kind::Shift),
            Binary::FloatDiv => (FloatDiv, false, Kind::Arith),
            Binary::FloatMult => (FloatMult, false, Kind::Arith),
            Binary::FloatAdd => (FloatAdd, false, Kind::Arith),
            Binary::FloatSub => (FloatSub, false, Kind::Arith),
            Binary::SigLess => (IntSLess, false, Kind::Compare),
            Binary::SigGreater => (IntSLess, true, Kind::Compare),
            Binary::SigLessEq => (IntSLessEqual, false, Kind::Compare),
            Binary::SigGreaterEq => (IntSLessEqual, true, Kind::Compare),
            Binary::Less => (IntLess, false, Kind::Compare),
            Binary::Greater => (IntLess, true, Kind::Compare),
            Binary::LessEq => (IntLessEqual, false, Kind::Compare),
            Binary::GreaterEq => (IntLessEqual, true, Kind::Compare),
            Binary::Eq => (IntEqual, false, Kind::Compare),
            Binary::Ne => (IntNotEqual, false, Kind::Compare),
            Binary::Carry => (IntCarry, false, Kind::Compare),
            Binary::SCarry => (IntSCarry, false, Kind::Compare),
            Binary::SBorrow => (IntSBorrow, false, Kind::Compare),
            Binary::FloatLess => (FloatLess, false, Kind::Compare),
            Binary::FloatGreater => (FloatLess, true, Kind::Compare),
            Binary::FloatLessEq => (FloatLessEqual, false, Kind::Compare),
            Binary::FloatGreaterEq => (FloatLessEqual, true, Kind::Compare),
            Binary::FloatEq => (FloatEqual, false, Kind::Compare),
            Binary::FloatNe => (FloatNotEqual, false, Kind::Compare),
            Binary::And => (BoolAnd, false, Kind::Bool),
            Binary::Xor => (BoolXor, false, Kind::Bool),
            Binary::Or => (BoolOr, false, Kind::Bool),
        };
        let (left, right) = if swap { (right, left) } else { (left, right) };
        let (left, right, output_size) = match kind {
            Kind::Arith => (self.resize(left, size), self.resize(right, size), size),
            Kind::Shift => (self.resize(left, size), right, size),
            Kind::Compare => {
                // Constants take the size of the other operand
                let operand_size = match (left.is_const(), right.is_const()) {
                    (true, false) => right.size,
                    (false, true) => left.size,
                    _ => left.size.max(right.size),
                };
                (
                    self.resize(left, operand_size),
                    self.resize(right, operand_size),
                    1,
                )
            }
            Kind::Bool => (self.resize(left, 1), self.resize(right, 1), 1),
        };
        Ok(self.op(opcode, output_size, vec![left, right]))
    }
}

enum Kind {
    Arith,
    Shift,
    Compare,
    Bool,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::context::Context;
    use crate::disassembler::Disassembler;
    use std::path::Path;

    fn load(slaspec_path: impl AsRef<Path>) -> Sleigh {
        let _ = env_logger::try_init();
        sleigh_rs::file_to_sleigh(slaspec_path.as_ref()).unwrap_or_else(|_| {
            panic!("Could not load slaspec: {:?}", slaspec_path.as_ref());
        })
    }

    /// The disassembly and p-code text of an instruction at `address`
    fn lift_text(
        sleigh: &Sleigh,
        context: &Context,
        address: u64,
        bytes: &[u8],
    ) -> (String, Result<Vec<String>, LiftError>) {
        let disassembler = Disassembler::new(sleigh);
        let instruction = disassembler.disassemble(address, context, bytes).unwrap();
        let pcode = instruction.pcode.clone().map(|ops| {
            ops.iter()
                .map(|op| op.display(sleigh).to_string())
                .collect()
        });
        (instruction.to_string(), pcode)
    }

    /// Each case is (bytes, disassembly, p-code), lifted at 0x1000
    fn assert_lifts(sleigh: &Sleigh, tests: &[(Vec<u8>, &str, &[&str])]) {
        assert_lifts_in_context(sleigh, &Context::new(sleigh), tests)
    }

    /// Like `assert_lifts`, decoding with `context`
    fn assert_lifts_in_context(
        sleigh: &Sleigh,
        context: &Context,
        tests: &[(Vec<u8>, &str, &[&str])],
    ) {
        let mut failures = vec![];
        for (bytes, expected_text, expected_pcode) in tests.iter() {
            let (text, pcode) = lift_text(sleigh, context, 0x1000, bytes);
            let expected_pcode: Vec<String> =
                expected_pcode.iter().map(|op| op.to_string()).collect();
            if text != *expected_text || pcode.as_ref() != Ok(&expected_pcode) {
                failures.push(format!(
                    "{:02x?}: expected {:?} {:?}, got {:?} {:?}",
                    bytes, expected_text, expected_pcode, text, pcode
                ));
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    #[test]
    fn context_pcode() {
        let sleigh = load("examples/context.slaspec");
        let mut context = Context::new(&sleigh);
        context.set(&sleigh, Context::id(&sleigh, "shift").unwrap(), 2);
        #[rustfmt::skip]
        assert_lifts_in_context(&sleigh, &context, &[
            (vec![0x00, 0x00, 0x12, 0x05], "shl r1, r2, 0x2", &["r1 = INT_LEFT r2, 0x2:4"]),
            (vec![0x07, 0x00, 0x18, 0x06], "mov r1, #0x7", &["r1 = COPY 0x7:4"]),
            (vec![0x00, 0x00, 0x12, 0x06], "mov r1, r2", &["r1 = COPY r2"]),
            (vec![0x00, 0x00, 0x00, 0x08], "inc r6", &["r6 = INT_ADD r6, 0x1:4"]),
        ]);
        context.set(&sleigh, Context::id(&sleigh, "bank").unwrap(), 1);
        #[rustfmt::skip]
        assert_lifts_in_context(&sleigh, &context, &[
            (vec![0x00, 0x00, 0x00, 0x08], "inc r7", &["r7 = INT_ADD r7, 0x1:4"]),
        ]);
    }

    #[test]
    fn risc_pcode() {
        let sleigh = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_lifts(&sleigh, &[
            (vec![0x01, 0x08, 0x12, 0x34], "add r1, r2, 0x1234", &["r1 = INT_ADD r2, 0x1234:4"]),
            (vec![0x01, 0x0c, 0xff, 0xff], "add r1, r2, -0x1", &["r1 = INT_ADD r2, 0xffffffff:4"]),
            (vec![0x01, 0x0e, 0xff, 0x00], "add r1, r2, -0x10000", &["r1 = INT_ADD r2, 0xffff0000:4"]),
            (vec![0x29, 0x08, 0xff, 0xff], "xor r1, r2, 0xffff", &["r1 = INT_XOR r2, 0xffff:4"]),
            (vec![0xf9, 0x09, 0x80, 0x01], "sub r1, r2, r3", &["r1 = INT_SUB r2, r3"]),
            (vec![0x39, 0x08, 0x00, 0x01], "unk.0x7 r1, r2, 0x1", &[]),
        ]);
    }

    #[test]
    fn cisc_pcode() {
        let sleigh = load("examples/cisc.slaspec");
        #[rustfmt::skip]
        assert_lifts(&sleigh, &[
            (vec![0x00], "nop", &[]),
            (vec![0x02, 0x0a], "add r1, r2", &["r1 = INT_ADD r1, r2", "ZF = INT_EQUAL r1, 0x0:4"]),
            (vec![0x07, 0x0a], "cmp r1, r2", &["ZF = INT_EQUAL r1, r2"]),
            (vec![0x01, 0xc8, 0x12, 0x34, 0x56, 0x78], "mov r1, #0x12345678", &["r1 = COPY 0x12345678:4"]),
            (vec![0x01, 0x8a, 0x04], "mov r1, [r2+0x4]", &["$U0:4 = INT_ADD r2, 0x4:4", "r1 = LOAD ram, $U0:4"]),
            (vec![0x08, 0x8a, 0xfc], "mov [r2+-0x4], r1", &["$U0:4 = INT_ADD r2, 0xfffffffc:4", "STORE ram, $U0:4, r1"]),
            (vec![0x02, 0x4a], "add r1, [r2]", &["$U0:4 = LOAD ram, r2", "r1 = INT_ADD r1, $U0:4", "ZF = INT_EQUAL r1, 0x0:4"]),
            (vec![0x10, 0x00, 0x00, 0x20, 0x00], "jmp 0x2000", &["BRANCH ram[0x2000]:4"]),
            (vec![0x11, 0x10], "jz 0x1012", &["CBRANCH ram[0x1012]:4, ZF"]),
            (vec![0x12, 0xfe], "jnz 0x1000", &["$U0:1 = BOOL_NEGATE ZF", "CBRANCH ram[0x1000]:4, $U0:1"]),
            (vec![0x20, 0x18], "out r3", &["CALLOTHER \"out\", r3"]),
            (vec![0x21, 0x08], "in r1", &["r1 = CALLOTHER \"in\""]),
        ]);
    }

    #[test]
    fn vliw_pcode() {
        let sleigh = load("examples/vliw.slaspec");
        // Slots are built in order, so slot 1 reads the r1 slot 0 wrote
        #[rustfmt::skip]
        assert_lifts(&sleigh, &[
            (vec![0x88, 0x04, 0x30, 0x10, 0x40, 0x00, 0x00, 0x01], "{ add r1, r1, 0x1 ; add r2, r1, 0x1 }", &["r1 = INT_ADD r1, 0x1:4", "r2 = INT_ADD r1, 0x1:4"]),
            (vec![0x08, 0x04, 0x54, 0x18, 0xe6, 0x42, 0xc0, 0xc7], "{ add r1, r2, 0 ; xor r3, r3, 0 ; or r4, r5, 0 ; mov r6, r7 }", &["r1 = INT_ADD r2, 0x0:4", "r3 = INT_XOR r3, 0x0:4", "r4 = INT_OR r5, 0x0:4", "r6 = COPY r7"]),
        ]);
    }

    #[test]
    fn layout_pcode() {
        let sleigh = load("examples/layout.slaspec");
        #[rustfmt::skip]
        assert_lifts(&sleigh, &[
            (vec![0x01, 0x02, 0x05], "ldi 0x5, r2", &["r2 = COPY 0x5:4"]),
            (vec![0x02, 0x01, 0x02], "mov r1, r2", &["r1 = COPY r2"]),
            (vec![0x03, 0x80, 0x07, 0x03], "add #0x7, r3", &["r3 = INT_ADD r3, 0x7:4"]),
        ]);
    }

    #[test]
    fn delay_slot_pcode() {
        let sleigh = load("examples/delay.slaspec");
        // The delay slot's p-code runs at delayslot, inst_next is past it and the branch's
        // commit reaches it
        #[rustfmt::skip]
        assert_lifts(&sleigh, &[
            (vec![0x10, 0x22, 0x00, 0x03, 0x20, 0x63, 0x00, 0x01], "beq r1, r2, 0x1010", &["$U0:1 = INT_EQUAL r1, r2", "r3 = INT_ADD r3, 0x1:4", "CBRANCH ram[0x1010]:4, $U0:1"]),
            (vec![0x0c, 0x00, 0x08, 0x00, 0x04, 0x03, 0x00, 0x00], "jal 0x2000", &["ra = COPY 0x1008:4", "r3 = COPY 0x1:4", "CALL ram[0x2000]:4"]),
            (vec![0x18, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], "jr ra", &["$U0:4 = COPY ra", "RETURN $U0:4"]),
            (vec![0x04, 0x03, 0x00, 0x00], "slot r3, 0x0", &["r3 = COPY 0x0:4"]),
            // Branch likely skips the delay slot by branching to inst_next
            (vec![0x50, 0x22, 0x00, 0x03, 0x20, 0x63, 0x00, 0x01], "beql r1, r2, 0x1010", &["$U0:1 = INT_EQUAL r1, r2", "$U8:1 = BOOL_NEGATE $U0:1", "CBRANCH ram[0x1008]:4, $U8:1", "r3 = INT_ADD r3, 0x1:4", "BRANCH ram[0x1010]:4"]),
            // A jump inside the delay slot stays relative to its own ops
            (vec![0x10, 0x00, 0x00, 0x03, 0x28, 0x22, 0x18, 0x00], "beq r0, r0, 0x1010", &["$U0:1 = INT_EQUAL r0, r0", "r3 = COPY r1", "$U10:1 = INT_SLESSEQUAL r2, r1", "CBRANCH +2, $U10:1", "r3 = COPY r2", "CBRANCH ram[0x1010]:4, $U0:1"]),
        ]);
    }

    #[test]
    fn delay_slot_instructions() {
        let sleigh = load("examples/delay.slaspec");
        let disassembler = Disassembler::new(&sleigh);
        let context = Context::new(&sleigh);
        let jal = [0x0c, 0x00, 0x08, 0x00, 0x04, 0x03, 0x00, 0x00];
        let instruction = disassembler.disassemble(0x1000, &context, &jal).unwrap();
        assert_eq!(instruction.inst_next, 0x1004);
        assert_eq!(instruction.fallthrough(), 0x1008);
        let slots: Vec<String> = instruction
            .delay_slots
            .iter()
            .map(|slot| slot.to_string())
            .collect();
        assert_eq!(slots, vec!["slot r3, 0x1"]);

        // Without the bytes after it, or with a branch in its delay slot, a branch still
        // disassembles
        let nested = [0x0c, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x08, 0x00];
        let missing = "delay slot at 0x1004: instruction: Failed to disassemble table";
        for (bytes, err) in [
            (&jal[..4], missing),
            (&nested, "delay slot at 0x1004: nested delay slot"),
        ] {
            let (text, pcode) = lift_text(&sleigh, &context, 0x1000, bytes);
            assert_eq!(text, "jal 0x2000");
            assert_eq!(pcode, Err(LiftError::Invalid(err.to_string())));
        }
        let instruction = disassembler.disassemble(0x1000, &context, &nested).unwrap();
        assert_eq!(instruction.fallthrough(), 0x1008);
    }

    #[test]
    fn partial_writes_pcode() {
        let sleigh = load("examples/pcode.slaspec");
        #[rustfmt::skip]
        assert_lifts(&sleigh, &[
            (vec![0x01, 0x02], "sethi r2", &["$U0:4 = INT_RIGHT r2, 0x0:4", "$U8:4 = INT_AND $U0:4, 0xff:4", "$U10:1 = SUBPIECE $U8:4, 0x0:4", "$U18:4 = INT_ZEXT $U10:1", "$U20:4 = INT_LEFT $U18:4, 0x10:4", "$U28:4 = INT_AND $U20:4, 0xff0000:4", "$U30:4 = INT_AND flags, 0xff00ffff:4", "flags = INT_OR $U30:4, $U28:4"]),
            (vec![0x02, 0x02], "setbits r2", &["$U0:4 = INT_RIGHT r2, 0x0:4", "$U8:4 = INT_AND $U0:4, 0xf:4", "$U10:1 = SUBPIECE $U8:4, 0x0:4", "$U18:4 = INT_ZEXT $U10:1", "$U20:4 = INT_LEFT $U18:4, 0x14:4", "$U28:4 = INT_AND $U20:4, 0xf00000:4", "$U30:4 = INT_AND flags, 0xff0fffff:4", "flags = INT_OR $U30:4, $U28:4"]),
            (vec![0x03, 0x01], "sth [r1], r0", &["$U0:4 = INT_RIGHT r0, 0x0:4", "$U8:4 = INT_AND $U0:4, 0xffff:4", "$U10:2 = SUBPIECE $U8:4, 0x0:4", "STORE ram, r1, $U10:2"]),
            (vec![0x04, 0x01], "stbits [r1], r0", &["$U0:4 = INT_RIGHT r0, 0x0:4", "$U8:4 = INT_AND $U0:4, 0xff:4", "$U10:1 = SUBPIECE $U8:4, 0x0:4", "$U18:4 = LOAD ram, r1", "$U20:4 = INT_ZEXT $U10:1", "$U28:4 = INT_LEFT $U20:4, 0x4:4", "$U30:4 = INT_AND $U28:4, 0xff0:4", "$U38:4 = INT_AND $U18:4, 0xfffff00f:4", "$U18:4 = INT_OR $U38:4, $U30:4", "STORE ram, r1, $U18:4"]),
            (vec![0x05, 0x01], "stmid [r1], r0", &["$U0:4 = INT_RIGHT r0, 0x0:4", "$U8:4 = INT_AND $U0:4, 0xffff:4", "$U10:2 = SUBPIECE $U8:4, 0x0:4", "$U18:4 = LOAD ram, r1", "$U20:4 = INT_ZEXT $U10:2", "$U28:4 = INT_LEFT $U20:4, 0x8:4", "$U30:4 = INT_AND $U28:4, 0xffff00:4", "$U38:4 = INT_AND $U18:4, 0xff0000ff:4", "$U18:4 = INT_OR $U38:4, $U30:4", "STORE ram, r1, $U18:4"]),
        ]);
        let (text, pcode) = lift_text(&sleigh, &Context::new(&sleigh), 0x1000, &[0x06, 0x01]);
        assert_eq!(text, "stbad [r1], r0");
        assert!(matches!(pcode, Err(LiftError::Invalid(_))), "{:?}", pcode);
    }

    #[test]
    fn references_pcode() {
        let sleigh = load("examples/pcode.slaspec");
        #[rustfmt::skip]
        assert_lifts(&sleigh, &[
            (vec![0x07, 0x2a], "lea r0, 0x2a", &["r0 = COPY 0x2a:4"]),
            (vec![0x08, 0x02], "leas r0, -0x1", &["r0 = COPY 0xffffffff:4"]),
            (vec![0x09, 0x2a], "leai r0, #0x2a", &["r0 = COPY 0x2a:4"]),
            (vec![0x0a, 0xfe], "leass r0, -0x2", &["r0 = COPY 0xfffffffe:4"]),
            (vec![0x0b, 0x02], "lean r0, c", &["r0 = COPY 0x2:4"]),
            (vec![0x0c, 0x34, 0x12], "leab r0, #0x1234", &["r0 = COPY 0x34:4"]),
        ]);
    }

    #[test]
    fn attached_numbers_pcode() {
        let sleigh = load("examples/pcode.slaspec");
        #[rustfmt::skip]
        assert_lifts(&sleigh, &[
            (vec![0x0d, 0x01], "adds r0, 0x8", &["r0 = INT_ADD r0, 0x8:4"]),
            (vec![0x0d, 0x02], "adds r0, -0x1", &["r0 = INT_ADD r0, 0xffffffff:4"]),
        ]);
    }

    #[test]
    fn new_and_cpool_pcode() {
        use sleigh_rs::execution::{ExprCPool, ExprNew};
        // sleigh-rs cannot size `newobject` and `cpool` yet and panics on specs using them, so
        // build them by hand from the operands of `r0 = r0 + sel`
        let sleigh = load("examples/pcode.slaspec");
        let disassembler = Disassembler::new(&sleigh);
        let instruction = disassembler
            .disassemble(0x1000, &Context::new(&sleigh), &[0x0d, 0x01])
            .unwrap();
        let execution = instruction.table.constructor.execution.as_ref().unwrap();
        let Statement::Assignment(assignment) =
            &execution.block(execution.entry_block).statements[0]
        else {
            panic!("adds is not an assignment");
        };
        let Expr::Op(binary) = &assignment.right else {
            panic!("adds does not add");
        };
        let Expr::Value(ExprElement::Value { location, .. }) = &*binary.left else {
            panic!("adds does not read r0");
        };
        let exprs = [
            Expr::Value(ExprElement::New(ExprNew {
                location: location.clone(),
                first: binary.left.clone(),
                second: None,
            })),
            Expr::Value(ExprElement::New(ExprNew {
                location: location.clone(),
                first: binary.left.clone(),
                second: Some(binary.right.clone()),
            })),
            Expr::Value(ExprElement::CPool(ExprCPool {
                location: location.clone(),
                params: vec![(*binary.left).clone(), (*binary.right).clone()].into(),
            })),
        ];

        let mut lifter = Lifter::new(&sleigh, &[]);
        let mut scope = Scope::new(&instruction.table, instruction.table.inst_next);
        for expr in exprs.iter() {
            lifter.expr(&mut scope, expr, Some(4)).unwrap();
        }
        let pcode: Vec<String> = lifter
            .ops
            .iter()
            .map(|op| op.display(&sleigh).to_string())
            .collect();
        assert_eq!(
            pcode,
            [
                "$U0:4 = NEW r0",
                "$U8:4 = NEW r0, 0x8:4",
                "$U10:4 = CPOOLREF r0, 0x8:4"
            ]
        );
    }

    #[test]
    fn empty_delay_slot() {
        let path =
            std::env::temp_dir().join(format!("sleigher-delayslot-{}.slaspec", std::process::id()));
        std::fs::write(
            &path,
            "
            define endian=big;
            define alignment=1;
            define space ram type=ram_space size=4 default;
            define space register type=register_space size=4;
            define register offset=0x00 size=4 [ r0 ];
            define register offset=0x100 size=4 contextreg;
            define context contextreg inslot=(0, 0) noflow;
            define token opbyte(8) opcode=(0, 7);
            :zl is inslot=1 { r0 = 1; }
            :dly is inslot=0 & opcode=0x01 [ inslot=1; globalset(inst_next, inslot); ] {
                delayslot(1);
            }
        ",
        )
        .unwrap();
        let sleigh = load(&path);
        std::fs::remove_file(&path).unwrap();

        // An instruction without bytes never fills the delay slot
        let (text, pcode) = lift_text(&sleigh, &Context::new(&sleigh), 0x1000, &[0x01, 0x00, 0x00]);
        assert_eq!(text, "dly");
        assert_eq!(
            pcode,
            Err(LiftError::Invalid(
                "delay slot at 0x1001: empty".to_string()
            ))
        );
    }

    /// A direct branch to memory at a dynamic address goes to the address, indirectly
    #[test]
    fn dynamic_branch_pcode() {
        let sleigh = load("examples/flow.slaspec");
        #[rustfmt::skip]
        assert_lifts(&sleigh, &[
            (vec![0x00, 0x00, 0x00, 0x0d], "jd [r0]", &["BRANCHIND r0"]),
            (vec![0x00, 0x00, 0x00, 0x0e], "calld [r0]", &["CALLIND r0"]),
            (vec![0x00, 0x00, 0x00, 0x0f], "jv [vec]", &["BRANCHIND vec"]),
            (vec![0x00, 0x20, 0x00, 0x01], "jmp 0x2000", &["BRANCH ram[0x2000]:4"]),
        ]);
    }
}
