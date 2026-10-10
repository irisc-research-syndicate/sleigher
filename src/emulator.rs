use std::collections::HashMap;

use anyhow::{anyhow, bail, Context as _, Result};

use sleigh_rs::{Endian, Sleigh, SpaceId, UserFunctionId};

use crate::context::{Context, ContextFlow};
use crate::disassembler::Disassembler;
use crate::pcode::{size_mask, BranchTarget, OpCode, PcodeOp, Varnode, VarnodeSpace};
use crate::space::{HashSpace, MemoryRegion};
use crate::value::{Address, Ref};

/// Bytes fetched per step when the instruction length is unbounded (recursive patterns)
const FALLBACK_INSTRUCTION_LEN: usize = 16;

/// Handler for a user defined p-code op: gets the state and the input values, returns the
/// output value if the op has one
pub type UserOp = Box<dyn FnMut(&mut State, &[u64]) -> Result<Option<u64>>>;

pub struct Cpu<'sleigh> {
    pub sleigh: &'sleigh Sleigh,
    pub disassembler: Disassembler<'sleigh>,
    pub state: State,
    /// Bytes fetched per step: the longest instruction and its delay slot
    pub fetch_len: usize,
    pub context: ContextFlow,
    user_ops: HashMap<String, UserOp>,
}

impl<'sleigh> std::ops::Deref for Cpu<'sleigh> {
    type Target = Sleigh;

    fn deref(&self) -> &Self::Target {
        self.sleigh
    }
}

impl<'sleigh> Cpu<'sleigh> {
    pub fn new(sleigh: &'sleigh Sleigh, state: State) -> Self {
        let max_instruction_len = sleigh
            .table(sleigh.instruction_table())
            .pattern_len
            .max()
            .map_or(FALLBACK_INSTRUCTION_LEN, |len| len as usize);
        let disassembler = Disassembler::new(sleigh);
        // Delay slot instructions start within the delay slot, so the last one ends at most
        // max_instruction_len past it
        let fetch_len = match disassembler.max_delay_slot_len() as usize {
            0 => max_instruction_len,
            len => 2 * max_instruction_len + len - 1,
        };
        Self {
            sleigh,
            disassembler,
            state,
            fetch_len,
            context: ContextFlow::new(Context::new(sleigh)),
            user_ops: HashMap::new(),
        }
    }

    /// Handle the user defined p-code op `name` (`define pcodeop name;`) with `handler`
    pub fn register_user_op(
        &mut self,
        name: &str,
        handler: impl FnMut(&mut State, &[u64]) -> Result<Option<u64>> + 'static,
    ) {
        self.user_ops.insert(name.to_string(), Box::new(handler));
    }

    pub fn step(&mut self) -> Result<()> {
        let mut instruction_bytes = vec![0u8; self.fetch_len];
        self.fetch_instruction(&mut instruction_bytes)?;

        // The context only moves on once the instruction executed. Commits a failed step
        // made stay, but they are what decoding the same bytes commits again.
        let next = self.context.next.clone();
        let result = self.execute_instruction(&instruction_bytes);
        if result.is_err() {
            self.context.next = next;
        }
        result
    }

    /// Decode the instruction at pc from `instruction_bytes` and run it
    fn execute_instruction(&mut self, instruction_bytes: &[u8]) -> Result<()> {
        let instruction = self.disassembler.disassemble_in_flow(
            self.state.pc,
            &mut self.context,
            instruction_bytes,
        )?;
        log::debug!(
            "Executing {:#010x}: {}",
            instruction.inst_start,
            instruction
        );
        for slot in instruction.delay_slots.iter() {
            log::debug!("  delay slot {:#010x}: {}", slot.inst_start, slot);
        }
        let ops = instruction
            .pcode
            .as_ref()
            .map_err(|err| anyhow!("{:#010x}: {}: {}", instruction.inst_start, instruction, err))?;

        let mut executor = PcodeExecutor {
            sleigh: self.sleigh,
            state: &mut self.state,
            user_ops: &mut self.user_ops,
            uniques: HashSpace::new(),
        };
        let pc = executor
            .execute(ops, instruction.fallthrough())
            .with_context(|| format!("{:#010x}: {}", instruction.inst_start, instruction))?;
        self.state.pc = pc;
        Ok(())
    }

    pub fn fetch_instruction(&mut self, instruction: &mut [u8]) -> Result<()> {
        let pc = self.state.pc;
        self.state.read_ref(
            Ref(self.sleigh.default_space(), instruction.len(), Address(pc)),
            instruction,
        )
    }
}

#[derive(Debug, Default)]
pub struct State {
    pub pc: u64,
    pub spaces: HashMap<SpaceId, Box<dyn MemoryRegion>>,
}

impl State {
    pub fn new() -> Self {
        State {
            pc: 0,
            spaces: HashMap::new(),
        }
    }

    fn space(&mut self, space: SpaceId) -> &mut Box<dyn MemoryRegion> {
        self.spaces
            .entry(space)
            .or_insert_with(|| Box::new(HashSpace::new()))
    }

    pub fn write_ref(&mut self, referance: Ref, data: &[u8]) -> Result<()> {
        log::trace!("Writing {} <- {:02x?}", referance, data);
        assert!(data.len() >= referance.1);
        let data = &data[data.len() - referance.1..];
        self.space(referance.0).write(referance.2, data)
    }

    pub fn read_ref(&mut self, referance: Ref, data: &mut [u8]) -> Result<()> {
        let len = data.len();
        assert!(len >= referance.1);
        for byte in &mut *data {
            *byte = 0;
        }
        let data = &mut data[len - referance.1..];
        self.space(referance.0).read(referance.2, data)?;
        log::trace!("Reading {} -> {:02x?}", referance, data);
        Ok(())
    }

    pub fn read_ref_u32be(&mut self, referance: Ref) -> Result<u32> {
        let mut bytes = [0u8; 4];
        self.read_ref(referance, &mut bytes)?;
        Ok(u32::from_be_bytes(bytes))
    }
}

fn read_value(region: &dyn MemoryRegion, endian: Endian, address: u64, size: u32) -> Result<u64> {
    let mut bytes = vec![0u8; size as usize];
    region.read(Address(address), &mut bytes)?;
    let fold = |value: u64, byte: &u8| (value << 8) | *byte as u64;
    Ok(match endian {
        Endian::Big => bytes.iter().fold(0, fold),
        Endian::Little => bytes.iter().rev().fold(0, fold),
    })
}

fn write_value(
    region: &mut dyn MemoryRegion,
    endian: Endian,
    address: u64,
    size: u32,
    value: u64,
) -> Result<()> {
    let bytes = match endian {
        Endian::Big => value.to_be_bytes()[8 - size as usize..].to_vec(),
        Endian::Little => value.to_le_bytes()[..size as usize].to_vec(),
    };
    region.write(Address(address), &bytes)
}

fn sign_extend(value: u64, size: u32) -> i64 {
    crate::value::sign_extend(value, 8 * size.min(8))
}

/// Runs the p-code of one instruction
struct PcodeExecutor<'e> {
    sleigh: &'e Sleigh,
    state: &'e mut State,
    user_ops: &'e mut HashMap<String, UserOp>,
    uniques: HashSpace,
}

impl PcodeExecutor<'_> {
    fn read(&mut self, varnode: &Varnode) -> Result<u64> {
        if varnode.size > 8 {
            bail!("varnode {:?} is wider than 8 bytes", varnode);
        }
        let endian = self.sleigh.endian();
        Ok(match varnode.space {
            VarnodeSpace::Const => varnode.offset & size_mask(varnode.size),
            VarnodeSpace::Unique => {
                read_value(&self.uniques, endian, varnode.offset, varnode.size)?
            }
            VarnodeSpace::Space(space) => read_value(
                self.state.space(space).as_ref(),
                endian,
                varnode.offset,
                varnode.size,
            )?,
        })
    }

    fn write(&mut self, varnode: &Varnode, value: u64) -> Result<()> {
        if varnode.size > 8 {
            bail!("varnode {:?} is wider than 8 bytes", varnode);
        }
        let endian = self.sleigh.endian();
        let value = value & size_mask(varnode.size);
        match varnode.space {
            VarnodeSpace::Const => bail!("write to constant {:?}", varnode),
            VarnodeSpace::Unique => write_value(
                &mut self.uniques,
                endian,
                varnode.offset,
                varnode.size,
                value,
            ),
            VarnodeSpace::Space(space) => write_value(
                self.state.space(space).as_mut(),
                endian,
                varnode.offset,
                varnode.size,
                value,
            ),
        }
    }

    fn input(&mut self, op: &PcodeOp, index: usize) -> Result<u64> {
        let varnode = op
            .inputs
            .get(index)
            .with_context(|| format!("{} has no input {}", op.opcode.name(), index))?;
        self.read(varnode)
    }

    /// Run `ops`, returning the address to continue at
    fn execute(&mut self, ops: &[PcodeOp], inst_next: u64) -> Result<u64> {
        let mut index = 0;
        while index < ops.len() {
            let op = &ops[index];
            log::trace!("{}", op.display(self.sleigh));
            match self.execute_op(op)? {
                Flow::Next => index += 1,
                Flow::Relative(offset) => {
                    index = index
                        .checked_add_signed(offset as isize)
                        .filter(|index| *index <= ops.len())
                        .context("relative branch out of the instruction")?;
                }
                Flow::Address(address) => return Ok(address),
            }
        }
        Ok(inst_next)
    }

    /// Where a branch op's target goes
    fn target(op: &PcodeOp) -> Result<Flow> {
        Ok(
            match op.branch_target().context("branch without a target")? {
                BranchTarget::Relative(offset) => Flow::Relative(offset),
                BranchTarget::Address(_, address) => Flow::Address(address),
            },
        )
    }

    fn execute_op(&mut self, op: &PcodeOp) -> Result<Flow> {
        use OpCode::*;
        let flow = match op.opcode {
            Branch | Call => Self::target(op)?,
            CBranch => {
                if self.input(op, 1)? != 0 {
                    Self::target(op)?
                } else {
                    Flow::Next
                }
            }
            BranchInd | CallInd | Return => Flow::Address(self.input(op, 0)?),
            Store => {
                let space = SpaceId(op.inputs[0].offset as usize);
                let address = self.input(op, 1)?;
                let value = self.input(op, 2)?;
                let size = op.inputs[2].size;
                write_value(
                    self.state.space(space).as_mut(),
                    self.sleigh.endian(),
                    address,
                    size,
                    value,
                )?;
                Flow::Next
            }
            _ => {
                let value = self.evaluate(op)?;
                match (&op.output, value) {
                    (Some(output), Some(value)) => self.write(output, value)?,
                    (Some(output), None) => {
                        bail!("{} produced no value for {:?}", op.opcode.name(), output)
                    }
                    (None, _) => {}
                }
                Flow::Next
            }
        };
        Ok(flow)
    }

    /// The value of an op that produces one
    fn evaluate(&mut self, op: &PcodeOp) -> Result<Option<u64>> {
        use OpCode::*;
        let in_size = op
            .inputs
            .get(1)
            .or(op.inputs.first())
            .map_or(8, |input| input.size);
        let size = op.inputs.first().map_or(8, |input| input.size);
        let bits = 8 * size as u64;
        let mask = size_mask(size);
        let a = |executor: &mut Self| executor.input(op, 0);
        let b = |executor: &mut Self| executor.input(op, 1);
        let value = match op.opcode {
            Copy => a(self)?,
            Load => {
                let space = SpaceId(op.inputs[0].offset as usize);
                let address = self.input(op, 1)?;
                let output = op.output.context("LOAD without output")?;
                read_value(
                    self.state.space(space).as_ref(),
                    self.sleigh.endian(),
                    address,
                    output.size,
                )?
            }
            CallOther => {
                let name = self
                    .sleigh
                    .user_function(UserFunctionId(op.inputs[0].offset as usize))
                    .name();
                let args = (1..op.inputs.len())
                    .map(|index| self.input(op, index))
                    .collect::<Result<Vec<_>>>()?;
                let handler = self
                    .user_ops
                    .get_mut(name)
                    .with_context(|| format!("no handler for user op {:?}", name))?;
                return handler(self.state, &args);
            }
            IntEqual => (a(self)? == b(self)?) as u64,
            IntNotEqual => (a(self)? != b(self)?) as u64,
            IntLess => (a(self)? < b(self)?) as u64,
            IntLessEqual => (a(self)? <= b(self)?) as u64,
            IntSLess => (sign_extend(a(self)?, size) < sign_extend(b(self)?, in_size)) as u64,
            IntSLessEqual => (sign_extend(a(self)?, size) <= sign_extend(b(self)?, in_size)) as u64,
            IntZExt => a(self)?,
            IntSExt => sign_extend(a(self)?, size) as u64,
            IntAdd => a(self)?.wrapping_add(b(self)?),
            IntSub => a(self)?.wrapping_sub(b(self)?),
            IntMult => a(self)?.wrapping_mul(b(self)?),
            IntDiv | IntRem | IntSDiv | IntSRem => {
                let (a, b) = (a(self)?, b(self)?);
                if b == 0 {
                    bail!("{}: division by zero", op.opcode.name());
                }
                let (sa, sb) = (sign_extend(a, size), sign_extend(b, size));
                match op.opcode {
                    IntDiv => a / b,
                    IntRem => a % b,
                    IntSDiv => sa.wrapping_div(sb) as u64,
                    _ => sa.wrapping_rem(sb) as u64,
                }
            }
            IntCarry => (a(self)? as u128 + b(self)? as u128 > mask as u128) as u64,
            IntSCarry | IntSBorrow => {
                let (a, b) = (a(self)?, b(self)?);
                let result = if op.opcode == IntSCarry {
                    a.wrapping_add(b)
                } else {
                    a.wrapping_sub(b)
                } & mask;
                let sign = |value: u64| (value >> (bits - 1)) & 1;
                let (sa, sb, sr) = (sign(a), sign(b), sign(result));
                if op.opcode == IntSCarry {
                    (sa == sb && sr != sa) as u64
                } else {
                    (sa != sb && sr != sa) as u64
                }
            }
            Int2Comp => a(self)?.wrapping_neg(),
            IntNegate => !a(self)?,
            IntXor => a(self)? ^ b(self)?,
            IntAnd => a(self)? & b(self)?,
            IntOr => a(self)? | b(self)?,
            IntLeft => a(self)?
                .checked_shl(b(self)?.try_into().unwrap_or(u32::MAX))
                .unwrap_or(0),
            IntRight => a(self)?
                .checked_shr(b(self)?.try_into().unwrap_or(u32::MAX))
                .unwrap_or(0),
            IntSRight => {
                let shift = b(self)?.min(63) as u32;
                (sign_extend(a(self)?, size) >> shift) as u64
            }
            BoolNegate => (a(self)? == 0) as u64,
            BoolXor => ((a(self)? != 0) ^ (b(self)? != 0)) as u64,
            BoolAnd => ((a(self)? != 0) & (b(self)? != 0)) as u64,
            BoolOr => ((a(self)? != 0) | (b(self)? != 0)) as u64,
            Subpiece => a(self)?.checked_shr(8 * b(self)? as u32).unwrap_or(0),
            Popcount => a(self)?.count_ones() as u64,
            Lzcount => (a(self)?.leading_zeros() - (64 - bits as u32)) as u64,
            Branch | CBranch | BranchInd | Call | CallInd | Return | Store => {
                unreachable!("handled by execute_op")
            }
            opcode => bail!("{} is not supported by the emulator", opcode.name()),
        };
        Ok(Some(value))
    }
}

enum Flow {
    Next,
    Relative(i64),
    Address(u64),
}

#[cfg(test)]
mod test {
    use super::*;
    use std::path::Path;

    const BASE: u64 = 0x1000;

    type Regs<'a> = &'a [(&'a str, u64)];
    type Mem<'a> = &'a [(u64, u32)];
    /// (asm, bytes, registers before, memory before, registers after, memory after)
    type Case<'a> = (&'a str, Vec<u8>, Regs<'a>, Mem<'a>, Regs<'a>, Mem<'a>);

    fn load(slaspec_path: impl AsRef<Path>) -> Sleigh {
        let _ = env_logger::try_init();
        sleigh_rs::file_to_sleigh(slaspec_path.as_ref()).unwrap_or_else(|_| {
            panic!("Could not load slaspec: {:?}", slaspec_path.as_ref());
        })
    }

    fn new_cpu<'sleigh>(sleigh: &'sleigh Sleigh, program: &[u8]) -> Cpu<'sleigh> {
        let mut state = State::new();
        state.pc = BASE;
        let program_ref = Ref(sleigh.default_space(), program.len(), Address(BASE));
        state.write_ref(program_ref, program).unwrap();
        Cpu::new(sleigh, state)
    }

    fn reg_ref(cpu: &Cpu, name: &str) -> Ref {
        cpu.varnodes()
            .iter()
            .find(|varnode| varnode.name() == name)
            .unwrap_or_else(|| panic!("No register named {:?}", name))
            .into()
    }

    /// Write `value` to `reference` in the spec's byte order
    fn write_value(cpu: &mut Cpu, reference: Ref, value: u64) {
        let size = reference.1;
        let bytes = match cpu.endian() {
            Endian::Big => value.to_be_bytes()[8 - size..].to_vec(),
            Endian::Little => value.to_le_bytes()[..size].to_vec(),
        };
        cpu.state.write_ref(reference, &bytes).unwrap();
    }

    /// Read `reference` in the spec's byte order
    fn read_value(cpu: &mut Cpu, reference: Ref) -> u64 {
        let mut bytes = vec![0u8; reference.1];
        cpu.state.read_ref(reference, &mut bytes).unwrap();
        let fold = |value: u64, byte: &u8| (value << 8) | *byte as u64;
        match cpu.endian() {
            Endian::Big => bytes.iter().fold(0, fold),
            Endian::Little => bytes.iter().rev().fold(0, fold),
        }
    }

    fn set_reg(cpu: &mut Cpu, name: &str, value: u64) {
        let reg = reg_ref(cpu, name);
        write_value(cpu, reg, value);
    }

    fn get_reg(cpu: &mut Cpu, name: &str) -> u64 {
        let reg = reg_ref(cpu, name);
        read_value(cpu, reg)
    }

    fn set_mem(cpu: &mut Cpu, address: u64, value: u32) {
        let mem = Ref(cpu.default_space(), 4, Address(address));
        write_value(cpu, mem, value as u64);
    }

    fn get_mem(cpu: &mut Cpu, address: u64) -> u32 {
        let mem = Ref(cpu.default_space(), 4, Address(address));
        read_value(cpu, mem) as u32
    }

    /// Execute each instruction once from `BASE` and compare registers, memory and pc.
    /// `pc` may be listed as an expected register, otherwise it must point past the instruction.
    /// The instruction bytes must also disassemble to `asm`, which keeps the encodings honest.
    fn assert_executes(sleigh: &Sleigh, tests: &[Case]) {
        let mut failures = vec![];
        for (asm, program, regs_in, mem_in, regs_out, mem_out) in tests.iter() {
            let mut cpu = new_cpu(sleigh, program);
            match cpu
                .disassembler
                .disassemble(BASE, &Context::new(sleigh), program)
            {
                Ok(instruction) if instruction.to_string() == *asm => {}
                result => failures.push(format!(
                    "{:?}: disassembles as {:?}",
                    asm,
                    result.map(|instruction| instruction.to_string())
                )),
            }
            for (name, value) in regs_in.iter() {
                set_reg(&mut cpu, name, *value);
            }
            for (address, value) in mem_in.iter() {
                set_mem(&mut cpu, *address, *value);
            }

            if let Err(err) = cpu.step() {
                failures.push(format!("{:?}: step failed: {:#}", asm, err));
                continue;
            }

            let mut expected_pc = BASE + program.len() as u64;
            for (name, expected) in regs_out.iter() {
                if *name == "pc" {
                    expected_pc = *expected;
                    continue;
                }
                let actual = get_reg(&mut cpu, name);
                if actual != *expected {
                    failures.push(format!(
                        "{:?}: {} = {:#x}, expected {:#x}",
                        asm, name, actual, expected
                    ));
                }
            }
            if cpu.state.pc != expected_pc {
                failures.push(format!(
                    "{:?}: pc = {:#x}, expected {:#x}",
                    asm, cpu.state.pc, expected_pc
                ));
            }
            for (address, expected) in mem_out.iter() {
                let actual = get_mem(&mut cpu, *address);
                if actual != *expected {
                    failures.push(format!(
                        "{:?}: [{:#x}] = {:#x}, expected {:#x}",
                        asm, address, actual, expected
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    #[test]
    fn risc_immediate_operands() {
        let sleigh = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("add r1, r2, 0x1234", vec![0x01, 0x08, 0x12, 0x34], &[("r2", 0x10)], &[], &[("r1", 0x1244), ("r2", 0x10)], &[]),
            ("sub r1, r2, 0x1", vec![0x09, 0x08, 0x00, 0x01], &[("r2", 0)], &[], &[("r1", 0xffffffff)], &[]),
            ("or r3, r4, 0xff00", vec![0x12, 0x18, 0xff, 0x00], &[("r4", 0x00ff)], &[], &[("r3", 0xffff)], &[]),
            ("and r3, r4, 0xff", vec![0x22, 0x18, 0x00, 0xff], &[("r4", 0x1234)], &[], &[("r3", 0x34)], &[]),
            ("xor r3, r4, 0xffff", vec![0x2a, 0x18, 0xff, 0xff], &[("r4", 0x1234)], &[], &[("r3", 0xedcb)], &[]),
            ("add r1, r2, -0x1", vec![0x01, 0x0c, 0xff, 0xff], &[("r2", 0x10)], &[], &[("r1", 0xf)], &[]),
            ("add r1, r2, 0x12340000", vec![0x01, 0x0a, 0x12, 0x34], &[("r2", 1)], &[], &[("r1", 0x12340001)], &[]),
            ("add r1, r2, 0x123400", vec![0x01, 0x0e, 0x12, 0x34], &[("r2", 0)], &[], &[("r1", 0x123400)], &[]),
            ("add r1, r2, -0x10000", vec![0x01, 0x0e, 0xff, 0x00], &[("r2", 0x20000)], &[], &[("r1", 0x10000)], &[]),
            ("unk.0x7 r1, r2, 0x1", vec![0x39, 0x08, 0x00, 0x01], &[("r1", 0x99), ("r2", 0x10)], &[], &[("r1", 0x99), ("r2", 0x10)], &[]),
        ]);
    }

    #[test]
    fn risc_register_operands() {
        let sleigh = load("examples/risc.slaspec");
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("add r1, r2, r3", vec![0xf9, 0x09, 0x80, 0x00], &[("r2", 0xfffffffe), ("r3", 3)], &[], &[("r1", 1)], &[]),
            ("sub r1, r2, r3", vec![0xf9, 0x09, 0x80, 0x01], &[("r2", 3), ("r3", 5)], &[], &[("r1", 0xfffffffe)], &[]),
            ("or r1, r2, r3", vec![0xf9, 0x09, 0x80, 0x02], &[("r2", 0xff00ff00), ("r3", 0x0ff00ff0)], &[], &[("r1", 0xfff0fff0)], &[]),
            ("and r1, r2, r3", vec![0xf9, 0x09, 0x80, 0x03], &[("r2", 0xff00ff00), ("r3", 0x0ff00ff0)], &[], &[("r1", 0x0f000f00)], &[]),
            ("xor r1, r2, r3", vec![0xf9, 0x09, 0x80, 0x04], &[("r2", 0xff00ff00), ("r3", 0x0ff00ff0)], &[], &[("r1", 0xf0f0f0f0)], &[]),
            ("xor r1, r1, r1", vec![0xf8, 0x88, 0x80, 0x04], &[("r1", 0x1234)], &[], &[("r1", 0)], &[]),
        ]);
    }

    #[test]
    fn vliw_bundles() {
        let sleigh = load("examples/vliw.slaspec");
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("{ add r1, r2, 0x5 }", vec![0xc8, 0x04, 0x40, 0x00, 0x00, 0x00, 0x00, 0x05], &[("r2", 10)], &[], &[("r1", 15)], &[]),
            ("{ sub r1, r2, -0x1 }", vec![0xc8, 0x84, 0x40, 0x00, 0xff, 0xff, 0xff, 0xff], &[("r2", 10)], &[], &[("r1", 11)], &[]),
            ("{ add r1, r2, 0x1 ; sub r3, r4, 0x1 }", vec![0x88, 0x04, 0x51, 0x19, 0x00, 0x00, 0x00, 0x01], &[("r2", 1), ("r4", 1)], &[], &[("r1", 2), ("r3", 0)], &[]),
            ("{ add r1, r2, 0x7 ; sub r3, r4, 0 }", vec![0xa8, 0x04, 0x51, 0x19, 0x00, 0x00, 0x00, 0x07], &[("r2", 1), ("r4", 5)], &[], &[("r1", 8), ("r3", 5)], &[]),
            ("{ add r1, r1, 0x1 ; add r2, r2, 0x1 ; add r3, r3, 0 }", vec![0x58, 0x04, 0x30, 0x10, 0xa0, 0x31, 0x80, 0x01], &[("r1", 1), ("r2", 2), ("r3", 3)], &[], &[("r1", 2), ("r2", 3), ("r3", 3)], &[]),
            ("{ add r1, r2, 0 ; xor r3, r3, 0 ; or r4, r5, 0 ; mov r6, r7 }", vec![0x08, 0x04, 0x54, 0x18, 0xe6, 0x42, 0xc0, 0xc7], &[("r2", 0x11), ("r3", 0x22), ("r5", 0x55), ("r7", 0x77)], &[], &[("r1", 0x11), ("r3", 0x22), ("r4", 0x55), ("r6", 0x77)], &[]),
            ("{ unk.0x0 r0, r0, 0 ; unk.0x0 r0, r0, 0 ; unk.0x0 r0, r0, 0 ; add r8, r9 }", vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x45, 0x09], &[("r8", 1), ("r9", 2)], &[], &[("r8", 3)], &[]),
            ("{ unk.0x0 r0, r0, 0 ; unk.0x0 r0, r0, 0 ; unk.0x0 r0, r0, 0 ; sub r8, r9 }", vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x49, 0x09], &[("r8", 5), ("r9", 2)], &[], &[("r8", 3)], &[]),
            ("{ unk.0x0 r1, r2, 0x5 }", vec![0xc0, 0x04, 0x40, 0x00, 0x00, 0x00, 0x00, 0x05], &[("r1", 0x99), ("r2", 10)], &[], &[("r1", 0x99)], &[]),
            ("{ add r1, r1, 0x1 ; add r2, r1, 0x1 }", vec![0x88, 0x04, 0x30, 0x10, 0x40, 0x00, 0x00, 0x01], &[("r1", 1)], &[], &[("r1", 2), ("r2", 3)], &[]),
        ]);
    }

    #[test]
    fn cisc_register_operands() {
        let sleigh = load("examples/cisc.slaspec");
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("nop", vec![0x00], &[("r1", 5)], &[], &[("r1", 5)], &[]),
            ("mov r1, r2", vec![0x01, 0x0a], &[("r2", 0x12345678)], &[], &[("r1", 0x12345678), ("r2", 0x12345678)], &[]),
            ("mov sp, r0", vec![0x01, 0x38], &[("r0", 0x8000)], &[], &[("sp", 0x8000)], &[]),
            ("add r1, r2", vec![0x02, 0x0a], &[("r1", 3), ("r2", 4), ("ZF", 1)], &[], &[("r1", 7), ("ZF", 0)], &[]),
            ("add r1, r2", vec![0x02, 0x0a], &[("r1", 0xffffffff), ("r2", 1)], &[], &[("r1", 0), ("ZF", 1)], &[]),
            ("sub r1, r2", vec![0x03, 0x0a], &[("r1", 10), ("r2", 3)], &[], &[("r1", 7), ("ZF", 0)], &[]),
            ("sub r1, r2", vec![0x03, 0x0a], &[("r1", 0), ("r2", 1)], &[], &[("r1", 0xffffffff), ("ZF", 0)], &[]),
            ("sub r1, r1", vec![0x03, 0x09], &[("r1", 42)], &[], &[("r1", 0), ("ZF", 1)], &[]),
            ("and r1, r2", vec![0x04, 0x0a], &[("r1", 0xff00ff00), ("r2", 0x0ff00ff0)], &[], &[("r1", 0x0f000f00), ("ZF", 0)], &[]),
            ("or r1, r2", vec![0x05, 0x0a], &[("r1", 0xff00ff00), ("r2", 0x0ff00ff0)], &[], &[("r1", 0xfff0fff0), ("ZF", 0)], &[]),
            ("xor r1, r2", vec![0x06, 0x0a], &[("r1", 0xff00ff00), ("r2", 0x0ff00ff0)], &[], &[("r1", 0xf0f0f0f0), ("ZF", 0)], &[]),
            ("xor r3, r3", vec![0x06, 0x1b], &[("r3", 0x1234)], &[], &[("r3", 0), ("ZF", 1)], &[]),
            ("cmp r1, r2", vec![0x07, 0x0a], &[("r1", 5), ("r2", 5)], &[], &[("r1", 5), ("ZF", 1)], &[]),
            ("cmp r1, r2", vec![0x07, 0x0a], &[("r1", 5), ("r2", 6), ("ZF", 1)], &[], &[("r1", 5), ("ZF", 0)], &[]),
        ]);
    }

    #[test]
    fn cisc_immediate_operands() {
        let sleigh = load("examples/cisc.slaspec");
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("mov r1, #0x12345678", vec![0x01, 0xc8, 0x12, 0x34, 0x56, 0x78], &[], &[], &[("r1", 0x12345678)], &[]),
            ("add r1, #0x1", vec![0x02, 0xc8, 0x00, 0x00, 0x00, 0x01], &[("r1", 1)], &[], &[("r1", 2), ("ZF", 0)], &[]),
            ("cmp r1, #0x2a", vec![0x07, 0xc8, 0x00, 0x00, 0x00, 0x2a], &[("r1", 0x2a)], &[], &[("ZF", 1)], &[]),
        ]);
    }

    #[test]
    fn cisc_memory_loads() {
        let sleigh = load("examples/cisc.slaspec");
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("mov r1, [r2]", vec![0x01, 0x4a], &[("r2", 0x2000)], &[(0x2000, 0xdeadbeef)], &[("r1", 0xdeadbeef)], &[]),
            ("mov r1, [r2+0x4]", vec![0x01, 0x8a, 0x04], &[("r2", 0x2000)], &[(0x2004, 0xdeadbeef)], &[("r1", 0xdeadbeef)], &[]),
            ("mov r1, [r2+-0x4]", vec![0x01, 0x8a, 0xfc], &[("r2", 0x2000)], &[(0x1ffc, 0xdeadbeef)], &[("r1", 0xdeadbeef)], &[]),
            ("add r1, [r2]", vec![0x02, 0x4a], &[("r1", 1), ("r2", 0x2000)], &[(0x2000, 0x41)], &[("r1", 0x42)], &[]),
        ]);
    }

    #[test]
    fn cisc_memory_stores() {
        let sleigh = load("examples/cisc.slaspec");
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("mov [r2], r1", vec![0x08, 0x4a], &[("r1", 0xcafebabe), ("r2", 0x2000)], &[], &[], &[(0x2000, 0xcafebabe)]),
            ("mov [r2+0x4], r1", vec![0x08, 0x8a, 0x04], &[("r1", 0xcafebabe), ("r2", 0x2000)], &[], &[], &[(0x2004, 0xcafebabe)]),
            ("mov [r2+-0x4], r1", vec![0x08, 0x8a, 0xfc], &[("r1", 0xcafebabe), ("r2", 0x2000)], &[], &[], &[(0x1ffc, 0xcafebabe)]),
        ]);
    }

    #[test]
    fn cisc_branches() {
        let sleigh = load("examples/cisc.slaspec");
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("jmp 0x2000", vec![0x10, 0x00, 0x00, 0x20, 0x00], &[], &[], &[("pc", 0x2000)], &[]),
            ("jz 0x1012", vec![0x11, 0x10], &[("ZF", 1)], &[], &[("pc", 0x1012)], &[]),
            ("jz 0x1012", vec![0x11, 0x10], &[("ZF", 0)], &[], &[("pc", 0x1002)], &[]),
            ("jz 0x1000", vec![0x11, 0xfe], &[("ZF", 1)], &[], &[("pc", 0x1000)], &[]),
            ("jnz 0x1012", vec![0x12, 0x10], &[("ZF", 0)], &[], &[("pc", 0x1012)], &[]),
            ("jnz 0x1012", vec![0x12, 0x10], &[("ZF", 1)], &[], &[("pc", 0x1002)], &[]),
        ]);
    }

    #[test]
    fn cisc_store_then_load() {
        let sleigh = load("examples/cisc.slaspec");
        #[rustfmt::skip]
        let mut cpu = new_cpu(&sleigh, &[
            0x08, 0x8a, 0x04, // mov [r2+0x4], r1
            0x01, 0x9a, 0x04, // mov r3, [r2+0x4]
        ]);
        set_reg(&mut cpu, "r1", 0x11223344);
        set_reg(&mut cpu, "r2", 0x2000);
        cpu.step().unwrap();
        cpu.step().unwrap();
        assert_eq!(get_reg(&mut cpu, "r3"), 0x11223344);
        assert_eq!(get_mem(&mut cpu, 0x2004), 0x11223344);
    }

    #[test]
    fn cisc_counting_loop() {
        let sleigh = load("examples/cisc.slaspec");
        #[rustfmt::skip]
        let mut cpu = new_cpu(&sleigh, &[
            0x06, 0x09, // 0x1000: xor r1, r1
            0x02, 0x0a, // 0x1002: add r1, r2
            0x03, 0x13, // 0x1004: sub r2, r3
            0x12, 0xfa, // 0x1006: jnz 0x1002
            0x00,       // 0x1008: nop
        ]);
        set_reg(&mut cpu, "r1", 0xffff);
        set_reg(&mut cpu, "r2", 5);
        set_reg(&mut cpu, "r3", 1);

        let mut steps = 0;
        while cpu.state.pc != 0x1008 {
            assert!(
                steps < 100,
                "loop did not terminate, pc = {:#x}",
                cpu.state.pc
            );
            cpu.step().unwrap();
            steps += 1;
        }

        assert_eq!(steps, 16);
        assert_eq!(get_reg(&mut cpu, "r1"), 5 + 4 + 3 + 2 + 1);
        assert_eq!(get_reg(&mut cpu, "r2"), 0);
        assert_eq!(get_reg(&mut cpu, "ZF"), 1);
    }

    #[test]
    fn cisc_user_ops() {
        let sleigh = load("examples/cisc.slaspec");
        #[rustfmt::skip]
        let mut cpu = new_cpu(&sleigh, &[
            0x20, 0x18, // out r3
            0x21, 0x08, // in r1
        ]);
        let output = std::rc::Rc::new(std::cell::RefCell::new(vec![]));
        let out = output.clone();
        cpu.register_user_op("out", move |_state, args| {
            out.borrow_mut().extend_from_slice(args);
            Ok(None)
        });
        cpu.register_user_op("in", |_state, args| {
            assert!(args.is_empty());
            Ok(Some(0x42))
        });
        set_reg(&mut cpu, "r3", 0x1234);

        cpu.step().unwrap();
        cpu.step().unwrap();
        assert_eq!(*output.borrow(), vec![0x1234]);
        assert_eq!(get_reg(&mut cpu, "r1"), 0x42);
        assert_eq!(cpu.state.pc, BASE + 4);
    }

    #[test]
    fn cisc_unregistered_user_op() {
        let sleigh = load("examples/cisc.slaspec");
        let mut cpu = new_cpu(&sleigh, &[0x20, 0x18]);
        let err = cpu.step().unwrap_err();
        assert_eq!(
            format!("{:#}", err),
            "0x00001000: out r3: no handler for user op \"out\""
        );
        assert_eq!(cpu.state.pc, BASE);
    }

    /// Partial writes through a pointer keep the bytes they do not select, in either byte order
    fn assert_partial_writes(sleigh: &Sleigh) {
        let regs: &[(&str, u64)] = &[("r0", 0xaabbccdd), ("r1", 0x2000)];
        let mem: &[(u64, u32)] = &[(0x2000, 0x11223344), (0x2004, 0x55667788)];
        #[rustfmt::skip]
        assert_executes(sleigh, &[
            ("sth [r1], r0", vec![0x03, 0x01], regs, mem, &[], &[(0x2000, 0x1122ccdd), (0x2004, 0x55667788)]),
            ("stbits [r1], r0", vec![0x04, 0x01], regs, mem, &[], &[(0x2000, 0x11223dd4), (0x2004, 0x55667788)]),
            ("stmid [r1], r0", vec![0x05, 0x01], regs, mem, &[], &[(0x2000, 0x11ccdd44), (0x2004, 0x55667788)]),
        ]);
    }

    #[test]
    fn partial_writes_little_endian() {
        assert_partial_writes(&load("examples/pcode.slaspec"));
    }

    #[test]
    fn partial_writes_big_endian() {
        let spec = std::fs::read_to_string("examples/pcode.slaspec")
            .unwrap()
            .replace("define endian=little;", "define endian=big;");
        let path =
            std::env::temp_dir().join(format!("sleigher-pcode-be-{}.slaspec", std::process::id()));
        // Removes the spec even if loading it panics
        struct TempFile(std::path::PathBuf);
        impl Drop for TempFile {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        std::fs::write(&path, spec).unwrap();
        let file = TempFile(path);
        let sleigh = load(&file.0);
        drop(file);
        assert_eq!(sleigh.endian(), Endian::Big);
        assert_partial_writes(&sleigh);
    }

    #[test]
    fn belt_drops() {
        let sleigh = load("examples/belt.slaspec");
        // Every result drops onto b0 and pushes the belt back, b15 falls off
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("con 0x5", vec![0x04, 0x00, 0x00, 0x05], &[("b0", 1), ("b1", 2), ("b14", 14), ("b15", 15)], &[], &[("b0", 5), ("b1", 1), ("b2", 2), ("b15", 14)], &[]),
            ("con -0x1", vec![0x04, 0x03, 0xff, 0xff], &[], &[], &[("b0", 0xffffffff)], &[]),
            ("conw 0xedb88320", vec![0x08, 0x00, 0x00, 0x00, 0xed, 0xb8, 0x83, 0x20], &[("b0", 1)], &[], &[("b0", 0xedb88320), ("b1", 1)], &[]),
            ("add b0, b1", vec![0x0c, 0x04, 0x00, 0x00], &[("b0", 3), ("b1", 4)], &[], &[("b0", 7), ("b1", 3), ("b2", 4)], &[]),
            ("sub b1, b0", vec![0x10, 0x40, 0x00, 0x00], &[("b0", 3), ("b1", 10)], &[], &[("b0", 7), ("b1", 3), ("b2", 10)], &[]),
            ("mul b2, b3", vec![0x14, 0x8c, 0x00, 0x00], &[("b2", 6), ("b3", 7)], &[], &[("b0", 42), ("b3", 6), ("b4", 7)], &[]),
            ("xor b0, b1", vec![0x20, 0x04, 0x00, 0x00], &[("b0", 0xff00), ("b1", 0x0ff0)], &[], &[("b0", 0xf0f0)], &[]),
            ("shl b0, b1", vec![0x28, 0x04, 0x00, 0x00], &[("b0", 0x3), ("b1", 4)], &[], &[("b0", 0x30)], &[]),
            ("eql b0, b1", vec![0x2c, 0x04, 0x00, 0x00], &[("b0", 5), ("b1", 6)], &[], &[("b0", 0), ("b1", 5), ("b2", 6)], &[]),
            ("addi b1, -0x1", vec![0x30, 0x43, 0xff, 0xff], &[("b0", 9), ("b1", 5)], &[], &[("b0", 4), ("b1", 9), ("b2", 5)], &[]),
            ("andi b0, 0x1", vec![0x34, 0x00, 0x00, 0x01], &[("b0", 0x7)], &[], &[("b0", 1), ("b1", 0x7)], &[]),
            ("xori b0, -0x1", vec![0x38, 0x03, 0xff, 0xff], &[("b0", 0x0f0f0f0f)], &[], &[("b0", 0xf0f0f0f0)], &[]),
            ("shri b0, 0x4", vec![0x3c, 0x00, 0x00, 0x04], &[("b0", 0x80000000)], &[], &[("b0", 0x08000000)], &[]),
            ("mov b15", vec![0x53, 0xc0, 0x00, 0x00], &[("b14", 0x66), ("b15", 0x77)], &[], &[("b0", 0x77), ("b15", 0x66)], &[]),
            // conform drops copies so the listed values end up at the front, in order
            ("conform b2, b0", vec![0x54, 0x80, 0x00, 0x02], &[("b0", 10), ("b1", 11), ("b2", 12)], &[], &[("b0", 12), ("b1", 10), ("b2", 10), ("b3", 11), ("b4", 12)], &[]),
            ("conform b3, b0, b7, b2, b1", vec![0x54, 0xc1, 0xc8, 0x45], &[("b0", 10), ("b1", 11), ("b2", 12), ("b3", 13), ("b7", 17)], &[], &[("b0", 13), ("b1", 10), ("b2", 17), ("b3", 12), ("b4", 11), ("b5", 10)], &[]),
            // Two drops: the quotient, then the remainder
            ("divu b0, b1", vec![0x58, 0x04, 0x00, 0x00], &[("b0", 17), ("b1", 5)], &[], &[("b0", 2), ("b1", 3), ("b2", 17), ("b3", 5)], &[]),
            ("nop", vec![0x00, 0x00, 0x00, 0x00], &[("b0", 1)], &[], &[("b0", 1)], &[]),
        ]);
    }

    #[test]
    fn belt_memory_and_branches() {
        let sleigh = load("examples/belt.slaspec");
        // Stores and branches drop nothing
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("ld b0", vec![0x40, 0x00, 0x00, 0x00], &[("b0", 0x2000)], &[(0x2000, 0xdeadbeef)], &[("b0", 0xdeadbeef), ("b1", 0x2000)], &[]),
            ("ldb b0", vec![0x44, 0x00, 0x00, 0x00], &[("b0", 0x2001)], &[(0x2000, 0xdeadbeef)], &[("b0", 0xad), ("b1", 0x2001)], &[]),
            ("st b1, b0", vec![0x48, 0x40, 0x00, 0x00], &[("b0", 0xcafebabe), ("b1", 0x2000)], &[], &[("b0", 0xcafebabe), ("b1", 0x2000)], &[(0x2000, 0xcafebabe)]),
            ("br b0, 0x1014", vec![0x5c, 0x00, 0x00, 0x04], &[("b0", 1)], &[], &[("pc", 0x1014)], &[]),
            ("br b0, 0x1014", vec![0x5c, 0x00, 0x00, 0x04], &[("b0", 0)], &[], &[], &[]),
            ("brz b0, 0x1014", vec![0x60, 0x00, 0x00, 0x04], &[("b0", 0)], &[], &[("pc", 0x1014)], &[]),
            ("brz b0, 0x1014", vec![0x60, 0x00, 0x00, 0x04], &[("b0", 1)], &[], &[], &[]),
            ("jmp 0xffc", vec![0x64, 0x03, 0xff, 0xfe], &[], &[], &[("pc", 0xffc)], &[]),
        ]);
    }

    #[test]
    fn belt_sum_loop() {
        let source = "
                    con 0x4         // n
                    con 0x0         // acc; the loop keeps b0 = acc, b1 = n
            loop:   add b0, b1      // acc + n
                    addi b2, -0x1   // n - 1
                    conform b1, b0
                    br b1, loop
                    out b0
            end:    nop
        ";
        let assembler = crate::assembler::InstructionAssembler::new(load("examples/belt.slaspec"));
        let program = assembler.assemble_program(source, BASE).unwrap();

        let sleigh = load("examples/belt.slaspec");
        let mut cpu = new_cpu(&sleigh, &program.bytes);
        let output = std::rc::Rc::new(std::cell::RefCell::new(vec![]));
        let out = output.clone();
        cpu.register_user_op("out", move |_state, args| {
            out.borrow_mut().extend_from_slice(args);
            Ok(None)
        });

        let mut steps = 0;
        while cpu.state.pc != program.labels["end"] {
            assert!(
                steps < 100,
                "loop did not terminate, pc = {:#x}",
                cpu.state.pc
            );
            cpu.step().unwrap();
            steps += 1;
        }
        assert_eq!(*output.borrow(), vec![4 + 3 + 2 + 1]);
        assert_eq!(steps, 2 + 4 * 4 + 1);
    }

    #[test]
    fn i8051_data_and_arithmetic() {
        let sleigh = load("examples/8051.slaspec");
        // R0-R7 and the SFRs live in data spaces, so direct addresses alias them.
        // PSW: CY = 0x80, OV = 0x04
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("mov A, #0x42", vec![0x74, 0x42], &[], &[], &[("ACC", 0x42)], &[]),
            ("mov ACC, R3", vec![0x8b, 0xe0], &[("R3", 7)], &[], &[("ACC", 7)], &[]),
            ("mov 0x5, R3", vec![0x8b, 0x05], &[("R3", 7)], &[], &[("R5", 7)], &[]),
            ("mov @R0, A", vec![0xf6], &[("R0", 3), ("ACC", 9)], &[], &[("R3", 9)], &[]),
            ("mov 0x4, 0x2", vec![0x85, 0x02, 0x04], &[("R2", 0x55)], &[], &[("R4", 0x55)], &[]),
            ("mov DPTR, #0x1234", vec![0x90, 0x12, 0x34], &[], &[], &[("DPH", 0x12), ("DPL", 0x34)], &[]),
            ("inc DPTR", vec![0xa3], &[("DPTR", 0x12ff)], &[], &[("DPH", 0x13), ("DPL", 0)], &[]),
            ("xch A, R7", vec![0xcf], &[("ACC", 1), ("R7", 2)], &[], &[("ACC", 2), ("R7", 1)], &[]),
            ("add A, R1", vec![0x29], &[("ACC", 0xf0), ("R1", 0x20)], &[], &[("ACC", 0x10), ("PSW", 0x80)], &[]),
            ("add A, #0x40", vec![0x24, 0x40], &[("ACC", 0x40)], &[], &[("ACC", 0x80), ("PSW", 0x04)], &[]),
            ("addc A, #0x1", vec![0x34, 0x01], &[("ACC", 0xff), ("PSW", 0x80)], &[], &[("ACC", 0x01), ("PSW", 0x80)], &[]),
            ("subb A, R0", vec![0x98], &[("ACC", 1), ("R0", 1), ("PSW", 0x80)], &[], &[("ACC", 0xff), ("PSW", 0x80)], &[]),
            ("mul AB", vec![0xa4], &[("ACC", 0x40), ("B", 0x10)], &[], &[("ACC", 0), ("B", 4), ("PSW", 0x04)], &[]),
            ("div AB", vec![0x84], &[("ACC", 17), ("B", 5)], &[], &[("ACC", 3), ("B", 2), ("PSW", 0)], &[]),
            ("div AB", vec![0x84], &[("ACC", 17), ("B", 0)], &[], &[("ACC", 17), ("PSW", 0x04)], &[]),
            ("rrc A", vec![0x13], &[("ACC", 0x03)], &[], &[("ACC", 0x01), ("PSW", 0x80)], &[]),
            ("rlc A", vec![0x33], &[("ACC", 0x80), ("PSW", 0x80)], &[], &[("ACC", 0x01), ("PSW", 0x80)], &[]),
            ("swap A", vec![0xc4], &[("ACC", 0x12)], &[], &[("ACC", 0x21)], &[]),
            ("cpl C", vec![0xb3], &[("PSW", 0x04)], &[], &[("PSW", 0x84)], &[]),
        ]);
    }

    #[test]
    fn i8051_branches_and_stack() {
        let sleigh = load("examples/8051.slaspec");
        // SP points into R0-R7 so the pushed bytes can be checked by name
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("djnz R2, 0x1000", vec![0xda, 0xfe], &[("R2", 2)], &[], &[("R2", 1), ("pc", 0x1000)], &[]),
            ("djnz R2, 0x1000", vec![0xda, 0xfe], &[("R2", 1)], &[], &[("R2", 0)], &[]),
            ("cjne A, #0x5, 0x1010", vec![0xb4, 0x05, 0x0d], &[("ACC", 3)], &[], &[("PSW", 0x80), ("pc", 0x1010)], &[]),
            ("cjne A, #0x5, 0x1010", vec![0xb4, 0x05, 0x0d], &[("ACC", 5)], &[], &[("PSW", 0)], &[]),
            ("jc 0x1010", vec![0x40, 0x0e], &[("PSW", 0x80)], &[], &[("pc", 0x1010)], &[]),
            ("jnz 0x1010", vec![0x70, 0x0e], &[("ACC", 0)], &[], &[], &[]),
            ("ajmp 0x17ff", vec![0xe1, 0xff], &[], &[], &[("pc", 0x17ff)], &[]),
            ("ljmp 0x2345", vec![0x02, 0x23, 0x45], &[], &[], &[("pc", 0x2345)], &[]),
            ("acall 0x1234", vec![0x51, 0x34], &[("SP", 5)], &[], &[("SP", 7), ("R6", 0x02), ("R7", 0x10), ("pc", 0x1234)], &[]),
            ("lcall 0x2345", vec![0x12, 0x23, 0x45], &[("SP", 5)], &[], &[("SP", 7), ("R6", 0x03), ("R7", 0x10), ("pc", 0x2345)], &[]),
            ("ret", vec![0x22], &[("SP", 7), ("R6", 0x34), ("R7", 0x12)], &[], &[("SP", 5), ("pc", 0x1234)], &[]),
            ("push B", vec![0xc0, 0xf0], &[("SP", 1), ("B", 0x99)], &[], &[("SP", 2), ("R2", 0x99)], &[]),
            ("pop DPL", vec![0xd0, 0x82], &[("SP", 3), ("R3", 0x77)], &[], &[("SP", 2), ("DPL", 0x77)], &[]),
        ]);
    }

    /// Assemble `examples/crc32/<arch>.s`, run it on several inputs and compare the result with
    /// crc32fast. The program gets the data address in `data_reg` and its length in `len_reg`,
    /// leaves the crc in `crc_reg` and finishes at its `end` label.
    fn assert_crc32(arch: &str, data_reg: &str, len_reg: &str, crc_reg: &str) {
        const DATA: u64 = 0x8000;
        let spec = format!("examples/{}.slaspec", arch);
        let source = std::fs::read_to_string(format!("examples/crc32/{}.s", arch)).unwrap();
        let assembler = crate::assembler::InstructionAssembler::new(load(&spec));
        let program = assembler.assemble_program(&source, BASE).unwrap();
        let end = program.labels["end"];

        let mut random = vec![];
        let mut seed = 0x2545f4914f6cdd1du64;
        for _ in 0..64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            random.push(seed as u8);
        }
        let inputs: [&[u8]; 4] = [b"", b"a", b"123456789", &random];

        let sleigh = load(&spec);
        for data in inputs {
            let mut cpu = new_cpu(&sleigh, &program.bytes);
            let data_ref = Ref(cpu.default_space(), data.len(), Address(DATA));
            if !data.is_empty() {
                cpu.state.write_ref(data_ref, data).unwrap();
            }
            set_reg(&mut cpu, data_reg, DATA);
            set_reg(&mut cpu, len_reg, data.len() as u64);

            let mut steps = 0;
            while cpu.state.pc != end {
                assert!(steps < 100_000, "{}: no end after {} steps", arch, steps);
                cpu.step()
                    .unwrap_or_else(|err| panic!("{}: {:#}", arch, err));
                steps += 1;
            }
            assert_eq!(
                get_reg(&mut cpu, crc_reg),
                crc32fast::hash(data) as u64,
                "{}: crc32 of {:02x?}",
                arch,
                data
            );
        }
    }

    #[test]
    fn crc32_cisc() {
        assert_crc32("cisc", "r1", "r2", "r0");
    }

    #[test]
    fn crc32_risc() {
        assert_crc32("risc", "r1", "r2", "r0");
    }

    #[test]
    fn crc32_vliw() {
        assert_crc32("vliw", "r1", "r2", "r3");
    }

    #[test]
    fn crc32_belt() {
        assert_crc32("belt", "b0", "b1", "b0");
    }

    #[test]
    fn crc32_8051() {
        assert_crc32("8051", "DPTR", "R2", "R4R5R6R7");
    }

    /// A little endian 32-bit register, the context example is little endian
    fn get_reg_le(cpu: &mut Cpu, name: &str) -> u32 {
        let mut bytes = [0u8; 4];
        cpu.state.read_ref(reg_ref(cpu, name), &mut bytes).unwrap();
        u32::from_le_bytes(bytes)
    }

    #[test]
    fn context_flow() {
        let sleigh = load("examples/context.slaspec");
        let id = |name| Context::id(&sleigh, name).unwrap();
        #[rustfmt::skip]
        let program = [
            0x05, 0x00, 0x11, 0x01, // add r1, r1, 0x5
            0x00, 0x00, 0x00, 0x02, // mode1
            0x02, 0x00, 0x11, 0x01, // sub r1, r1, 0x2
            0x02, 0x00, 0x00, 0x04, // pfx 0x2
            0x00, 0x00, 0x11, 0x05, // shl r1, r1, 0x2
            0x00, 0x00, 0x11, 0x05, // shl r1, r1, 0x0
        ];
        let mut cpu = new_cpu(&sleigh, &program);
        let mut r1 = vec![];
        for _ in 0..6 {
            cpu.step().unwrap();
            r1.push(get_reg_le(&mut cpu, "r1"));
        }
        // mode flows on after mode1, shift only reaches the instruction after pfx
        assert_eq!(r1, vec![5, 5, 3, 3, 12, 12]);
        assert_eq!(cpu.context.next.get(&sleigh, id("mode")), 1);
        assert_eq!(cpu.context.next.get(&sleigh, id("shift")), 0);
    }

    #[test]
    fn context_initial() {
        let sleigh = load("examples/context.slaspec");
        // add r1, r1, 0x5 decodes as sub in mode 1
        let mut cpu = new_cpu(&sleigh, &[0x05, 0x00, 0x11, 0x01]);
        cpu.context
            .next
            .set(&sleigh, Context::id(&sleigh, "mode").unwrap(), 1);
        cpu.step().unwrap();
        assert_eq!(get_reg_le(&mut cpu, "r1"), 5u32.wrapping_neg());

        // A noflow value set up front reaches only the first instruction
        let mut cpu = new_cpu(&sleigh, &[0x00, 0x00, 0x11, 0x05, 0x00, 0x00, 0x11, 0x05]);
        cpu.context
            .next
            .set(&sleigh, Context::id(&sleigh, "shift").unwrap(), 2);
        cpu.state
            .write_ref(reg_ref(&cpu, "r1"), &1u32.to_le_bytes())
            .unwrap();
        cpu.step().unwrap();
        cpu.step().unwrap();
        assert_eq!(get_reg_le(&mut cpu, "r1"), 4);
    }

    #[test]
    fn delay_slot_branches() {
        let sleigh = load("examples/delay.slaspec");
        // The delay slot runs before the branch takes effect, but after its operands are read
        #[rustfmt::skip]
        assert_executes(&sleigh, &[
            ("beq r1, r2, 0x1010", vec![0x10, 0x22, 0x00, 0x03, 0x20, 0x63, 0x00, 0x01], &[("r1", 5), ("r2", 5), ("r3", 1)], &[], &[("r3", 2), ("pc", 0x1010)], &[]),
            ("beq r1, r2, 0x1010", vec![0x10, 0x22, 0x00, 0x03, 0x20, 0x63, 0x00, 0x01], &[("r1", 1), ("r2", 2), ("r3", 1)], &[], &[("r3", 2)], &[]),
            ("beq r1, r2, 0x1010", vec![0x10, 0x22, 0x00, 0x03, 0x20, 0x21, 0x00, 0x01], &[("r1", 5), ("r2", 5)], &[], &[("r1", 6), ("pc", 0x1010)], &[]),
            ("bne r1, r0, 0x1000", vec![0x14, 0x20, 0xff, 0xff, 0x20, 0x21, 0xff, 0xff], &[("r1", 1)], &[], &[("r1", 0), ("pc", 0x1000)], &[]),
            ("j 0x2000", vec![0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00], &[], &[], &[("pc", 0x2000)], &[]),
            ("jal 0x2000", vec![0x0c, 0x00, 0x08, 0x00, 0x04, 0x03, 0x00, 0x00], &[], &[], &[("ra", 0x1008), ("r3", 1), ("pc", 0x2000)], &[]),
            ("jr ra", vec![0x18, 0xe0, 0x00, 0x00, 0x20, 0xe7, 0x00, 0x04], &[("ra", 0x3000)], &[], &[("ra", 0x3004), ("pc", 0x3000)], &[]),
            ("beql r1, r2, 0x1010", vec![0x50, 0x22, 0x00, 0x03, 0x20, 0x63, 0x00, 0x01], &[("r1", 5), ("r2", 5), ("r3", 1)], &[], &[("r3", 2), ("pc", 0x1010)], &[]),
            ("beql r1, r2, 0x1010", vec![0x50, 0x22, 0x00, 0x03, 0x20, 0x63, 0x00, 0x01], &[("r1", 1), ("r2", 2), ("r3", 1)], &[], &[("r3", 1)], &[]),
            ("beq r0, r0, 0x1010", vec![0x10, 0x00, 0x00, 0x03, 0x28, 0x22, 0x18, 0x00], &[("r1", 3), ("r2", 7)], &[], &[("r3", 7), ("pc", 0x1010)], &[]),
            ("beq r0, r0, 0x1010", vec![0x10, 0x00, 0x00, 0x03, 0x28, 0x22, 0x18, 0x00], &[("r1", 7), ("r2", 3)], &[], &[("r3", 7), ("pc", 0x1010)], &[]),
        ]);
    }

    #[test]
    fn delay_slot_runs_once() {
        let sleigh = load("examples/delay.slaspec");
        #[rustfmt::skip]
        let mut cpu = new_cpu(&sleigh, &[
            0x10, 0x00, 0x00, 0x02, // 0x1000: beq r0, r0, 0x100c
            0x04, 0x03, 0x00, 0x00, // 0x1004: slot r3, 0x1
            0x04, 0x05, 0x00, 0x00, // 0x1008: slot r5, 0x0
            0x04, 0x04, 0x00, 0x00, // 0x100c: slot r4, 0x0
        ]);
        set_reg(&mut cpu, "r3", 0x99);
        set_reg(&mut cpu, "r4", 0x99);
        set_reg(&mut cpu, "r5", 0x99);
        cpu.step().unwrap();
        assert_eq!(cpu.state.pc, 0x100c);
        assert_eq!(get_reg(&mut cpu, "r3"), 1);
        // The commit to the delay slot does not reach the branch target
        cpu.step().unwrap();
        assert_eq!(cpu.state.pc, 0x1010);
        assert_eq!(get_reg(&mut cpu, "r4"), 0);
        assert_eq!(get_reg(&mut cpu, "r5"), 0x99);
    }

    #[test]
    fn delay_slot_sum_loop() {
        let source = "
                    addi r1, r0, 0x4    // n
                    addi r2, r0, 0x0    // acc
            loop:   add r2, r2, r1
                    bne r1, r0, loop
                    addi r1, r1, -0x1   // delay slot, also on the way out
            end:
        ";
        let assembler = crate::assembler::InstructionAssembler::new(load("examples/delay.slaspec"));
        let program = assembler.assemble_program(source, BASE).unwrap();

        let sleigh = load("examples/delay.slaspec");
        let mut cpu = new_cpu(&sleigh, &program.bytes);
        let mut steps = 0;
        while cpu.state.pc != program.labels["end"] {
            assert!(
                steps < 100,
                "loop did not terminate, pc = {:#x}",
                cpu.state.pc
            );
            cpu.step().unwrap();
            steps += 1;
        }
        assert_eq!(get_reg(&mut cpu, "r2"), 4 + 3 + 2 + 1);
        assert_eq!(get_reg(&mut cpu, "r1"), 0xffffffff);
        // Each bne step runs its delay slot, which is never stepped on its own
        assert_eq!(steps, 2 + 5 * 2);
    }
}
