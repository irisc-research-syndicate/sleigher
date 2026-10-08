use std::collections::HashMap;

use anyhow::{anyhow, bail, Context as _, Result};

use sleigh_rs::{Endian, Sleigh, SpaceId, UserFunctionId};

use crate::disassembler::{Context, Disassembler};
use crate::pcode::{size_mask, OpCode, PcodeOp, Varnode, VarnodeSpace};
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
    pub max_instruction_len: usize,
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
        Self {
            sleigh,
            disassembler: Disassembler::new(sleigh),
            state,
            max_instruction_len,
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
        let mut instruction_bytes = vec![0u8; self.max_instruction_len];
        self.fetch_instruction(&mut instruction_bytes)?;

        let instruction =
            self.disassembler
                .disassemble(self.state.pc, Context, &instruction_bytes)?;
        log::debug!(
            "Executing {:#010x}: {}",
            instruction.inst_start,
            instruction
        );
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
            .execute(ops, instruction.inst_next)
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
    let shift = 64 - 8 * size.min(8);
    ((value << shift) as i64) >> shift
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
    fn target(&mut self, target: &Varnode) -> Flow {
        if target.is_const() {
            Flow::Relative(target.offset as i64)
        } else {
            Flow::Address(target.offset)
        }
    }

    fn execute_op(&mut self, op: &PcodeOp) -> Result<Flow> {
        use OpCode::*;
        let flow = match op.opcode {
            Branch | Call => self.target(&op.inputs[0]),
            CBranch => {
                if self.input(op, 1)? != 0 {
                    self.target(&op.inputs[0])
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

    fn set_reg(cpu: &mut Cpu, name: &str, value: u64) {
        let reg = reg_ref(cpu, name);
        cpu.state.write_ref(reg, &value.to_be_bytes()).unwrap();
    }

    fn get_reg(cpu: &mut Cpu, name: &str) -> u64 {
        let reg = reg_ref(cpu, name);
        let mut bytes = [0u8; 8];
        cpu.state.read_ref(reg, &mut bytes).unwrap();
        u64::from_be_bytes(bytes)
    }

    fn set_mem(cpu: &mut Cpu, address: u64, value: u32) {
        let mem = Ref(cpu.default_space(), 4, Address(address));
        cpu.state.write_ref(mem, &value.to_be_bytes()).unwrap();
    }

    fn get_mem(cpu: &mut Cpu, address: u64) -> u32 {
        let mem = Ref(cpu.default_space(), 4, Address(address));
        cpu.state.read_ref_u32be(mem).unwrap()
    }

    /// Execute each instruction once from `BASE` and compare registers, memory and pc.
    /// `pc` may be listed as an expected register, otherwise it must point past the instruction.
    /// The instruction bytes must also disassemble to `asm`, which keeps the encodings honest.
    fn assert_executes(sleigh: &Sleigh, tests: &[Case]) {
        let mut failures = vec![];
        for (asm, program, regs_in, mem_in, regs_out, mem_out) in tests.iter() {
            let mut cpu = new_cpu(sleigh, program);
            match cpu.disassembler.disassemble(BASE, Context, program) {
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
}
