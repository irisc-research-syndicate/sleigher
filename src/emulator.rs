use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context as _, Result};

use sleigh_rs::execution::{
    Assignment, Binary, Block, BlockId, Build, CpuBranch, DynamicValueType, Export, Expr,
    ExprElement, ExprValue, LocalGoto, Statement, UserCall, VariableId,
};
use sleigh_rs::{
    user_function::UserFunction, AttachVarnodeId, Sleigh, SpaceId, TableId, TokenFieldId,
};

use crate::disassembler::{Context, DisassembledTable, Disassembler};
use crate::space::{HashSpace, MemoryRegion};
use crate::value::{Address, Ref, Value, Var};

/// Bytes fetched per step when the instruction length is unbounded (recursive patterns)
const FALLBACK_INSTRUCTION_LEN: usize = 16;

pub struct Cpu<'sleigh> {
    pub sleigh: &'sleigh Sleigh,
    pub disassembler: Disassembler<'sleigh>,
    pub state: State,
    pub max_instruction_len: usize,
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
        }
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

        let mut table_executor = TableExecutor::new(&instruction.table);
        let (_export, pc) = table_executor.execute(&mut self.state)?;
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

    pub fn write_ref(&mut self, referance: Ref, data: &[u8]) -> Result<()> {
        log::trace!("Writing {} <- {:02x?}", referance, data);
        assert!(data.len() >= referance.1);
        let data = &data[data.len() - referance.1..];
        self.spaces
            .entry(referance.0)
            .or_insert_with(|| Box::new(HashSpace::new()))
            .write(referance.2, data)
    }

    pub fn read_ref(&mut self, referance: Ref, data: &mut [u8]) -> Result<()> {
        let len = data.len();
        assert!(len >= referance.1);
        for byte in &mut *data {
            *byte = 0;
        }
        let data = &mut data[len - referance.1..];
        self.spaces
            .entry(referance.0)
            .or_insert_with(|| Box::new(HashSpace::new()))
            .read(referance.2, data)?;
        log::trace!("Reading {} -> {:02x?}", referance, data);
        Ok(())
    }

    pub fn read_ref_u32be(&mut self, referance: Ref) -> Result<u32> {
        let mut bytes = [0u8; 4];
        self.read_ref(referance, &mut bytes)?;
        Ok(u32::from_be_bytes(bytes))
    }

    fn get_u64(&mut self, value: Value) -> Result<u64> {
        Ok(match value {
            Value::Int(x) => x,
            Value::Ref(referance) => {
                let mut data = [0u8; 8];
                self.read_ref(referance, &mut data)?;
                u64::from_be_bytes(data)
            }
        })
    }

    fn user_call(&mut self, _function: &UserFunction, _params: Vec<Value>) -> Result<Value> {
        todo!();
    }
}

pub struct TableExecutor<'st> {
    table: &'st DisassembledTable<'st>,
    variables: HashMap<VariableId, Value>,
    built: HashSet<TableId>,
    exports: HashMap<TableId, Value>,
    export: Option<Value>,
}

pub enum ControlFlow {
    Goto(Option<BlockId>),
    Branch(u64),
}

impl<'st> TableExecutor<'st> {
    pub fn new(table: &'st DisassembledTable<'st>) -> Self {
        Self {
            table,
            variables: HashMap::new(),
            built: HashSet::new(),
            exports: HashMap::new(),
            export: None,
        }
    }

    pub fn execute(&mut self, state: &mut State) -> Result<(Option<Value>, u64)> {
        let branch = self.run(state)?;
        Ok((self.export, branch.unwrap_or(self.table.inst_next)))
    }

    /// Run the constructor's semantics, returning the destination if it branched
    fn run(&mut self, state: &mut State) -> Result<Option<u64>> {
        log::trace!("Executing table {}", self.table.table.name());

        let Some(execution) = &self.table.constructor.execution else {
            log::warn!("Constructor has no execution(check sleigh-rs!?)");
            return Ok(None);
        };

        // Subtables without an explicit `build` are built before the constructor's own semantics
        let explicit_builds = execution
            .blocks()
            .iter()
            .flat_map(|block| block.statements.iter())
            .filter_map(|stmt| match stmt {
                Statement::Build(build) => Some(build.table),
                _ => None,
            })
            .collect::<HashSet<_>>();
        for pattern_block in self.table.constructor.pattern.blocks() {
            for produced_table in pattern_block.tables() {
                if explicit_builds.contains(&produced_table.table)
                    || !self.table.tables.contains_key(&produced_table.table)
                {
                    continue;
                }
                if let Some(pc) = self.build_table(state, produced_table.table)? {
                    return Ok(Some(pc));
                }
            }
        }

        let mut next_block = Some(execution.entry_block);
        while let Some(block_id) = next_block.take() {
            match self.execute_block(state, execution.block(block_id))? {
                ControlFlow::Goto(block_id) => next_block = block_id,
                ControlFlow::Branch(pc) => return Ok(Some(pc)),
            };
        }
        Ok(None)
    }

    /// Run a subtable's semantics once and record its export, returning the destination if it
    /// branched
    pub fn build_table(&mut self, state: &mut State, table_id: TableId) -> Result<Option<u64>> {
        if !self.built.insert(table_id) {
            return Ok(None);
        }
        let table = self
            .table
            .tables
            .get(&table_id)
            .context("build of a table that is not an operand")?;
        let mut executor = TableExecutor::new(table);
        let branch = executor.run(state)?;
        if let Some(export) = executor.export {
            self.exports.insert(table_id, export);
        }
        Ok(branch)
    }

    pub fn execute_block(&mut self, state: &mut State, block: &Block) -> Result<ControlFlow> {
        for stmt in block.statements.iter() {
            if let Some(flow) = self.execute_statement(state, stmt)? {
                return Ok(flow);
            }
        }
        Ok(ControlFlow::Goto(block.next))
    }

    pub fn execute_statement(
        &mut self,
        state: &mut State,
        stmt: &Statement,
    ) -> Result<Option<ControlFlow>> {
        match stmt {
            Statement::Delayslot(delay_slot) => self.execute_delay_slot(*delay_slot)?,
            Statement::Export(export) => self.execute_export(state, export)?,
            Statement::CpuBranch(cpu_branch) => return self.execute_cpu_branch(state, cpu_branch),
            Statement::LocalGoto(local_goto) => return self.execute_local_goto(state, local_goto),
            Statement::UserCall(user_call) => self.execute_user_call(state, user_call)?,
            Statement::Build(build) => return self.execute_build(state, build),
            Statement::Declare(variable_id) => self.execute_declare(state, *variable_id)?,
            Statement::Assignment(assignment) => self.execute_assignment(state, assignment)?,
        };
        Ok(None)
    }

    pub fn execute_delay_slot(&self, delay_slot: u64) -> Result<()> {
        log::trace!("DELAY_SLOT {delay_slot:?}");
        todo!()
    }

    pub fn execute_export(&mut self, state: &mut State, export: &Export) -> Result<()> {
        let export_value = match export {
            Export::Value(expr) => self.evaluate_expr(state, expr)?,
            Export::Reference { addr, memory } => {
                let address_value = self.evaluate_expr(state, addr)?;
                let address = state.get_u64(address_value)?;
                Value::Ref(Ref(
                    memory.space,
                    memory.len_bytes.get() as usize / 8,
                    Address(address),
                ))
            }
            Export::AttachVarnode {
                location: _,
                attach_value,
                attach_id: _,
            } => {
                match attach_value {
                    sleigh_rs::execution::DynamicValueType::TokenField(token_field_id) => {
                        self.get_token_field_value(*token_field_id)?
                    } // TODO: should we attach a different varnode?
                    sleigh_rs::execution::DynamicValueType::Context(_context_id) => todo!(),
                }
            }
            Export::Table {
                location: _,
                table_id,
            } => self.get_table_export(*table_id)?,
        };
        self.export = Some(export_value);
        Ok(())
    }

    pub fn get_table_export(&self, table_id: TableId) -> Result<Value> {
        if let Some(table_export_value) = self.exports.get(&table_id) {
            Ok(*table_export_value)
        } else if self.built.contains(&table_id) {
            bail!("table did not export a value")
        } else {
            bail!("table used before it was built")
        }
    }

    pub fn execute_cpu_branch(
        &self,
        state: &mut State,
        cpu_branch: &CpuBranch,
    ) -> Result<Option<ControlFlow>> {
        let dst_value = self.evaluate_expr(state, &cpu_branch.dst)?;
        let dst = if cpu_branch.direct {
            dst_value.to_u64()
        } else {
            state.get_u64(dst_value)?
        };
        if let Some(cond) = &cpu_branch.cond {
            let cond_value = self.evaluate_expr(state, cond)?;
            if state.get_u64(cond_value)? == 1 {
                log::trace!("CpuBranch {:#018x} taken conditionally", dst);
                return Ok(Some(ControlFlow::Branch(dst)));
            }
        } else {
            log::trace!("CpuBranch {:#018x} taken unconditionally", dst);
            return Ok(Some(ControlFlow::Branch(dst)));
        }
        log::trace!("CpuBranch {:#018x} not taken", dst);
        Ok(None)
    }

    pub fn execute_local_goto(
        &self,
        state: &mut State,
        local_goto: &LocalGoto,
    ) -> Result<Option<ControlFlow>> {
        if let Some(cond_expr) = &local_goto.cond {
            if self.evaluate_expr(state, cond_expr)? == Value::Int(1) {
                // FIXME
                log::trace!("LocalGoto {:?} taken conditionally", local_goto.dst);
                return Ok(Some(ControlFlow::Goto(Some(local_goto.dst))));
            }
        } else {
            log::trace!("LocalGoto {:?} taken unconditionally", local_goto.dst);
            return Ok(Some(ControlFlow::Goto(Some(local_goto.dst))));
        }
        Ok(None)
    }

    pub fn execute_user_call(&self, state: &mut State, user_call: &UserCall) -> Result<()> {
        self.evaluate_user_call(state, user_call)?;
        Ok(())
    }

    pub fn execute_build(
        &mut self,
        state: &mut State,
        build: &Build,
    ) -> Result<Option<ControlFlow>> {
        log::trace!("BUILD {build:?}");
        Ok(self
            .build_table(state, build.table)?
            .map(ControlFlow::Branch))
    }

    pub fn execute_declare(&self, _state: &mut State, variable_id: VariableId) -> Result<()> {
        log::trace!("DECLARE {variable_id:?}");
        Ok(())
    }

    pub fn execute_assignment(&mut self, state: &mut State, assignment: &Assignment) -> Result<()> {
        let right_value = self.evaluate_expr(state, &assignment.right)?;
        log::trace!("Assignment {:?} = {:?}", assignment.var, right_value);
        let var = match &assignment.var {
            sleigh_rs::execution::AssignmentWrite::Variable { value, op: _ } => match value {
                sleigh_rs::execution::AssignmentWriteVariable::Varnode(varnode_id) => {
                    let varnode = self.table.disassembler.varnode(*varnode_id);
                    Var::Ref(Ref(
                        varnode.space,
                        varnode.len_bytes.get() as usize,
                        Address(varnode.address),
                    ))
                }
                sleigh_rs::execution::AssignmentWriteVariable::Bitrange(_) => todo!(),
                sleigh_rs::execution::AssignmentWriteVariable::DynVarnode {
                    value_id,
                    attach_id,
                } => match value_id {
                    DynamicValueType::TokenField(token_field_id) => self
                        .get_attach_varnode(*attach_id, *token_field_id)?
                        .to_var(),
                    DynamicValueType::Context(_context_id) => {
                        bail!("AssignmentWriteVariable {:?} not implemented", value)
                    }
                },
                sleigh_rs::execution::AssignmentWriteVariable::Variable(variable_id) => {
                    Var::Local(*variable_id)
                }
            },
            sleigh_rs::execution::AssignmentWrite::Memory { mem, addr } => {
                let addr_value = self.evaluate_expr(state, addr)?.to_u64();
                Var::Ref(Ref(
                    mem.space,
                    mem.len_bytes.get() as usize / 8,
                    Address(addr_value),
                ))
            }
            sleigh_rs::execution::AssignmentWrite::TableExport {
                table_id,
                op: _,
                size: _,
            } => self.get_table_export(*table_id)?.to_var(),
        };
        match var {
            Var::Ref(referance) => {
                let value = state.get_u64(right_value)?;
                state.write_ref(referance, &value.to_be_bytes())?;
            }
            Var::Local(variable_id) => {
                self.variables.insert(variable_id, right_value);
            }
        };
        Ok(())
    }

    pub fn evaluate_expr(&self, state: &mut State, expr: &Expr) -> Result<Value> {
        Ok(match expr {
            Expr::Value(expr_element) => {
                match expr_element {
                    ExprElement::Op(expr_unary_op) => {
                        let value = self.evaluate_expr(state, &expr_unary_op.input)?;
                        match &expr_unary_op.op {
                            sleigh_rs::execution::Unary::Dereference(memory_location) => {
                                let referance = Ref(
                                    memory_location.space,
                                    memory_location.len_bytes.get() as usize / 8,
                                    Address(state.get_u64(value)?),
                                );
                                let mut data = [0u8; 8];
                                state.read_ref(referance, &mut data)?;
                                Value::Int(u64::from_be_bytes(data))
                            }
                            sleigh_rs::execution::Unary::Zext(_) => value,
                            sleigh_rs::execution::Unary::TakeLsb(_) => value,
                            sleigh_rs::execution::Unary::TrunkLsb { .. } => value,
                            sleigh_rs::execution::Unary::Negation => {
                                Value::Int((state.get_u64(value)? == 0) as u64)
                            }
                            sleigh_rs::execution::Unary::BitRange { .. } => value,
                            op => bail!(format!("Unimplemented ExprUnaryOp {:?}", op)),
                        }
                    }
                    ExprElement::Value { value, .. } => self.evaluate_expr_value(value)?,
                    ExprElement::UserCall(user_call) => {
                        self.evaluate_user_call(state, user_call)?
                    }
                    //execution::ExprElement::Reference(reference) => todo!(),
                    //execution::ExprElement::New(expr_new) => todo!(),
                    //execution::ExprElement::CPool(expr_cpool) => todo!(),
                    expr_element => bail!(format!("Unimplemented ExprElement {:?}", expr_element)),
                }
            }
            Expr::Op(expr_binop) => {
                let left_value = self.evaluate_expr(state, &expr_binop.left)?;
                let left = state.get_u64(left_value)?;
                let right_value = self.evaluate_expr(state, &expr_binop.right)?;
                let right = state.get_u64(right_value)?;
                Value::Int(match expr_binop.op {
                    Binary::Add => left.wrapping_add(right),
                    Binary::Sub => left.wrapping_sub(right),
                    Binary::And => left & right,
                    Binary::Xor => left ^ right,
                    Binary::Or => left | right,
                    Binary::BitAnd => left & right,
                    Binary::BitOr => left | right,
                    Binary::BitXor => left ^ right,
                    Binary::Lsl => left << right,
                    Binary::Lsr => left >> right,
                    Binary::SigLess => ((left as i64) < (right as i64)) as u64,
                    Binary::Eq => (left == right) as u64,
                    Binary::Greater => (left > right) as u64,
                    Binary::Less => (left < right) as u64,
                    op => bail!("ExprBinaryOp {:?} not implemented", op),
                })
            }
        })
    }

    pub fn evaluate_expr_value(&self, expr_value: &ExprValue) -> Result<Value> {
        Ok(match expr_value {
            ExprValue::Int(expr_number) => match expr_number.number {
                sleigh_rs::Number::Positive(x) => Value::Int(x),
                sleigh_rs::Number::Negative(x) => Value::Int(-(x as i64) as u64),
            },
            ExprValue::TokenField(expr_token_field) => {
                self.get_token_field_value(expr_token_field.id)?
            }
            ExprValue::InstStart(_) => Value::Int(self.table.inst_start),
            ExprValue::InstNext(_) => Value::Int(self.table.inst_next),
            ExprValue::Varnode(varnode_id) => {
                let varnode = self.table.varnode(*varnode_id);
                let referance = Ref(
                    varnode.space,
                    varnode.len_bytes.get() as usize,
                    Address(varnode.address),
                );
                Value::Ref(referance)
            }
            //ExprValue::Context(expr_context) => todo!(),
            //ExprValue::Bitrange(expr_bitrange) => todo!(),
            ExprValue::Table(table_id) => self.get_table_export(*table_id)?,
            ExprValue::DisVar(expr_dis_var) => Value::Int(
                self.table
                    .variables
                    .get(&expr_dis_var.id)
                    .cloned()
                    .context("Disassembly var undefined")? as u64,
            ),
            ExprValue::ExeVar(variable_id) => self
                .variables
                .get(variable_id)
                .cloned()
                .context("Execution var undefined")?,
            ExprValue::VarnodeDynamic(varnode_dynamic) => match varnode_dynamic.attach_value {
                DynamicValueType::TokenField(token_field_id) => {
                    self.get_attach_varnode(varnode_dynamic.attach_id, token_field_id)?
                }
                DynamicValueType::Context(_context_id) => {
                    bail!("ExprValue {:?} not implemented", expr_value)
                }
            },
            expr_value => bail!("ExprValue {:?} not implemented", expr_value),
        })
    }

    pub fn get_token_field_value(&self, id: TokenFieldId) -> Result<Value> {
        let token_field = self.table.disassembler.token_field(id);
        let token_field_value = self
            .table
            .token_fields
            .get(&id)
            .context("Could not get token field")?;
        Ok(match token_field.attach {
            sleigh_rs::token::TokenFieldAttach::NoAttach(_value_fmt) => {
                Value::Int(*token_field_value as u64)
            }
            sleigh_rs::token::TokenFieldAttach::Varnode(attach_varnode_id) => {
                self.get_attach_varnode(attach_varnode_id, id)?
            }
            sleigh_rs::token::TokenFieldAttach::Literal(_attach_literal_id) => todo!(),
            sleigh_rs::token::TokenFieldAttach::Number(_print_base, _attach_number_id) => todo!(),
        })
    }

    pub fn get_attach_varnode(
        &self,
        attach_id: AttachVarnodeId,
        token_field_id: TokenFieldId,
    ) -> Result<Value> {
        let token_field_value = self
            .table
            .token_fields
            .get(&token_field_id)
            .context("Could not get token field")?;
        let attach_varnode = self.table.disassembler.attach_varnode(attach_id);
        let varnode_id = attach_varnode
            .find_value(*token_field_value as usize)
            .context("Could not find attach varnode value")?;
        let varnode = self.table.disassembler.varnode(varnode_id);
        Ok(Value::Ref(varnode.into()))
    }

    pub fn evaluate_user_call(&self, state: &mut State, user_call: &UserCall) -> Result<Value> {
        let user_function = self.table.user_function(user_call.function);
        let params: Vec<Value> = user_call
            .params
            .iter()
            .map(|expr| self.evaluate_expr(state, expr))
            .collect::<Result<Vec<_>>>()?;
        state.user_call(user_function, params)
    }
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
}
