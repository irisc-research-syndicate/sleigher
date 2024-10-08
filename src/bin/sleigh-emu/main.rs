#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(dead_code)]

use std::{cell::RefCell, collections::HashMap, path::PathBuf, rc::Rc};

use clap::Parser;

use anyhow::{anyhow, bail, Context, Result};

use sleigh_rs::{execution::{BlockId, Expr, ExprValue, Statement, VariableId, WriteValue}, varnode::Varnode, SpaceId, TableId};
use sleigher::*;
use sleigher::execution::*;
use table::TableContext;


#[derive(Debug)]
pub struct Space(HashMap<Address, u8>);

impl Space {
    pub fn write_bytes(&mut self, address: Address, bytes: &[u8]) {

        for (offset, byte) in bytes.iter().enumerate() {
            self.0.insert(Address(address.0 + offset as u64), *byte);
        }
    }

    pub fn read_bytes(&mut self, address: Address, bytes: &mut [u8]) {
        for (offset, byte) in bytes.iter_mut().enumerate() {
            *byte = *self.0.entry(Address(address.0 + offset as u64)).or_default();
        }

    }
}

impl Space {
    pub fn new() -> Self {
        Space(HashMap::new())
    }
}

impl Default for Space {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct StateInner {
    pc: u64,
    spaces: HashMap<SpaceId, Space>,
    context: (),
}

#[derive(Debug, Clone)]
pub struct State(Rc<RefCell<StateInner>>);

impl State {
    pub fn new() -> Self {
        State(Rc::new(RefCell::new(StateInner {
            pc: 0,
            spaces: HashMap::new(),
            context: (),
        })))
    }

    pub fn new_instruction(self, instruction_bytes: Vec<u8>) -> InstructionExecutor {
        InstructionExecutor(Rc::new(InstructionExecutorInner {
            state: self.clone(),
            table_exports: RefCell::new(HashMap::new()),
            current_instruction: instruction_bytes
        }))
    }

    pub fn write_ref(&self, referance: Ref, data: &[u8]) {
        log::debug!("Writing {} <- {:02x?}", referance, data);
        assert!(data.len() >= referance.1);
        let data = &data[data.len()-referance.1..];
        let mut inner_state = self.0.borrow_mut();
        let space = inner_state.spaces.entry(referance.0).or_insert_with(|| Space::new());
        space.write_bytes(referance.2, data);
    }

    pub fn read_ref(&self, referance: Ref, data: &mut [u8]) {
        let len = data.len();
        assert!(len >= referance.1);
        for byte in &mut *data { *byte = 0; }
        let data = &mut data[len-referance.1..];
        let mut inner_state = self.0.borrow_mut();
        let space = inner_state.spaces.entry(referance.0).or_insert_with(|| Space::new());
        space.read_bytes(referance.2, data);
        log::debug!("Reading {} -> {:02x?}", referance, data);
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct InstructionExecutorInner {
    state: State,
    table_exports: RefCell<HashMap<TableId, Value>>,
    current_instruction: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct InstructionExecutor(Rc<InstructionExecutorInner>);

impl std::ops::Deref for InstructionExecutor {
    type Target = InstructionExecutorInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Eq, PartialEq, Default)]
struct Locals(RefCell<HashMap<VariableId, Value>>);

#[derive(Debug)]
pub struct TableExecutor {
    instruction: InstructionExecutor,
    locals: Locals,
    export: Option<Value>,
}

impl InstructionExecutor {
    pub fn execute_table(&self, table: SleighTable) -> Result<Option<Value>> {
        let disasm_table = table.disassemble(self.state.0.borrow().pc, &self.current_instruction)?;

        let mut executor = TableExecutor {
            instruction: self.clone(),
            locals: Locals(RefCell::new(HashMap::new())),
            export: None,
        };

        if let Some(exec) = disasm_table.constructor.execution() {
            let mut current_block = Some(exec.entry_block());
            while let Some(block) = current_block.take() {
                let next_block_id = executor.execute_block(block)?;
                current_block = next_block_id.map(|id| exec.block(id));
            }
        }

        Ok(executor.into_export())
    }
}

impl TableExecutor {
    pub fn into_export(self) -> Option<Value> {
        self.export
    }

    pub fn execute_block(&mut self, block: SleighBlock) -> Result<Option<BlockId>> {
        for stmt in block.statements() {
            if let Some(goto_block_id) = self.execute_statement(stmt)? {
                return Ok(Some(goto_block_id));
            }
        }
        Ok(block.inner.next)
    }

    pub fn execute_statement(&mut self, stmt: SleighStatement) -> Result<Option<BlockId>> {
        match stmt.inner {
            Statement::Export(export) => self.execute_export(stmt.self_ctx(export))?,
            Statement::CpuBranch(cpu_branch) => self.execute_cpubranch(stmt.self_ctx(cpu_branch))?,
            Statement::LocalGoto(local_goto) => {
                return self.execute_localgoto(stmt.self_ctx(local_goto));
            },
            Statement::Build(build) => self.execute_build(stmt.self_ctx(build))?,
            Statement::Assignment(assignment) => self.execute_assignment(stmt.self_ctx(assignment))?,
            Statement::MemWrite(mem_write) => self.execute_memwrite(stmt.self_ctx(mem_write))?,
            stmt => bail!("Statement {:?} not implemented", stmt),
        };
        return Ok(None);
    }

    fn execute_localgoto(&mut self, local_goto: SleighLocalGoto) -> Result<Option<BlockId>> {
        if let Some(cond_expr) = &local_goto.inner.cond {
            if self.evaluate_expr(local_goto.same_ctx(cond_expr))? == Value::Int(1) {
                log::debug!("LocalGoto {:?} taken conditionally", &local_goto.inner.dst);
                return Ok(Some(local_goto.inner.dst.clone()));
            }
        } else {
            log::debug!("LocalGoto {:?} taken unconditionally", &local_goto.inner.dst);
            return Ok(Some(local_goto.inner.dst.clone()));
        }
        log::debug!("LocalGoto {:?} not taken", &local_goto.inner.dst);
        Ok(None)
    }

    fn execute_cpubranch(&mut self, cpu_branch: SleighCpuBranch) -> Result<()> {
        let dst = self.evaluate_expr(cpu_branch.same_ctx(&cpu_branch.inner.dst))?.to_u64();
        if let Some(cond_expr) = &cpu_branch.inner.cond {
            if self.evaluate_expr(cpu_branch.same_ctx(cond_expr))? == Value::Int(1) {
                log::debug!("CpuBranch {:#018x} taken conditionally", dst);
                self.instruction.state.0.borrow_mut().pc = dst;
                return Ok(());
            }
        } else {
            log::debug!("CpuBranch {:#018x} taken unconditionally", dst);
            self.instruction.state.0.borrow_mut().pc = dst;
            return Ok(());
        };
        log::debug!("CpuBranch {:#018x} not taken", dst);
        Ok(())
    }

    fn execute_memwrite(&mut self, mem_write: SleighMemWrite) -> Result<()> {
        let addr = self.get_value(self.evaluate_expr(mem_write.same_ctx(&mem_write.inner.addr))?);
        let data = self.get_value(self.evaluate_expr(mem_write.same_ctx(&mem_write.inner.right))?);
        let referance = Ref(mem_write.inner.mem.space, mem_write.inner.mem.len_bytes.get() as usize / 8, Address(addr));
        log::debug!("MemWrite: {} <- {:02x?}", referance, data);
        self.write_space(referance, &data.to_be_bytes());
        Ok(())
    }

    fn execute_export(&mut self, export: SleighExport) -> Result<()> {
        let export_value = match export.inner {
            sleigh_rs::execution::Export::Const { len_bits, location, export: export_const} => {
                self.evaluate_export_const(export.statement().self_ctx(export_const))?
            },
            sleigh_rs::execution::Export::Value(expr) => self.evaluate_expr(export.statement().self_ctx(expr))?,
            sleigh_rs::execution::Export::Reference { addr, memory } => {
                let address = self.get_value(self.evaluate_expr(export.statement().self_ctx(addr))?);
                Value::Ref(Ref(memory.space, memory.len_bytes.get() as usize / 8, Address(address)))
            },
        };
        log::debug!("Export {:?} from table {}", export_value, export.table().name());
        self.export = Some(export_value);
        Ok(())
    }

    fn evaluate_export_const(&self, export_const: SleighExportConst) -> Result<Value> {
        Ok(match export_const.inner {
            sleigh_rs::execution::ExportConst::DisVar(variable_id) => {
                let table = export_const.table();
                let disassembled_table = table.disassemble(self.instruction.state.0.borrow().pc, &self.instruction.current_instruction)?;
                let disvar = disassembled_table.variables.get(variable_id).context("Could not find disassembly var")?;
                Value::Int(*disvar as u64)
            },
            export_const => bail!("ExportConst {:?} unimplemented", export_const),
            //sleigh_rs::execution::ExportConst::TokenField(token_field_id) => todo!(),
            //sleigh_rs::execution::ExportConst::Context(context_id) => todo!(),
            //sleigh_rs::execution::ExportConst::InstructionStart => todo!(),
            //sleigh_rs::execution::ExportConst::Table(table_id) => todo!(),
            //sleigh_rs::execution::ExportConst::ExeVar(variable_id) => todo!(),
        })
    }

    fn get_value(&self, value: Value) -> u64 {
        match value {
            Value::Int(x) => x,
            Value::Ref(referance) => {
                let mut data = [0u8; 8];
                self.read_space(referance, &mut data);
                u64::from_be_bytes(data)
            },
        }
    }

    fn execute_build(&self, build: SleighBuild) -> Result<()> {
        log::debug!("Build: {}", build.table().name());

        if let Some(export) = self.instruction.execute_table(build.table())? {
            self.instruction.table_exports.borrow_mut().insert(build.inner.table.id, export);
        }

        Ok(())
    }

    fn execute_assignment(&self, assignment: SleighAssignment) -> Result<()> {
        let value = self.evaluate_expr(assignment.right())?;
        let var = self.write_value(assignment.var())?;

        log::debug!("Assignment: {:?} <- {:?} {:?}", var, value, assignment.inner.op);

        match &assignment.inner.op {
            None => (),
            Some(op) => {
                log::warn!("Special assignment op {:?} not implemented", op);
            },
        }

        self.write_var(var, value);

        Ok(())
    }

    fn write_var(&self, var: Var, value: Value) {
        match var {
            Var::Ref(referance) => {
                self.write_space(referance, &self.get_value(value).to_be_bytes());
            },
            Var::Local(variable_id) => {
                self.locals.0.borrow_mut().insert(variable_id, value);
            },
        };
    }

    fn write_space(&self, referance: Ref, data: &[u8]) {
        self.instruction.state.write_ref(referance, data);
    }

    fn read_space(&self, referance: Ref, data: &mut [u8]) {
        self.instruction.state.read_ref(referance, data);
    }

    fn write_value(&self, write_value: SleighWriteValue) -> Result<Var> {
        Ok(match write_value.inner {
            WriteValue::Varnode(write_varnode) => {
                let varnode = write_value.sleigh().varnode(write_varnode.id);
                Var::Ref(varnode.referance())
            },
            WriteValue::Bitrange(write_bitrange) => todo!(),
            WriteValue::TokenField(write_token_field) => todo!(),
            WriteValue::TableExport(write_table) => {
                if let Some(value) = self.instruction.table_exports.borrow().get(&write_table.id) {
                    value.to_var()
                } else {
                    let table = write_value.sleigh().table(write_table.id);
                    self.instruction.execute_table(table.clone())?
                        .context(format!("Table {}/{} did not export anything", table.clone().name(), write_table.id.0))?
                        .to_var()
                }
            },
            WriteValue::Local(write_exe_var) => {
                Var::Local(write_exe_var.id)
            },
        })
    }

    fn evaluate_expr(&self, expr: SleighExpr) -> Result<Value> {
        match expr.inner {
            Expr::Value(expr_elem) => self.evaluate_expr_elem(expr.self_ctx(expr_elem)),
            Expr::Op(expr_binop) => self.evaluate_expr_binop(expr.self_ctx(expr_binop)),
        }
    }

    fn evaluate_expr_elem(&self, elem: SleighExprElement) -> Result<Value> {
        match elem.inner {
            sleigh_rs::execution::ExprElement::Value(expr_value) => self.evaluate_expr_value(elem.same_ctx(expr_value)),
            sleigh_rs::execution::ExprElement::Op(expr_unary_op) => self.evaluare_expr_unary_op(elem.same_ctx(expr_unary_op)),
            expr_elem => bail!(format!("ExprElement {:?} not implemented", expr_elem)),
        }
    }

    fn evaluare_expr_unary_op(&self, expr_unary_op: SleighExprUnaryOp) -> Result<Value> {
        let value = self.evaluate_expr(expr_unary_op.statement().self_ctx(&expr_unary_op.inner.input))?;
        Ok(match &expr_unary_op.inner.op {
            sleigh_rs::execution::Unary::Dereference(memory_location) => {
                //log::debug!("Dereferance {:?} {:?}", memory_location, value);
                let referance = Ref(memory_location.space, memory_location.len_bytes.get() as usize / 8, Address(self.get_value(value)));
                let mut data = [0u8; 8];
                self.read_space(referance, &mut data);
                Value::Int(u64::from_be_bytes(data))
            },
            sleigh_rs::execution::Unary::Zext => value,
            sleigh_rs::execution::Unary::TakeLsb(non_zero) => value,
            op => bail!(format!("Unimplemented ExprUnaryOp {:?}", op)),
        })
    }

    fn evaluate_expr_value(&self, expr_value: SleighExprValue) -> Result<Value> {
        Ok(match expr_value.inner {
            ExprValue::Int(expr_number) => {
                match expr_number.number {
                    sleigh_rs::Number::Positive(x) => Value::Int(x),
                    sleigh_rs::Number::Negative(x) => Value::Int((-(x as i64)) as u64),
                }
            },
            ExprValue::TokenField(expr_token_field) => {
                let token_field = expr_value.sleigh().token_field(expr_token_field.id);
                //log::debug!("Eval TokenField Expr: {:?}", token_field);
                let decoded_token_field = token_field.decode(&self.instruction.current_instruction);
                match token_field.inner.attach {
                    sleigh_rs::token::TokenFieldAttach::NoAttach(value_fmt) => {
                        Value::Int(decoded_token_field.value as u64)
                    },
                    sleigh_rs::token::TokenFieldAttach::Varnode(attach_varnode_id) => {
                        let attach_varnode = expr_value.sleigh().inner.attach_varnode(attach_varnode_id);
                        let varnode_id = attach_varnode.find_value(decoded_token_field.value).unwrap();
                        let varnode = expr_value.sleigh().varnode(varnode_id);
                        Value::Ref(varnode.referance())
                    }
                    sleigh_rs::token::TokenFieldAttach::Literal(attach_literal_id) => todo!(),
                    sleigh_rs::token::TokenFieldAttach::Number(print_base, attach_number_id) => todo!(),
                }
            },
            //ExprValue::InstStart(expr_inst_start) => todo!(),
            //ExprValue::InstNext(expr_inst_next) => todo!(),
            //ExprValue::Varnode(expr_varnode) => todo!(),
            //ExprValue::Context(expr_context) => todo!(),
            //ExprValue::Bitrange(expr_bitrange) => todo!(),
            ExprValue::Table(expr_table) => {
                let table = expr_value.sleigh().table(expr_table.id);
                self.instruction.execute_table(table)?.context("Table as no export")?
            },
            ExprValue::DisVar(expr_dis_var) => {
                let disassembled_table = expr_value.table().disassemble(self.instruction.state.0.borrow().pc, &self.instruction.current_instruction)?;
                let value = disassembled_table.variables.get(&expr_dis_var.id).context(format!("Could not find Disassembly Variable {:?}", expr_dis_var.id))?;
                Value::Int(*value as u64)
            },
            //ExprValue::ExeVar(expr_exe_var) => todo!(),
            expr_value => bail!("ExprValue {:?} not implemented", expr_value),
        })
    }

    fn evaluate_expr_binop(&self, expr_binop: SleighExprBinaryOp) -> Result<Value> {
        let left = self.get_value(self.evaluate_expr(expr_binop.statement().self_ctx(&expr_binop.inner.left))?);
        let right = self.get_value(self.evaluate_expr(expr_binop.statement().self_ctx(&expr_binop.inner.right))?);
        Ok(Value::Int(match expr_binop.inner.op {
            sleigh_rs::execution::Binary::Add => left.wrapping_add(right),
            sleigh_rs::execution::Binary::Sub => left.wrapping_sub(right),
            sleigh_rs::execution::Binary::And => left & right,
            sleigh_rs::execution::Binary::Xor => left ^ right,
            sleigh_rs::execution::Binary::Or => left | right,
            sleigh_rs::execution::Binary::BitAnd => left & right,
            sleigh_rs::execution::Binary::BitOr => left | right,
            sleigh_rs::execution::Binary::BitXor => left ^ right,
            sleigh_rs::execution::Binary::Lsl => left << right,
            sleigh_rs::execution::Binary::Lsr => left >> right,
            sleigh_rs::execution::Binary::SigLess => ((left as i64) < (right as i64)) as u64,
            sleigh_rs::execution::Binary::Eq => (left == right) as u64,
            unknown_op => bail!("ExprBinaryOp {:?} not implemented", unknown_op),
        }))
    }
}

#[derive(Debug, Parser)]
struct Args {
    slaspec: PathBuf,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();
    let binding = sleigh_rs::file_to_sleigh(&args.slaspec).ok().ok_or(anyhow!("Could not open or parse slaspec"))?;
    let sleigh: SleighSleigh = (&binding).into();

    let instrs = vec![
    //    0x4a, 0x04, 0x08, 0x00,
        0x00, 0x21, 0x03, 0x98,
        0x00, 0x21, 0x03, 0x98,
    //    0x11, 0x22, 0x33, 0x44,
    //    0x94, 0xff, 0xf3, 0x86,
    //    0x66, 0x84, 0x00, 0x02
    ];

    let instrs = vec![
        0x6c, 0x20, 0x18, 0x06,
        0x70, 0x3f, 0x0c, 0x6a,
        0x6c, 0x20, 0x7b, 0x96,
        0x6c, 0x20, 0x83, 0x92,
        0x6c, 0x20, 0x8b, 0x8e,
        0x6c, 0x20, 0x93, 0x8a,
        0x6c, 0x20, 0x9b, 0x86,
        0x6c, 0x20, 0xa3, 0x82,
        0x6c, 0x20, 0xab, 0x7e,
        0x6c, 0x20, 0xb3, 0x7a,
        0x6c, 0x20, 0xbb, 0x76,
        0xfc, 0x97, 0x20, 0x08,
        0x02, 0xf6, 0x01, 0x00,
        0x02, 0xf4, 0x00, 0x80,
        0x66, 0xe7, 0x01, 0x52,
        0x66, 0xe4, 0x00, 0x82,
        0x24, 0x05, 0x20, 0x00,
        0x20, 0xa5, 0x00, 0x04,
        0xfc, 0x85, 0x28, 0x0b,
        0xa0, 0x00, 0x00, 0x0d,
        0x00, 0x12, 0x00, 0x00,
        0xac, 0x9d, 0x00, 0x29,
        0x38, 0x84, 0x04, 0x94,
        0x6e, 0x80, 0x20, 0x02,
        0xfc, 0xe4, 0x38, 0x08,
        0xfe, 0xc5, 0xb0, 0x08,
        0xfe, 0x86, 0xa0, 0x08,
        0x94, 0xff, 0xfd, 0xdf,
        0x66, 0x84, 0x00, 0x02,
        0x38, 0x84, 0x04, 0x94,
        0x6e, 0x80, 0x20, 0x02,
        0x95, 0x00, 0x00, 0x1f,
        0x6c, 0x20, 0x9b, 0x86,
    ];

    const START_ADDRESS: u64 = 0x008af5f8u64;

    let state = State::new();
    state.0.borrow_mut().pc = 0x008af5f8u64;

    state.write_ref(Ref(SpaceId(0), instrs.len(), Address(START_ADDRESS)), &instrs);

    let regs = vec![
        ("r1", 0xffffffffffffff00u64),
        ("r3", 0x0303030303030303u64),
        ("r4", 0x0404040404040404u64),
        ("r5", 0x0505050505050505u64),
        ("r6", 0x0606060606060606u64),
        ("r7", 0x0707070707070707u64),
        ("r15", 0x1515151515151515u64),
        ("r16", 0x1616161616161616u64),
        ("r17", 0x1717171717171717u64),
        ("r18", 0x1818181818181818u64),
        ("r19", 0x1919191919191919u64),
        ("r20", 0x2020202020202020u64),
        ("r21", 0x2121212121212121u64),
        ("r22", 0x2222222222222222u64),
        ("r23", 0x2323232323232323u64),
    ];

    log::debug!("=== Initializing registers ===");
    for (reg_name, reg_value) in regs {
        let varnode = sleigh.varnode_by_name(reg_name).unwrap();
        state.write_ref(varnode.referance(), &reg_value.to_be_bytes());
    }

    for _ in 0..24 {
        let pc = state.0.borrow().pc;
        log::debug!("========== PC {:#018x} ==========", pc);

        let mut instr = [0u8; 4];
        state.read_ref(Ref(SpaceId(0), 4, Address(pc)), &mut instr);

        let table = sleigh.instruction_table();
        let instruction = table.disassemble(pc, &instr).unwrap();
        log::debug!("=== INSTRUCTION {} ===", instruction);

        let instr_exec = InstructionExecutor(Rc::new(InstructionExecutorInner {
            state: state.clone(),
            table_exports: RefCell::new(Default::default()),
            current_instruction: instr.to_vec()
        }));

        instr_exec.execute_table(sleigh.instruction_table())?;
        state.0.borrow_mut().pc += 4;
    }

    for (space_id, space) in &state.0.borrow().spaces {
        let mut addresses = space.0.keys().collect::<Vec<_>>();
        addresses.sort();
        let mut last_address = None;
        let mut current_bytes = vec![];
        let mut start_address = None;
        for address in addresses {
            let byte = space.0.get(address).unwrap().clone();
            if last_address == Some(address.0.wrapping_sub(1)) && current_bytes.len() < 8 {
                current_bytes.push(byte);
            } else {
                if let Some(start_address) = start_address {
                    println!("{}:{}:{} -> {:02x?}", space_id.0, start_address, current_bytes.len(), current_bytes)
                }
                start_address = Some(address.clone());
                current_bytes = vec![byte];
            }
            last_address = Some(address.0)
        }
        if let Some(start_address) = start_address {
            println!("{}:{}:{} -> {:02x?}", space_id.0, start_address, current_bytes.len(), current_bytes)
        }
    }

    Ok(())
}