#![allow(unused_variables)]
#![allow(dead_code)]

use std::{cell::RefCell, collections::HashMap, fs::File, path::PathBuf, rc::Rc, str::FromStr};

use clap::Parser;

use anyhow::{bail, Context, Result};

mod space;

use sleigh_rs::{execution::{BlockId, Expr, ExprValue, Statement, VariableId, WriteValue}, SpaceId, TableId};
use sleigher::*;
use sleigher::execution::*;
use table::TableContext;

use space::{HashSpace, TraceSpace, FileRegion};

#[derive(Debug)]
pub struct StateInner {
    pc: u64,
    spaces: HashMap<SpaceId, Box<dyn space::MemoryRegion>>,
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

    pub fn write_ref(&self, referance: Ref, data: &[u8]) -> Result<()> {
        log::trace!("Writing {} <- {:02x?}", referance, data);
        assert!(data.len() >= referance.1);
        let data = &data[data.len()-referance.1..];
        let mut inner_state = self.0.borrow_mut();
        let space = inner_state.spaces.entry(referance.0).or_insert_with(|| Box::new(HashSpace::new()));
        space.write(referance.2, data)?;
        Ok(())
    }

    pub fn read_ref(&self, referance: Ref, data: &mut [u8]) -> Result<()> {
        let len = data.len();
        assert!(len >= referance.1);
        for byte in &mut *data { *byte = 0; }
        let data = &mut data[len-referance.1..];
        let mut inner_state = self.0.borrow_mut();
        let space = inner_state.spaces.entry(referance.0).or_insert_with(|| Box::new(HashSpace::new()));
        space.read(referance.2, data)?;
        log::trace!("Reading {} -> {:02x?}", referance, data);
        Ok(())
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
            Statement::Export(export) => {
                self.execute_export(stmt.self_ctx(export))?
            },
            Statement::CpuBranch(cpu_branch) => {
                self.execute_cpubranch(stmt.self_ctx(cpu_branch))?
            },
            Statement::LocalGoto(local_goto) => {
                return self.execute_localgoto(stmt.self_ctx(local_goto));
            },
            Statement::Build(build) => {
                self.execute_build(stmt.self_ctx(build))?
            },
            Statement::Assignment(assignment) => {
                self.execute_assignment(stmt.self_ctx(assignment))?
            },
            Statement::MemWrite(mem_write) => {
                self.execute_memwrite(stmt.self_ctx(mem_write))?
            },
            Statement::Declare(variable_id) => {
                log::trace!("Declare({:?})", variable_id);
                self.write_var(Var::Local(*variable_id), Value::Int(0));
            },
            stmt => bail!("Statement {:?} not implemented", stmt),
        };
        return Ok(None);
    }

    fn execute_localgoto(&mut self, local_goto: SleighLocalGoto) -> Result<Option<BlockId>> {
        if let Some(cond_expr) = &local_goto.inner.cond {
            if self.evaluate_expr(local_goto.same_ctx(cond_expr))? == Value::Int(1) {  //FIXME 
                log::trace!("LocalGoto {:?} taken conditionally", &local_goto.inner.dst);
                return Ok(Some(local_goto.inner.dst.clone()));
            }
        } else {
            log::trace!("LocalGoto {:?} taken unconditionally", &local_goto.inner.dst);
            return Ok(Some(local_goto.inner.dst.clone()));
        }
        log::trace!("LocalGoto {:?} not taken", &local_goto.inner.dst);
        Ok(None)
    }

    fn execute_cpubranch(&mut self, cpu_branch: SleighCpuBranch) -> Result<()> {
        let dst_value = self.evaluate_expr(cpu_branch.same_ctx(&cpu_branch.inner.dst))?;
        let dst = if cpu_branch.inner.direct {
            dst_value.to_u64()
        } else {
            self.get_value(dst_value)
        };
        if let Some(cond_expr) = &cpu_branch.inner.cond {
            let cond_value = self.evaluate_expr(cpu_branch.same_ctx(cond_expr))?;
            if self.get_value(cond_value) == 1 {
                log::trace!("CpuBranch {:#018x} taken conditionally", dst);
                self.instruction.state.0.borrow_mut().pc = dst.wrapping_sub(4);
                return Ok(());
            }
        } else {
            log::trace!("CpuBranch {:#018x} taken unconditionally", dst);
            self.instruction.state.0.borrow_mut().pc = dst.wrapping_sub(4);
            return Ok(());
        };
        log::trace!("CpuBranch {:#018x} not taken", dst);
        Ok(())
    }

    fn execute_memwrite(&mut self, mem_write: SleighMemWrite) -> Result<()> {
        let addr = self.get_value(self.evaluate_expr(mem_write.same_ctx(&mem_write.inner.addr))?);
        let data = self.get_value(self.evaluate_expr(mem_write.same_ctx(&mem_write.inner.right))?);
        let referance = Ref(mem_write.inner.mem.space, mem_write.inner.mem.len_bytes.get() as usize / 8, Address(addr));
        log::trace!("MemWrite: {} <- {:02x?}", referance, data);
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
        log::trace!("Export {:?} from table {}", export_value, export.table().name());
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
        log::trace!("Build: {}", build.table().name());

        if let Some(export) = self.instruction.execute_table(build.table())? {
            self.instruction.table_exports.borrow_mut().insert(build.inner.table.id, export);
        }

        Ok(())
    }

    fn execute_assignment(&self, assignment: SleighAssignment) -> Result<()> {
        let value = self.evaluate_expr(assignment.right())?;
        let var = self.write_value(assignment.var())?;

        log::trace!("Assignment: {:?} <- {:?} {:?}", var, value, assignment.inner.op);

        match &assignment.inner.op {
            None => (),
            Some(op) => {
                //log::warn!("Special assignment op {:?} not implemented", op);
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
                let referance = Ref(memory_location.space, memory_location.len_bytes.get() as usize / 8, Address(self.get_value(value)));
                let mut data = [0u8; 8];
                self.read_space(referance, &mut data);
                Value::Int(u64::from_be_bytes(data))
            },
            sleigh_rs::execution::Unary::Zext => value,
            sleigh_rs::execution::Unary::TakeLsb(_) => value,
            sleigh_rs::execution::Unary::Negation => Value::Int((self.get_value(value) == 0) as u64),
            sleigh_rs::execution::Unary::BitRange(_) => value,
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
            ExprValue::InstNext(expr_inst_next) => {
                Value::Int(self.instruction.state.0.borrow().pc + 4)
            },
            ExprValue::Varnode(expr_varnode) => {
                let varnode = expr_value.sleigh().varnode(expr_varnode.id);
                Value::Ref(varnode.referance())
            },
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
            ExprValue::ExeVar(expr_exe_var) => {
                self.locals.0.borrow().get(&expr_exe_var.id).context(format!("Local variable {:?} not Declared", expr_exe_var.id))?.clone()
            }
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
            sleigh_rs::execution::Binary::Greater => (left > right) as u64,
            sleigh_rs::execution::Binary::Less => (left < right) as u64,
            unknown_op => bail!("ExprBinaryOp {:?} not implemented", unknown_op),
        }))
    }
}

#[derive(Debug, Clone)]
struct FileMap {
    address: u64,
    path: PathBuf,
}

fn parse_int(s: &str) -> std::result::Result<u64, std::num::ParseIntError> {
    if let Some(s) = s.strip_prefix("0x") {
        u64::from_str_radix(s, 16)
    } else {
        u64::from_str_radix(s, 10)
    }
}

impl FromStr for FileMap {
    type Err=anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let (address, path) = s.split_once(":").context("File mapping does not contain ':'")?;
        let address = parse_int(address)?;
        let path = PathBuf::from_str(path)?;
        Ok(FileMap { address, path })
    }
}

#[derive(Debug, Clone)]
struct RamMap {
    address: u64,
    length: u64,
}

impl FromStr for RamMap {
    type Err=anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let (address, length) = s.split_once(":").context("Ram mapping does not contain ':'")?;
        let address = parse_int(address)?;
        let length = parse_int(length)?;
        Ok(RamMap{ address, length })
    }
}

#[derive(Debug, Clone)]
struct RegAssignment {
    name: String,
    value: u64,
}

impl FromStr for RegAssignment {
    type Err=anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let (name, value) = s.split_once("=").context("register assignemnt must contain '='")?;
        let name = name.to_string();
        let value = parse_int(value)?;
        Ok(RegAssignment{ name, value })
    }
}

#[derive(Debug, Clone)]
struct Breakpoint {
    address: u64,
    name: String, 
    regs: Vec<String>,
}

impl FromStr for Breakpoint {
    type Err=anyhow::Error;
    
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let (address, name_and_regs) = s.split_once(':').context("Breakpoint must be on the form: address:name:reglist")?;
        let (name, regs) = name_and_regs.split_once(":").context("Breakpoint must be on the form: address:name:reglist")?;
        let address = parse_int(address)?;
        let name = name.to_string();
        let regs = regs.split(',').map(|reg| reg.to_string()).collect::<Vec<String>>();

        Ok(Self { address, name, regs })
    }

}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    slaspec: PathBuf,

    #[arg(short='f', long="map")]
    file_mappings: Vec<FileMap>,

    #[arg(short='m', long="ram")]
    ram_mappings: Vec<RamMap>,

    #[arg(short='e', long="entrypoint", value_parser=parse_int)]
    entrypoint: u64,

    #[arg(short='r', long="reg")]
    registers: Vec<RegAssignment>,

    #[arg(short='b', long="breakpoint")]
    breakpoints: Vec<Breakpoint>,

    #[arg(short='s', long="steps", value_parser=parse_int)]
    steps: Option<u64>
}

impl State {
    fn read_ref_vec(&self, referance: Ref) -> Result<Vec<u8>> {
        let mut data = vec![0u8; referance.1];
        self.read_ref(referance, &mut data)?;
        Ok(data)
    }

    fn read_ref_u32be(&self, referance: Ref) -> Result<u32> {
        let mut data = [0u8; 4];
        self.read_ref(referance, &mut data)?;
        Ok(u32::from_be_bytes(data))
    }

    fn read_ref_u32le(&self, referance: Ref) -> Result<u32> {
        let mut data = [0u8; 4];
        self.read_ref(referance, &mut data)?;
        Ok(u32::from_le_bytes(data))
    }

    fn read_ref_u64be(&self, referance: Ref) -> Result<u64> {
        let mut data = [0u8; 8];
        self.read_ref(referance, &mut data)?;
        Ok(u64::from_be_bytes(data))
    }

    fn read_ref_u64le(&self, referance: Ref) -> Result<u64> {
        let mut data = [0u8; 8];
        self.read_ref(referance, &mut data)?;
        Ok(u64::from_le_bytes(data))
    }
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();
    let sleigh = match sleigh_rs::file_to_sleigh(&args.slaspec){
        Ok(sleigh) => sleigh,
        Err(err) => {
            println!("Could not open or parse slaspec: {:#?}", err);
            bail!("Could not open or pasrse slaspec");
        },
    };

    let sleigh: SleighSleigh = (&sleigh).into();

    let mut memory = space::MappedSpace::new();
    for file_map in args.file_mappings {
        let file = File::open(&file_map.path).context(format!("Could not open file: {:?}", &file_map.path))?;
        let size = file.metadata()?.len();
        memory.add_mapping(Address(file_map.address), size, Box::new(FileRegion(file)))
    }

    for ram_map in args.ram_mappings {
        memory.add_mapping(Address(ram_map.address), ram_map.length, Box::new(HashSpace::new()))
    }

    let state = State::new();
    state.0.borrow_mut().spaces.insert(SpaceId(0), Box::new(memory));
    state.0.borrow_mut().pc = args.entrypoint;

    log::info!("=== Initializing registers ===");
    for reg in args.registers {
        let varnode = sleigh.varnode_by_name(&reg.name).unwrap();
        state.write_ref(varnode.referance(), &reg.value.to_be_bytes())?;
    }

    let read_reg_u32 = |name| {
        let varnode = sleigh.varnode_by_name(name).unwrap();
        state.read_ref_u32be(varnode.referance()).unwrap()
    };

    for step in args.steps.map(|steps| 0..steps).unwrap_or(0..u64::MAX) {
        let pc = state.0.borrow().pc;

        if pc == 0u64 {
            log::info!("PC = null! Exiting!");
            break;
        }

        for bp in &args.breakpoints {
            if bp.address == pc {
                let regs = bp.regs.iter().map(|reg|
                    format!("{}={:#010x}", reg, read_reg_u32(&reg))
                ).collect::<Vec<_>>().join(" ");
                log::info!("Breakpoint {:#010x}: {} {}", bp.address, bp.name, regs);
            }
        }

        let mut instr = [0u8; 4];
        state.read_ref(Ref(SpaceId(0), 4, Address(pc)), &mut instr)?;

        let table = sleigh.instruction_table();
        let instruction = table.disassemble(pc, &instr).unwrap();
        log::debug!("=== {:5}: PC {:#018x} INSTRUCTION {} ===", step, pc, instruction);

        let instr_exec = InstructionExecutor(Rc::new(InstructionExecutorInner {
            state: state.clone(),
            table_exports: RefCell::new(Default::default()),
            current_instruction: instr.to_vec()
        }));

        instr_exec.execute_table(sleigh.instruction_table())?;

        let pc = state.0.borrow().pc.clone();
        state.0.borrow_mut().pc = pc.wrapping_add(4);
    }

    let mut hash_bytes = [0u8; 32];
    state.read_ref(Ref(SpaceId(0), 32, Address(0xffffff80u64)), &mut hash_bytes)?;
    log::info!("{:02x?}", hash_bytes);

    Ok(())
}