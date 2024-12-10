#![allow(unused_variables)]
//#![allow(dead_code)]

use std::{cell::RefCell, collections::HashMap, fs::File, path::PathBuf, rc::Rc};

use clap::Parser;

use anyhow::{bail, Context as AnyhowContext, Result};

use sleigh_rs::{user_function::UserFunction, Sleigh, SpaceId, TableId, TokenFieldId};
use sleigh_rs::execution::{self, Assignment, Block, BlockId, Build, CpuBranch, Export, Expr, ExprValue, LocalGoto, MemWrite, Statement, UserCall, VariableId, WriteValue};

use sleigher::disassembler::{Context, DisassembledTable, Disassembler};
use sleigher::value::{Address, Ref, Value, Var};
use sleigher::space::{FileRegion, HashSpace, MappedSpace, MemoryRegion};

pub struct Cpu<'sleigh> {
    sleigh: &'sleigh Sleigh,
    disassembler: Disassembler<'sleigh>,
    state: State,
}

impl<'sleigh> std::ops::Deref for Cpu<'sleigh> {
    type Target = Sleigh;

    fn deref(&self) -> &Self::Target {
        self.sleigh
    }
}

#[derive(Debug)]
pub struct StateInner {
    pc: u64,
    spaces: HashMap<SpaceId, Box<dyn MemoryRegion>>,
}

#[derive(Debug, Clone)]
pub struct State(Rc<RefCell<StateInner>>);

impl<'sleigh> std::ops::Deref for State {
    type Target = Rc<RefCell<StateInner>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl State {
    pub fn new() -> Self {
        State(Rc::new(RefCell::new(StateInner {
            pc: 0,
            spaces: HashMap::new(),
        })))
    }

    pub fn write_ref(&mut self, referance: Ref, data: &[u8]) -> Result<()> {
        log::trace!("Writing {} <- {:02x?}", referance, data);
        assert!(data.len() >= referance.1);
        let data = &data[data.len()-referance.1..];
        self.borrow_mut().spaces
            .entry(referance.0)
            .or_insert_with(|| Box::new(HashSpace::new()))
            .write(referance.2, data)
    }

    pub fn read_ref(&mut self, referance: Ref, data: &mut [u8]) -> Result<()> {
        let len = data.len();
        assert!(len >= referance.1);
        for byte in &mut *data { *byte = 0; }
        let data = &mut data[len-referance.1..];
        self.borrow_mut().spaces
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
            },
        })
    }

    fn user_call(&mut self, function: &UserFunction, params: Vec<Value>) -> Result<Value> {
        todo!();
    }
}

impl<'sleigh> Cpu<'sleigh> {
    pub fn step(&mut self) -> Result<()> {
        let mut instruction_bytes= [0u8; 4];
        self.fetch_instruction(&mut instruction_bytes)?;

        let instruction = self.disassembler.disassemble(self.state.borrow().pc, Context, &instruction_bytes)?;
        log::debug!("Executing {:#010x}: {}", instruction.inst_start, instruction);

        let mut table_executor = TableExecutor::new(&instruction.table);
        let (export, pc) = table_executor.execute(&mut self.state)?;
        self.state.borrow_mut().pc = pc;
        Ok(())
    }

    pub fn fetch_instruction(&mut self, instruction: &mut [u8]) -> Result<()> {
        let pc = self.state.borrow().pc.clone();
        self.state.read_ref(Ref(self.sleigh.default_space(), instruction.len(), Address(pc)), instruction)
    }
}

pub struct TableExecutor<'st> {
    table: &'st DisassembledTable<'st>,
    variables: HashMap<VariableId, Value>,
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
            exports: HashMap::new(),
            export: None
        }
    }

    pub fn execute(&mut self, state: &mut State) -> Result<(Option<Value>, u64)> {
        log::trace!("Executing table {}", self.table.table.name());

        if let Some(execution) = &self.table.constructor.execution {
            let mut next_block = Some(execution.entry_block);
            while let Some(block_id) = next_block.take() {
                match self.execute_block(state, execution.block(block_id))? {
                    ControlFlow::Goto(block_id) => next_block = block_id,
                    ControlFlow::Branch(pc) => return Ok((self.export, pc)),
                };
            }
        } else {
            log::warn!("Constructor has no execution(check sleigh-rs!?)");
        }
        Ok((self.export, self.table.inst_next))
    }

    pub fn execute_block(&mut self, state: &mut State, block: &Block) -> Result<ControlFlow> {
        for stmt in block.statements.iter() {
            if let Some(flow) = self.execute_statement(state, stmt)? {
                return Ok(flow);
            }
        }
        Ok(ControlFlow::Goto(block.next))
    }

    pub fn execute_statement(&mut self, state: &mut State, stmt: &Statement) -> Result<Option<ControlFlow>> {
        match stmt {
            Statement::Delayslot(delay_slot) => self.execute_delay_slot(*delay_slot)?,
            Statement::Export(export) => self.execute_export(state, export)?,
            Statement::CpuBranch(cpu_branch) => return self.execute_cpu_branch(state, cpu_branch),
            Statement::LocalGoto(local_goto) => return self.execute_local_goto(state, local_goto),
            Statement::UserCall(user_call) => self.execute_user_call(state, user_call)?,
            Statement::Build(build) => self.execute_build(state, build)?,
            Statement::Declare(variable_id) => self.execute_declare(state, *variable_id)?,
            Statement::Assignment(assignment) => self.execute_assignment(state, assignment)?,
            Statement::MemWrite(mem_write) => self.execute_mem_write(state, mem_write)?,
        };
        Ok(None)
    }

    pub fn execute_delay_slot(&self, delay_slot: u64) -> Result<()> {
        todo!()
    }

    pub fn execute_export(&mut self, state: &mut State, export: &Export) -> Result<()> {
        let export_value = match export {
            Export::Const { len_bits, location, export } => match export {
                execution::ExportConst::DisVar(variable_id) => Value::Int(*self.table.variables.get(variable_id).unwrap() as u64),
                execution::ExportConst::TokenField(token_field_id) => Value::Int(*self.table.token_fields.get(token_field_id).unwrap() as u64),
                execution::ExportConst::Context(context_id) => todo!(),
                execution::ExportConst::InstructionStart => Value::Int(self.table.inst_start),
                execution::ExportConst::Table(table_id) => self.get_table_export(state, *table_id)?,
                execution::ExportConst::ExeVar(variable_id) => self.variables.get(variable_id).unwrap().clone(),
            },
            Export::Value(expr) => self.evaluate_expr(state, expr)?,
            Export::Reference { addr, memory } => {
                let address_value = self.evaluate_expr(state, addr)?;
                let address = state.get_u64(address_value)?;
                Value::Ref(Ref(memory.space, memory.len_bytes.get() as usize / 8, Address(address)))
            },
        };
        self.export = Some(export_value);
        Ok(())
    }

    pub fn get_table_export(&self, state: &mut State, table_id: TableId) -> Result<Value> {
        if let Some(table_export_value) = self.exports.get(&table_id) {
            Ok(table_export_value.clone())
        } else {
            let table = self.table.tables.get(&table_id).unwrap();
            if let (Some(export_value), _) = TableExecutor::new(table).execute(state)? {
                Ok(export_value)
            } else {
                bail!("table did not export a value")
            }
        }
    }

    pub fn execute_cpu_branch(&self, state: &mut State, cpu_branch: &CpuBranch) -> Result<Option<ControlFlow>> {
        let dst_value = self.evaluate_expr(state, &cpu_branch.dst)?;
        let dst = if cpu_branch.direct {
            dst_value.to_u64()
        } else {
            state.get_u64(dst_value)?
        };
        let taken = if let Some(cond) = &cpu_branch.cond {
            let cond_value = self.evaluate_expr(state, cond)?;
            if state.get_u64(cond_value)? == 1 {
                log::trace!("CpuBranch {:#018x} taken conditionally", dst);
                return Ok(Some(ControlFlow::Branch(dst)));
            }
        } else {
            log::trace!("CpuBranch {:#018x} taken unconditionally", dst);
            return Ok(Some(ControlFlow::Branch(dst)));
        };
        log::trace!("CpuBranch {:#018x} not taken", dst);
        Ok(None)
    }

    pub fn execute_local_goto(&self, state: &mut State, local_goto: &LocalGoto) -> Result<Option<ControlFlow>> {
        if let Some(cond_expr) = &local_goto.cond {
            if self.evaluate_expr(state, cond_expr)? == Value::Int(1) { // FIXME
                log::trace!("LocalGoto {:?} taken conditionally", &local_goto.dst);
                return Ok(Some(ControlFlow::Goto(Some(local_goto.dst.clone()))));
            }
        } else {
            log::trace!("LocalGoto {:?} taken unconditionally", &local_goto.dst);
            return Ok(Some(ControlFlow::Goto(Some(local_goto.dst.clone()))));

        }
        Ok(None)
    }

    pub fn execute_user_call(&self, state: &mut State, user_call: &UserCall) -> Result<()> {
        self.evaluate_user_call(state, user_call)?;
        Ok(())
    }

    pub fn execute_build(&self, state: &mut State, build: &Build) -> Result<()> {
        todo!()
    }

    pub fn execute_declare(&self, state: &mut State, variable_id: VariableId) -> Result<()> {
        log::trace!("DECLARE {variable_id:?}");
        Ok(())
    }

    pub fn execute_assignment(&mut self, state: &mut State, assignment: &Assignment) -> Result<()> {
        let right_value = self.evaluate_expr(state, &assignment.right)?;
        log::trace!("Assignment {:?} = {:?}", assignment.var, right_value);
        let var = match &assignment.var {
            WriteValue::Varnode(write_varnode) => {
                let varnode = self.table.disassembler.varnode(write_varnode.id);
                Var::Ref(Ref(varnode.space, varnode.len_bytes.get() as usize, Address(varnode.address)))
            },
            WriteValue::Bitrange(write_bitrange) => todo!(),
            WriteValue::TokenField(write_token_field) => todo!(),
            WriteValue::TableExport(write_table) => self.get_table_export(state, write_table.id)?.to_var(),
            WriteValue::Local(write_exe_var) => Var::Local(write_exe_var.id),
        };
        match var {
            Var::Ref(referance) => {
                let value = state.get_u64(right_value)?;
                state.write_ref(referance, &value.to_be_bytes())?;
            },
            Var::Local(variable_id) => {
                self.variables.insert(variable_id, right_value);
            },
        };
        Ok(())
    }

    pub fn execute_mem_write(&self, state: &mut State, mem_write: &MemWrite) -> Result<()> {
        let right_value = self.evaluate_expr(state, &mem_write.right)?;
        let addr_value = self.evaluate_expr(state, &mem_write.addr)?.to_u64();
        let right_bytes = state.get_u64(right_value)?.to_be_bytes();
        let referance = Ref(mem_write.mem.space, mem_write.mem.len_bytes.get() as usize / 8, sleigher::value::Address(addr_value));
        state.write_ref(referance, &right_bytes)?;
        Ok(())
    }

    pub fn evaluate_expr(&self, state: &mut State, expr: &Expr) -> Result<Value> {
        Ok(match expr {
            Expr::Value(expr_element) => {
                match expr_element {
                    execution::ExprElement::Op(expr_unary_op) => {
                        let value = self.evaluate_expr(state, &expr_unary_op.input)?;
                        match &expr_unary_op.op {
                            sleigh_rs::execution::Unary::Dereference(memory_location) => {
                                let referance = Ref(memory_location.space, memory_location.len_bytes.get() as usize / 8, Address(state.get_u64(value)?));
                                let mut data = [0u8; 8];
                                state.read_ref(referance, &mut data)?;
                                Value::Int(u64::from_be_bytes(data))
                            },
                            sleigh_rs::execution::Unary::Zext => value,
                            sleigh_rs::execution::Unary::TakeLsb(_) => value,
                            sleigh_rs::execution::Unary::Negation => Value::Int((state.get_u64(value)? == 0) as u64),
                            sleigh_rs::execution::Unary::BitRange(_) => value,
                            op => bail!(format!("Unimplemented ExprUnaryOp {:?}", op)),
                        }
                    },
                    execution::ExprElement::Value(expr_value) => self.evaluate_expr_value(state, expr_value)?,
                    execution::ExprElement::UserCall(user_call) => {
                        self.evaluate_user_call(state, user_call)?
                    }
                    //execution::ExprElement::Reference(reference) => todo!(),
                    //execution::ExprElement::New(expr_new) => todo!(),
                    //execution::ExprElement::CPool(expr_cpool) => todo!(),
                    
                    expr_element => bail!(format!("Unimplemented ExprElement {:?}", expr_element)),
                }
            },
            Expr::Op(expr_binop) => {
                let left_value = self.evaluate_expr(state, &expr_binop.left)?;
                let left = state.get_u64(left_value)?;
                let right_value = self.evaluate_expr(state, &expr_binop.right)?;
                let right = state.get_u64(right_value)?;
                Value::Int(match expr_binop.op {
                    execution::Binary::Add => left.wrapping_add(right),
                    execution::Binary::Sub => left.wrapping_sub(right),
                    execution::Binary::And => left & right,
                    execution::Binary::Xor => left ^ right,
                    execution::Binary::Or => left | right,
                    execution::Binary::BitAnd => left & right,
                    execution::Binary::BitOr => left | right,
                    execution::Binary::BitXor => left ^ right,
                    execution::Binary::Lsl => left << right,
                    execution::Binary::Lsr => left >> right,
                    execution::Binary::SigLess => ((left as i64) < (right as i64)) as u64,
                    execution::Binary::Eq => (left == right) as u64,
                    execution::Binary::Greater => (left > right) as u64,
                    execution::Binary::Less => (left < right) as u64,
                    op => bail!("ExprBinaryOp {:?} not implemented", op),
                })
            }
        })
    }

    pub fn evaluate_expr_value(&self, state: &mut State, expr_value: &ExprValue) -> Result<Value> {
        Ok(match expr_value {
            ExprValue::Int(expr_number) => {
                match expr_number.number {
                    sleigh_rs::Number::Positive(x) => Value::Int(x),
                    sleigh_rs::Number::Negative(x) => Value::Int(-(x as i64) as u64),
                }
            },
            ExprValue::TokenField(expr_token_field) => {
                self.get_token_field_value(expr_token_field.id)?
            },
            ExprValue::InstStart(expr_inst_start) => Value::Int(self.table.inst_start),
            ExprValue::InstNext(expr_inst_next) => Value::Int(self.table.inst_next),
            ExprValue::Varnode(expr_varnode) => {
                let varnode = self.table.varnode(expr_varnode.id);
                let referance = Ref(varnode.space, varnode.len_bytes.get() as usize, Address(varnode.address));
                Value::Ref(referance)
            },
            //ExprValue::Context(expr_context) => todo!(),
            //ExprValue::Bitrange(expr_bitrange) => todo!(),
            ExprValue::Table(expr_table) => self.get_table_export(state, expr_table.id)?,
            ExprValue::DisVar(expr_dis_var) => {
                Value::Int(self.table.variables.get(&expr_dis_var.id).cloned().context("Disassembly var undefined")? as u64)
            },
            ExprValue::ExeVar(expr_exe_var) => self.variables.get(&expr_exe_var.id).cloned().context("Execution var undefined")?,
            expr_value => bail!("ExprValue {:?} not implemented", expr_value),
        })
    }

    pub fn get_token_field_value(&self, id: TokenFieldId) -> Result<Value> {
        let token_field = self.table.disassembler.token_field(id);
        let token_field_value = self.table.token_fields.get(&id).context("Could not get token field")?;
        Ok(match token_field.attach {
            sleigh_rs::token::TokenFieldAttach::NoAttach(value_fmt) => Value::Int(*token_field_value as u64),
            sleigh_rs::token::TokenFieldAttach::Varnode(attach_varnode_id) => {
                let attach_varnode = self.table.disassembler.attach_varnode(attach_varnode_id);
                let varnode_id = attach_varnode.find_value(*token_field_value as usize).context("Could not find attach varnode value")?;
                let varnode = self.table.disassembler.varnode(varnode_id);
                Value::Ref(Ref(varnode.space, varnode.len_bytes.get() as usize, Address(varnode.address)))
            },
            sleigh_rs::token::TokenFieldAttach::Literal(attach_literal_id) => todo!(),
            sleigh_rs::token::TokenFieldAttach::Number(print_base, attach_number_id) => todo!(),
        })
    }

    pub fn evaluate_user_call(&self, state: &mut State, user_call: &UserCall) -> Result<Value> {
        let user_function = self.table.user_function(user_call.function);
        let params: Vec<Value> = user_call.params.iter().map(|expr| self.evaluate_expr(state, expr)).collect::<Result<Vec<_>>>()?;
        Ok(state.user_call(user_function, params)?)
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

impl std::str::FromStr for FileMap {
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

impl std::str::FromStr for RamMap {
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

impl std::str::FromStr for RegAssignment {
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

impl std::str::FromStr for Breakpoint {
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



    let mut memory = MappedSpace::new();
    for file_map in args.file_mappings {
        let file = File::open(&file_map.path).context(format!("Could not open file: {:?}", &file_map.path))?;
        let size = file.metadata()?.len();
        memory.add_mapping(Address(file_map.address), size, Box::new(FileRegion(file)))
    }

    for ram_map in args.ram_mappings {
        memory.add_mapping(Address(ram_map.address), ram_map.length, Box::new(HashSpace::new()))
    }

    let mut state = State::new();
    state.0.borrow_mut().spaces.insert(SpaceId(0), Box::new(memory));
    state.0.borrow_mut().pc = args.entrypoint;

    log::info!("=== Initializing registers ===");
    for reg in args.registers {
        let varnode = sleigh.varnodes().iter().find(|varnode| varnode.name() == reg.name).unwrap();
        let referance = Ref(varnode.space, varnode.len_bytes.get() as usize, Address(varnode.address));
        state.write_ref(referance, &reg.value.to_be_bytes())?;
    }

    let mut cpu = Cpu {
        sleigh: &sleigh,
        disassembler: Disassembler::new(&sleigh),
        state: state,
    };

    fn read_reg_u32(cpu: &mut Cpu, name: &str) -> u32 {
        let varnode = cpu.sleigh.varnodes().iter().find(|varnode| varnode.name() == name).unwrap();
        let referance = Ref(varnode.space, varnode.len_bytes.get() as usize, Address(varnode.address));
        cpu.state.read_ref_u32be(referance).unwrap()
    }

    for step in args.steps.map(|steps| 0..steps).unwrap_or(0..u64::MAX) {
        let pc = cpu.state.borrow().pc;

        if pc == 0u64 {
            log::info!("PC = null! Exiting!");
            break;
        }

        for bp in &args.breakpoints {
            if bp.address == pc {
                let regs = bp.regs.iter().map(|reg|
                    format!("{}={:#010x}", reg, read_reg_u32(&mut cpu, reg.as_str()))
                ).collect::<Vec<_>>().join(" ");
                log::info!("Breakpoint {:#010x}: {} {}", bp.address, bp.name, regs);
            }
        }

        cpu.step()?;
    }

    let mut hash_bytes = [0u8; 32];
    cpu.state.read_ref(Ref(SpaceId(0), 32, Address(0xffffff80u64)), &mut hash_bytes)?;
    log::info!("{:02x?}", hash_bytes);

    Ok(())
}