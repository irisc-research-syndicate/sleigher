use std::collections::HashMap;

use anyhow::{bail, Context as _, Result};

use sleigh_rs::execution::{
    Assignment, Binary, Block, BlockId, Build, CpuBranch, Export, Expr, ExprElement, ExprValue,
    LocalGoto, Statement, UserCall, VariableId,
};
use sleigh_rs::{user_function::UserFunction, Sleigh, SpaceId, TableId, TokenFieldId};

use crate::disassembler::{Context, DisassembledTable, Disassembler};
use crate::space::{HashSpace, MemoryRegion};
use crate::value::{Address, Ref, Value, Var};

pub struct Cpu<'sleigh> {
    pub sleigh: &'sleigh Sleigh,
    pub disassembler: Disassembler<'sleigh>,
    pub state: State,
}

impl<'sleigh> std::ops::Deref for Cpu<'sleigh> {
    type Target = Sleigh;

    fn deref(&self) -> &Self::Target {
        self.sleigh
    }
}

impl<'sleigh> Cpu<'sleigh> {
    pub fn new(sleigh: &'sleigh Sleigh, state: State) -> Self {
        Self {
            sleigh,
            disassembler: Disassembler::new(sleigh),
            state,
        }
    }

    pub fn step(&mut self) -> Result<()> {
        let mut instruction_bytes = [0u8; 4];
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
            export: None,
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
            Statement::Build(build) => self.execute_build(state, build)?,
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
            } => self.get_table_export(state, *table_id)?,
        };
        self.export = Some(export_value);
        Ok(())
    }

    pub fn get_table_export(&self, state: &mut State, table_id: TableId) -> Result<Value> {
        if let Some(table_export_value) = self.exports.get(&table_id) {
            Ok(*table_export_value)
        } else {
            let table = self.table.tables.get(&table_id).unwrap();
            if let (Some(export_value), _) = TableExecutor::new(table).execute(state)? {
                Ok(export_value)
            } else {
                bail!("table did not export a value")
            }
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

    pub fn execute_build(&self, _state: &mut State, build: &Build) -> Result<()> {
        log::trace!("BUILD {build:?}");
        todo!()
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
                sleigh_rs::execution::AssignmentWriteVariable::DynVarnode { .. } => todo!(),
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
            } => self.get_table_export(state, *table_id)?.to_var(),
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
                    ExprElement::Value { value, .. } => self.evaluate_expr_value(state, value)?,
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

    pub fn evaluate_expr_value(&self, state: &mut State, expr_value: &ExprValue) -> Result<Value> {
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
            ExprValue::Table(table_id) => self.get_table_export(state, *table_id)?,
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
                let attach_varnode = self.table.disassembler.attach_varnode(attach_varnode_id);
                let varnode_id = attach_varnode
                    .find_value(*token_field_value as usize)
                    .context("Could not find attach varnode value")?;
                let varnode = self.table.disassembler.varnode(varnode_id);
                Value::Ref(Ref(
                    varnode.space,
                    varnode.len_bytes.get() as usize,
                    Address(varnode.address),
                ))
            }
            sleigh_rs::token::TokenFieldAttach::Literal(_attach_literal_id) => todo!(),
            sleigh_rs::token::TokenFieldAttach::Number(_print_base, _attach_number_id) => todo!(),
        })
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
