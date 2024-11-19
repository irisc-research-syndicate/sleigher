use std::{collections::{HashMap, HashSet}, fmt::Debug, ops::{Deref, DerefMut}, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use nom::{branch::alt, bytes::complete::tag, character::complete::one_of, combinator::{map_res, recognize}, multi::many1, sequence::{preceded}, IResult};

use sleigh_rs::{token::{self}, Number, Sleigh, TokenFieldId, TokenId, ValueFmt};
use sleigh_rs::table::{Constructor, Table};
use sleigh_rs::meaning::AttachVarnode;
use sleigh_rs::display::DisplayElement;
use sleigh_rs::pattern::{CmpOp, Verification};
use sleigh_rs::disassembly::{Assertation, Expr, ExprElement, Op, OpUnary, ReadScope, VariableId, WriteScope};
use z3::ast::{Ast, BV, Bool};


#[derive(Debug, Clone, Parser)]
struct Cli {
    slaspec: PathBuf,
    instruction: String,
}

fn parse_dec(s: &str) -> IResult<&str, usize> {
    map_res(
        recognize(many1(one_of("0123456789"))),
        |out: &str| usize::from_str_radix(out, 10)
    )(s)
}

fn parse_hex(s: &str) -> IResult<&str, usize> {
    map_res(
        preceded(
        alt((tag("0x"), tag("0X"))),
        recognize(
            many1(one_of("0123456789abcdefABCDEF")))
        ),
        |out: &str| usize::from_str_radix(out, 16)
    )(s)
}

#[derive(Debug, Clone)]
pub struct Constraints<'asm> {
    asm: &'asm Assembler,

    token_order: Vec<TokenId>,

    tokens: HashMap<TokenId, BV<'asm>>,
    fields: HashMap<TokenFieldId, BV<'asm>>,

    eqs: HashSet<Bool<'asm>>,
}

impl<'asm> Constraints<'asm> {
    pub fn new(asm: &'asm Assembler) -> Self {
        Self {
            asm,
            token_order: Vec::new(),
            tokens: HashMap::new(),
            fields: HashMap::new(),
            eqs: HashSet::new(),
        }
    }

    pub fn token(&mut self, token_id: TokenId) -> BV<'asm> {
        let token = self.asm.sleigh.token(token_id);
        let bv = self.tokens.entry(token_id).or_insert_with(|| {
            self.token_order.push(token_id);
            BV::fresh_const(&self.asm.ctx, token.name(), 8*(token.len_bytes().get() as u32))
        });
        bv.clone()
    }

    pub fn token_field(&mut self, token_field_id: TokenFieldId, sz: Option<u32>) -> BV<'asm> {
        let token_field = self.asm.sleigh.token_field(token_field_id);
        let field_bv = self.fields.entry(token_field_id).or_insert_with(|| {
            let field_bv = BV::fresh_const(&self.asm.ctx, token_field.name(), token_field.bits.len().get() as u32);
            field_bv
        }).clone();
        let token_bv = self.token(token_field.token);
        self.eq(token_bv.extract((token_field.bits.end().get()-1) as u32, token_field.bits.start() as u32)._eq(&field_bv));

        if let Some(sz) = sz {
            if field_bv.get_size() < sz {
                if token_field.raw_value_is_signed() {
                    field_bv.sign_ext(sz - field_bv.get_size())
                } else {
                    field_bv.zero_ext(sz - field_bv.get_size())
                }
            } else if field_bv.get_size() > sz {
                field_bv.extract(sz-1, 0)
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
        BV::from_u64(&self.asm.ctx, u, sz as u32)
    }

    pub fn build_i64_const(&self, i: i64, sz: u32) -> BV<'asm> {
        BV::from_i64(&self.asm.ctx, i, sz as u32)
    }

}

#[derive(Debug)]
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
    pub fn new(constraints: Constraints<'asm>, constructor: &'asm Constructor) -> Self{
        Self {
            constraints,
            constructor,
            variables: HashMap::new(),
        }
    }

    pub fn variable(&mut self, variable_id: VariableId) -> BV<'asm> {
        self.variables.entry(variable_id).or_insert_with(|| {
            let variable = self.constructor.pattern.disassembly_var(variable_id);
            BV::fresh_const(&self.constraints.asm.ctx, variable.name(), 64)
        }).clone()
    }

    pub fn build_expr_bv(&mut self, expr: &Expr, sz: u32) -> BV<'asm> {
        let expr_bv = match expr {
            Expr::Value(expr_element) => {
                match expr_element {
                    ExprElement::Value { value, location: _ } => {
                        match value {
                            ReadScope::Integer(number) => match number {
                                Number::Positive(x) => self.build_u64_const(*x, sz),
                                Number::Negative(x) => self.build_i64_const(-(*x as i64), sz),
                            },
                            ReadScope::Context(_context_id) => todo!(),
                            ReadScope::TokenField(token_field_id) => self.token_field(*token_field_id, Some(sz)),
                            ReadScope::InstStart(_inst_start) => self.build_u64_const( 0x1234567890abcdef, sz),
                            ReadScope::InstNext(_inst_next) => todo!(),
                            ReadScope::Local(variable_id) => self.variable(*variable_id),
                        }
                    },
                    ExprElement::Op(_span, op_unary, expr) => {
                        let expr_bv = self.build_expr_bv(expr, 64);
                        match op_unary {
                            OpUnary::Negation => expr_bv.bvnot(),
                            OpUnary::Negative => expr_bv.bvneg(),
                        }
                    },
                }
            },
            Expr::Op(_span, op, expr, expr1) => {
                let expr_r = self.build_expr_bv(expr, sz);
                let expr_l = self.build_expr_bv(expr1, sz);
                //dbg!(&expr_r, expr_r.get_size(), &expr_l, expr_l.get_size(), op);
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
            },
        };
        if expr_bv.get_size() < sz {
            expr_bv.sign_ext(sz - expr_bv.get_size())
        } else if expr_bv.get_size() > sz {
            expr_bv.extract(sz-1, 0)
        } else {
            expr_bv
        }
    }
}


#[derive(Debug)]
pub struct Assembler {
    sleigh: Sleigh,
    ctx: z3::Context,
}

impl Deref for Assembler {
    type Target = Sleigh;

    fn deref(&self) -> &Self::Target {
        &self.sleigh
    }
}


impl Assembler {
    pub fn new(sleigh: Sleigh) -> Self {
        Self {
            sleigh,
            ctx: z3::Context::new(&z3::Config::new()),
        }
    }

    pub fn assemble_instruction<'a, 'asm>(&'asm self, s: &'a str) -> IResult<&'a str, Constraints<'asm>> {
        let constraints = Constraints::new(self);
        self.assemble_table(self.table(self.instruction_table()), constraints, s)
    }

    pub fn assemble_table<'a, 'asm>(&'asm self, table: &'asm Table, constraints: Constraints<'asm>, s: &'a str) -> IResult<&'a str, Constraints<'asm>> {
        for constructor in table.constructors() {
            match self.assemble_constructor(constructor, constraints.clone(), s) {
                Ok(x) => return Ok(x),
                Err(_) => continue
            }
        }
        nom::combinator::fail(s)
    }

    pub fn assemble_constructor<'a, 'asm>(&'asm self, constructor: &'asm Constructor, variables: Constraints<'asm>, mut s: &'a str) -> IResult<&'a str, Constraints<'asm>> {
//        dbg!(constructor);
        if let Some(mneumonic) = constructor.display.mneumonic.as_ref() {
            s = nom::bytes::complete::tag(mneumonic.as_str())(s)?.0;
            dbg!(mneumonic);
        }

        let mut variables = Variables::new(variables, constructor);

        for block in constructor.pattern.blocks() {
            for verification in block.verifications() {
                match verification {
                    Verification::ContextCheck { context: _, op: _, value: _ } => todo!(),
                    Verification::TableBuild { produced_table: _, verification: _ } => {
                        continue;
                    },
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
                    },
                    Verification::SubPattern { location: _, pattern: _ } => todo!(),
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
                            },
                        }
                    },
                }
            }
        }

        for elem in constructor.display.elements() {
            s = match elem {
                DisplayElement::Varnode(_varnode_id) => todo!(),
                DisplayElement::Context(_context_id) => todo!(),
                DisplayElement::TokenField(token_field_id) => {
                    let token_field = self.token_field(*token_field_id);
                    println!("TOKEN_FIELD: {:?} {:?}", token_field.name(), s);
                    let token_field_bv = variables.token_field(*token_field_id, None);
                    let (s, value) = match token_field.attach {
                        token::TokenFieldAttach::NoAttach(value_fmt) => {
                            self.parse_value_fmt(&value_fmt, s)?
                        },
                        token::TokenFieldAttach::Varnode(attach_varnode_id) => {
                            let attach_varnode = self.attach_varnode(attach_varnode_id);
                            self.parse_attach_varnode(attach_varnode, s)?
                        },
                        token::TokenFieldAttach::Literal(_attach_literal_id) => todo!(),
                        token::TokenFieldAttach::Number(_print_base, _attach_number_id) => todo!(),
                    };
                    // TODO: check value bits
                    let const_bv = variables.build_u64_const(value as u64, token_field_bv.get_size());
                    variables.eq(token_field_bv._eq(&const_bv));
                    s
                },
                DisplayElement::InstStart(_inst_start) => todo!(),
                DisplayElement::InstNext(_inst_next) => todo!(),
                DisplayElement::Table(table_id) => {
                    let table = self.table(*table_id);
                    println!("TABLE: {:?}/{:?} {:?}", table.name(), table_id, s);
                    let (s, table_constraints) = self.assemble_table(table, variables.constraints.clone(), s)?;
                    variables.merge(table_constraints);
                    s
                },
                DisplayElement::Disassembly(variable_id) => {
                    let variable = constructor.pattern.disassembly_var(*variable_id);
                    println!("DISASSEMBLY: {:?} {:?}", variable.name(), s);
                    let (s, negate) = if let Some(s) = s.strip_prefix('-') {
                        (s, true)
                    } else {
                        (s, false)
                    };
                    let (s, value) = nom::combinator::map(
                        alt((parse_hex, parse_dec)),
                         |value| if negate { -(value as isize) as usize } else { value }
                    )(s)?;
                    let var = variables.variable(*variable_id);
                    let const_bv = variables.build_u64_const(value as u64, var.get_size());
                    variables.eq(var._eq(&const_bv));
                    s

                },
                DisplayElement::Literal(lit) => {
                    println!("LITERAL: {:?} {:?}", lit, s);
                    nom::bytes::complete::tag(lit.as_str())(s)?.0
                },
                DisplayElement::Space => {
                    println!("SPACE: {:?}", s);
                    nom::character::complete::space1(s)?.0
                },
            };
        }


        if variables.check() {
            Ok((s, variables.constraints))
        } else {
            println!("constraint check failed");
            println!("fields: {:#?}", variables.fields.values());
            println!("variables: {:#?}", variables.variables.values());
            println!("eqs: {:#?}", variables.eqs);
            nom::combinator::fail(s)
        }
    }

    pub fn parse_value_fmt<'a>(&self, value_fmt: &ValueFmt, s: &'a str) -> IResult<&'a str, usize> {
        let (s, negate) = if value_fmt.signed {
            if let Some(s) = s.strip_prefix('-') {
                (s, true)
            } else {
                (s, false)
            }
        } else {
            (s, false)
        };
        let (s, value) = match value_fmt.base {
            sleigh_rs::PrintBase::Dec => parse_dec(s),
            sleigh_rs::PrintBase::Hex => parse_hex(s)
        }?;
        let value = if negate {
            -(value as isize) as usize
        } else {
            value
        };
        Ok((s, value))
    }

    pub fn parse_attach_varnode<'a>(&self, attach_varnode: &AttachVarnode, s: &'a str) -> IResult<&'a str, usize> {
        let mut attach_varnodes = attach_varnode.0.iter().map(|(value, id)|
            (*value, self.varnode(*id))
        ).collect::<Vec<_>>();
        attach_varnodes.sort_by(|(_, v1), (_, v2)| v2.name().cmp(v1.name()));
        for (value, varnode) in attach_varnodes.into_iter() {
            if let Some(s) = s.strip_prefix(varnode.name()) {
                return Ok((s, value))
            }
        }
        nom::combinator::fail(s)
    }
}

pub fn main() -> Result<()> {
    let args = Cli::parse();
    let sleigh = sleigh_rs::file_to_sleigh(&args.slaspec.clone()).ok().context("Could not open or parse slaspec")?;
    let assembler = Assembler::new(sleigh);
    let (rest, constraints) = assembler.assemble_instruction(&args.instruction).ok().context("Failed to parse instruction")?;

    println!("rest: {:?}", rest);
    println!("token_order: {:?}", constraints.token_order);
    println!("tokens: {:?}", constraints.tokens);
    println!("fields: {:#?}", constraints.fields.values());
    println!("eqs: {:#?}", constraints.eqs);
    println!("model: {:#?}", constraints.model());

    Ok(())
}