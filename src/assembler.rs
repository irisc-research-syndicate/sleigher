use std::{collections::{HashMap, HashSet}, fmt::Debug, ops::{Deref, DerefMut}};

use nom::{branch::alt, bytes::complete::tag, character::complete::one_of, combinator::{map_res, recognize}, multi::many1, sequence::{preceded}, IResult};

use sleigh_rs::{token::TokenFieldAttach, Endian, Number, Sleigh, TokenFieldId, TokenId};
use sleigh_rs::table::{Constructor, Table};
use sleigh_rs::meaning::AttachVarnode;
use sleigh_rs::display::DisplayElement;
use sleigh_rs::pattern::{CmpOp, Verification};
use sleigh_rs::disassembly::{Assertation, Expr, ExprElement, Op, OpUnary, ReadScope, VariableId, WriteScope};
use z3::ast::{Ast, BV, Bool};

fn parse_dec(s: &str) -> IResult<&str, u64> {
    map_res(
        recognize(many1(one_of("0123456789"))),
        |out: &str| u64::from_str_radix(out, 10)
    )(s)
}

fn parse_hex(s: &str) -> IResult<&str, u64> {
    map_res(
        preceded(
        alt((tag("0x"), tag("0X"))),
        recognize(
            many1(one_of("0123456789abcdefABCDEF")))
        ),
        |out: &str| u64::from_str_radix(out, 16)
    )(s)
}

#[derive(Debug, Clone)]
pub struct Constraints<'asm> {
    pub asm: &'asm InstructionAssembler,

    pub token_order: Vec<TokenId>,

    pub tokens: HashMap<TokenId, BV<'asm>>,
    pub fields: HashMap<TokenFieldId, BV<'asm>>,

    pub eqs: HashSet<Bool<'asm>>,
}

impl<'asm> Constraints<'asm> {
    pub fn new(asm: &'asm InstructionAssembler) -> Self {
        Self {
            asm,
            token_order: Vec::new(),
            tokens: HashMap::new(),
            fields: HashMap::new(),
            eqs: HashSet::new(),
        }
    }

    pub fn token(&mut self, token_id: TokenId) -> BV<'asm> {
        let token = self.asm.token(token_id);
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

    pub fn to_bytes(&self) -> Option<Vec<u8>> {
        log::debug!("Generating instruction bytes");
        let model = self.model()?;
        let mut instruction_bytes = vec![];
        for token_id in self.token_order.iter() {
            let token = self.asm.token(*token_id);
            let token_bv = self.tokens.get(token_id)?;
            let token_value = model.eval(token_bv, true)?.as_u64()?;
            let token_length = token.len_bytes().get() as usize;

            log::debug!("{}: {:#010X}/{}", token.name(), token_value, token_length);

            instruction_bytes.extend(match token.endian() {
                Endian::Little => token_value.to_le_bytes()[..token_length].to_vec(),
                Endian::Big => token_value.to_be_bytes()[8-token_length..].to_vec(),
            });
        }
        Some(instruction_bytes)
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
pub struct InstructionAssembler {
    sleigh: Sleigh,
    ctx: z3::Context,
}

impl Deref for InstructionAssembler {
    type Target = Sleigh;

    fn deref(&self) -> &Self::Target {
        &self.sleigh
    }
}

impl InstructionAssembler {
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
        if let Some(mneumonic) = constructor.display.mneumonic.as_ref() {
            s = nom::bytes::complete::tag(mneumonic.as_str())(s)?.0;
            log::trace!("MNEUMONIC: {}", mneumonic);
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
                    log::trace!("TOKEN_FIELD: {:?} {:?}", token_field.name(), s);
                    let token_field_bv = variables.token_field(*token_field_id, None);
                    let (s, value) = match token_field.attach {
                        TokenFieldAttach::NoAttach(value_fmt) => {
                            self.parse_value(value_fmt.signed, s)?
                        },
                        TokenFieldAttach::Varnode(attach_varnode_id) => {
                            let attach_varnode = self.attach_varnode(attach_varnode_id);
                            self.parse_attach_varnode(attach_varnode, s)?
                        },
                        TokenFieldAttach::Literal(_attach_literal_id) => todo!(),
                        TokenFieldAttach::Number(_print_base, _attach_number_id) => todo!(),
                    };
                    let ivalue = (value as isize) >> token_field_bv.get_size();
                    if ivalue != 0 && ivalue != -1 {
                        log::trace!("Immidiate out of range {} {} {}", token_field_bv.get_size(), value, ivalue);
                        return nom::combinator::fail(s);
                    }
                    // TODO: check value bits
                    let const_bv = variables.build_u64_const(value as u64, token_field_bv.get_size());
                    variables.eq(token_field_bv._eq(&const_bv));
                    s
                },
                DisplayElement::InstStart(_inst_start) => todo!(),
                DisplayElement::InstNext(_inst_next) => todo!(),
                DisplayElement::Table(table_id) => {
                    let table = self.table(*table_id);
                    log::trace!("TABLE: {:?}/{:?} {:?}", table.name(), table_id, s);
                    let (s, table_constraints) = self.assemble_table(table, variables.constraints.clone(), s)?;
                    variables.merge(table_constraints);
                    s
                },
                DisplayElement::Disassembly(variable_id) => {
                    let variable = constructor.pattern.disassembly_var(*variable_id);
                    log::trace!("DISASSEMBLY: {:?} {:?}", variable.name(), s);
                    let (s, value) = self.parse_value(true, s)?;
                    let var = variables.variable(*variable_id);
                    let const_bv = variables.build_u64_const(value as u64, var.get_size());
                    variables.eq(var._eq(&const_bv));
                    s

                },
                DisplayElement::Literal(lit) => {
                    log::trace!("LITERAL: {:?} {:?}", lit, s);
                    nom::bytes::complete::tag(lit.as_str())(s)?.0
                },
                DisplayElement::Space => {
                    log::trace!("SPACE: {:?}", s);
                    nom::character::complete::space1(s)?.0
                },
            };
        }

        if variables.check() {
            Ok((s, variables.constraints))
        } else {
            log::trace!("Constraint check failed, Eqs: {:#?}", variables.eqs);
            nom::combinator::fail(s)
        }
    }

    pub fn parse_value<'a>(&self, signed: bool, s: &'a str) -> IResult<&'a str, i64> {
        //TODO: labels?
        let (s, sign) = if signed {
            if let Some(s) = s.strip_prefix('-') {
                (s, true)
            } else {
                (s, false)
            }
        } else {
            (s, false)
        };
        let (s, value) = alt((parse_hex, parse_dec))(s)?;
        let value = if sign {
            -(value as i64)
        } else {
            value as i64
        };
        Ok((s, value))
    }

    pub fn parse_attach_varnode<'a>(&self, attach_varnode: &AttachVarnode, s: &'a str) -> IResult<&'a str, i64> {
        let mut attach_varnodes = attach_varnode.0.iter().map(|(value, id)|
            (*value, self.varnode(*id))
        ).collect::<Vec<_>>();
        attach_varnodes.sort_by(|(_, v1), (_, v2)| v2.name().cmp(v1.name()));
        for (value, varnode) in attach_varnodes.into_iter() {
            if let Some(s) = s.strip_prefix(varnode.name()) {
                return Ok((s, value as i64))
            }
        }
        nom::combinator::fail(s)
    }
}


#[cfg(test)]
mod test {
    use std::path::Path;
    use super::*;

    fn run_tests(slaspec_path: impl AsRef<Path>, tests: &[(&str, Vec<u8>, &str)]) {
        let _ = env_logger::try_init();
        log::info!("Loading slaspec: {:?}", slaspec_path.as_ref());
        let slaspec = sleigh_rs::file_to_sleigh(slaspec_path.as_ref())
            .expect(&format!("Could not load slaspec: {:?}", slaspec_path.as_ref()));
        let assembler = InstructionAssembler::new(slaspec);

        for (input, expected_bytes, expected_rest) in tests.iter() {
            log::info!("Assembling {:?} expecting {:02x?} {:?}", input, expected_bytes, expected_rest);

            let (rest, constraints) = assembler.assemble_instruction(input).unwrap();
            log::info!("Rest of input: {:?}", rest);

            let bytes = constraints.to_bytes().expect("Constraints failed to produce bytes");
            log::info!("Produced bytes: {:02x?}", bytes);

            assert_eq!(rest, *expected_rest);
            assert_eq!(bytes, *expected_bytes);
        }
    }

    #[test]
    pub fn test_vliw_assemble() {
        run_tests("examples/vliw.slaspec", &[
            ("{ unk.0x0 r1, r2, 0x1234 ; unk.0xa r5, r1, 0x1234 ; unk.0xb r10, r11, 0 }", vec![0x50, 0x04, 0x4a, 0x28, 0x56, 0xa5, 0x92, 0x34], ""),
            ("{ unk.0x0 r1, r2, 0xffffffff87654321 }", vec![0xc0, 0x04, 0x40, 0x00, 0x87, 0x65, 0x43, 0x21], "")
        ]);
    }

    #[test]
    pub fn test_risc_assemble() {
        run_tests("examples/risc.slaspec", &[
            ("xor r2, r15, 0xffff", vec![0x2f, 0x90, 0xff, 0xff], ""),
            ("add r4, r5, 0x1234", vec![0x02, 0xa0, 0x12, 0x34], ""),
            ("add r4, r5, 0x1234", vec![0x02, 0xa0, 0x12, 0x34], ""),
            ("xor r4, r5, 0x23450000", vec![0x2a, 0xa2, 0x23, 0x45], ""),
            ("and r1, r2, r3", vec![0xf9, 0x09, 0x80, 0x03], ""),
        ]);
    }
}