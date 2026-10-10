use anyhow::{Context as _, Result};
use sleigh_rs::Sleigh;
use sleigher::context::{Context, ContextFlow};
use sleigher::disassembler::{DisassembledInstruction, Disassembler};
use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    slaspec: PathBuf,

    #[clap(short, long, default_value_t = 0)]
    address: u64,

    code: PathBuf,

    /// Print each instruction's p-code below it
    #[clap(short, long)]
    pcode: bool,

    /// Start from a context variable value, as name=value; may be repeated
    #[clap(short, long)]
    context: Vec<String>,

    /// Follow control flow from the start address instead of sweeping linearly, keeping the
    /// context by address as Ghidra's disassembler does
    #[clap(long)]
    flow: bool,
}

fn print(sleigh: &Sleigh, instruction: &DisassembledInstruction, pcode: bool) {
    println!("{:#010x}: {}", instruction.inst_start, instruction);
    if pcode {
        match &instruction.pcode {
            Ok(ops) => {
                for op in ops.iter() {
                    println!("            {}", op.display(sleigh));
                }
            }
            Err(err) => println!("            <{}>", err),
        }
    }
    // Delay slot instructions are marked as in Ghidra, their p-code is in the branch's
    for slot in instruction.delay_slots.iter() {
        println!("{:#010x}: _{}", slot.inst_start, slot);
    }
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    let sleigh = sleigh_rs::file_to_sleigh(&args.slaspec)
        .ok()
        .context("Could not open or parse slaspec")?;
    let disassembler = Disassembler::new(&sleigh);
    let context = Context::parse_values(&sleigh, &args.context)?;

    let code = std::fs::read(args.code)?;

    if args.flow {
        let instructions =
            disassembler.disassemble_flow(args.address, &code, args.address, &context)?;
        for instruction in instructions.values() {
            print(&sleigh, instruction, args.pcode);
        }
        return Ok(());
    }

    let mut flow = ContextFlow::new(context);
    let mut pc = args.address;
    let mut cursor = &code[..];

    loop {
        let Ok(instruction) = disassembler.disassemble_in_flow(pc, &mut flow, cursor) else {
            break;
        };
        print(&sleigh, &instruction, args.pcode);
        let len = (instruction.fallthrough() - pc) as usize;
        pc += len as u64;
        cursor = &cursor[len..];
    }

    Ok(())
}
