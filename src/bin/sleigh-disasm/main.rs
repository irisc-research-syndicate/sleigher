use anyhow::{Context as _, Result};
use sleigher::context::{Context, ContextFlow};
use sleigher::disassembler::Disassembler;
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
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    let sleigh = sleigh_rs::file_to_sleigh(&args.slaspec)
        .ok()
        .context("Could not open or parse slaspec")?;
    let disassembler = Disassembler::new(&sleigh);
    let mut flow = ContextFlow::new(Context::parse_values(&sleigh, &args.context)?);

    let code = std::fs::read(args.code)?;

    let mut pc = args.address;
    let mut cursor = &code[..];

    loop {
        let context = flow.at(&sleigh, pc);
        let Ok(instruction) = disassembler.disassemble(pc, &context, cursor) else {
            break;
        };
        println!("{:#010x}: {}", pc, instruction);
        if args.pcode {
            match &instruction.pcode {
                Ok(ops) => {
                    for op in ops.iter() {
                        println!("            {}", op.display(&sleigh));
                    }
                }
                Err(err) => println!("            <{}>", err),
            }
        }
        pc += instruction.len as u64;
        cursor = &cursor[instruction.len..];
        flow.advance(&sleigh, context, &instruction.commits);
    }

    Ok(())
}
