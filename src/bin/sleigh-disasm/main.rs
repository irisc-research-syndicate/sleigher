use anyhow::{Context as _, Result};
use sleigher::context::Context;
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
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    let sleigh = sleigh_rs::file_to_sleigh(&args.slaspec)
        .ok()
        .context("Could not open or parse slaspec")?;
    let disassembler = Disassembler::new(&sleigh);
    let context = Context::new(&sleigh);

    let code = std::fs::read(args.code)?;

    let mut pc = args.address;
    let mut cursor = &code[..];

    while let Ok(instruction) = disassembler.disassemble(pc, &context, cursor) {
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
    }

    Ok(())
}
