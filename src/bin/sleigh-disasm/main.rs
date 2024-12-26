use anyhow::{Context as AnyhowContext, Result};
use sleigher::disassembler::{Context, Disassembler};
use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    slaspec: PathBuf,

    #[clap(short, long, default_value_t = 0)]
    address: u64,

    code: PathBuf,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    let sleigh = sleigh_rs::file_to_sleigh(&args.slaspec)
        .ok()
        .context("Could not open or parse slaspec")?;
    let disassembler = Disassembler::new(&sleigh);

    let code = std::fs::read(args.code)?;

    let mut pc = args.address;
    let mut cursor = &code[..];

    while let Ok(instruction) = disassembler.disassemble(pc, Context, cursor) {
        println!("{:#010x}: {}", pc, instruction);
        pc += instruction.len() as u64;
        cursor = &cursor[instruction.len()..];
    }

    Ok(())
}
