use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use sleigher::assembler::InstructionAssembler;

#[derive(Debug, Clone, Parser)]
struct Cli {
    slaspec: PathBuf,
    instruction: String,
}

pub fn main() -> Result<()> {
    env_logger::init();

    let args = Cli::parse();
    let sleigh = sleigh_rs::file_to_sleigh(&args.slaspec.clone()).ok().context("Could not open or parse slaspec")?;
    let assembler = InstructionAssembler::new(sleigh);
    let (rest, constraints) = assembler.assemble_instruction(&args.instruction).ok().context("Failed to parse instruction")?;

    println!("rest: {:?}", rest);

    println!("token_order: {:?}", constraints.token_order);
    println!("tokens: {:?}", constraints.tokens);
    println!("fields: {:#?}", constraints.fields.values());
    println!("eqs: {:#?}", constraints.eqs);
    println!("model: {:#?}", constraints.model());
    println!("bytes: {:02x?}", constraints.to_bytes());

    Ok(())
}
