use std::path::PathBuf;

use anyhow::{bail, Context as _, Result};
use clap::Parser;
use sleigher::assembler::{InstructionAssembler, Labels};
use sleigher::context::Context;

fn parse_int(s: &str) -> std::result::Result<u64, std::num::ParseIntError> {
    if let Some(s) = s.strip_prefix("0x") {
        u64::from_str_radix(s, 16)
    } else {
        s.parse::<u64>()
    }
}

#[derive(Debug, Clone, Parser)]
struct Cli {
    slaspec: PathBuf,

    /// A single instruction to assemble
    instruction: Option<String>,

    /// Assemble a program file: one instruction per line, `name:` labels and `//` comments
    #[arg(short, long, conflicts_with = "instruction")]
    file: Option<PathBuf>,

    /// Address of the first instruction
    #[arg(short, long, default_value = "0", value_parser = parse_int)]
    base: u64,

    /// Write the program's raw bytes to this file
    #[arg(short, long, requires = "file")]
    output: Option<PathBuf>,

    /// Start from a context variable value, as name=value; may be repeated
    #[arg(short, long)]
    context: Vec<String>,
}

pub fn main() -> Result<()> {
    env_logger::init();

    let args = Cli::parse();
    let sleigh = sleigh_rs::file_to_sleigh(&args.slaspec.clone())
        .ok()
        .context("Could not open or parse slaspec")?;
    let assembler = InstructionAssembler::new(sleigh);
    let context = Context::parse_values(&assembler, &args.context)?;

    if let Some(path) = &args.file {
        let source =
            std::fs::read_to_string(path).with_context(|| format!("Could not read {:?}", path))?;
        let program = assembler.assemble_program_in_context(&source, args.base, &context)?;

        for line in program.lines.iter() {
            let bytes = line
                .bytes
                .iter()
                .map(|byte| format!("{:02x}", byte))
                .collect::<Vec<_>>()
                .join(" ");
            println!("{:#010x}: {:<24} {}", line.address, bytes, line.source);
        }
        if !program.labels.is_empty() {
            println!();
            for (name, address) in program.labels.iter() {
                println!("{:#010x} {}", address, name);
            }
        }

        if let Some(output) = &args.output {
            std::fs::write(output, &program.bytes)
                .with_context(|| format!("Could not write {:?}", output))?;
        }
        return Ok(());
    }

    let Some(instruction) = &args.instruction else {
        bail!("Give an instruction or --file");
    };
    let labels = Labels::new();
    let constraints =
        assembler.assemble_instruction_at(instruction, args.base, &context, &labels)?;

    println!("tokens: {:?}", constraints.tokens);
    println!("fields: {:#?}", constraints.fields.values());
    println!("eqs: {:#?}", constraints.eqs);
    println!("model: {:#?}", constraints.model());
    println!("bytes: {:02x?}", constraints.to_bytes());

    Ok(())
}
