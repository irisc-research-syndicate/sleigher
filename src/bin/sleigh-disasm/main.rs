use sleigher::SleighSleigh;
use anyhow::{anyhow, bail, Result};
use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    slaspec: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let sleigh = match sleigh_rs::file_to_sleigh(&args.slaspec) {
        Ok(sleigh) => sleigh,
        Err(err) => {
            dbg!(err);
            bail!("Failed to open or parse slaspec");
        },
    };
    let sleigh: SleighSleigh = (&sleigh).into();
    let instrs = vec![
        0x4a, 0x04, 0x08, 0x00,
        0x00, 0x21, 0x03, 0x98,
        0x11, 0x22, 0x33, 0x44,
        0x94, 0xff, 0xf3, 0x86,
    ];
    for instr in instrs.chunks(4) {
        let instruction_table = sleigh.instruction_table();
        let instruction = instruction_table.disassemble(0, &instr).unwrap();
        println!("{}", instruction);
    }

    Ok(())
}
