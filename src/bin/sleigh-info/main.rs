use std::path::PathBuf;

use clap::Parser;
use anyhow::{anyhow, Result}; 

use sleigher::SleighSleigh;

#[derive(Debug, Parser)]
struct Args {
    slaspec: PathBuf,

    #[arg(short, long, default_value_t=false)]
    tables: bool
}

fn main() -> Result<()> {
    let args = Args::parse();
    let binding = sleigh_rs::file_to_sleigh(&args.slaspec).ok().ok_or(anyhow!("Could not open or parse slaspec"))?;
    let sleigh: SleighSleigh = (&binding).into();
    if args.tables {
        for table in sleigh.tables() {
            for constructor in table.constructors() {
                for (_, _, bitconstraint) in constructor.variants(){
                    println!("{}:\t{}\t{}", table.name(), bitconstraint, constructor)
                }
            }
        }
    }
    Ok(())
}