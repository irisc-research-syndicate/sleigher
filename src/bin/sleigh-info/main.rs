use std::path::PathBuf;

use clap::Parser;
use anyhow::{anyhow, Result}; 

use sleigher::SleighSleigh;

#[derive(Debug, Parser)]
struct Args {
    slaspec: PathBuf,

    #[arg(short, long, default_value_t=false)]
    tables: bool,

    #[arg(short='f', long, default_value_t=false)]
    token_fields: bool,

    #[arg(short='v', long, default_value_t=false)]
    varnodes: bool,
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
    if args.token_fields {
        for token_field in sleigh.token_fields() {
            println!("{}\t{}\t{}\t{}", token_field.name(), token_field, token_field.inner.raw_value_is_signed(), token_field.execution_value_is_signed());
        }
    }
    if args.varnodes {
        for varnode in sleigh.inner.varnodes() {
            let space = sleigh.inner.space(varnode.space);
            println!("{}\t{:?}/{}\t{:#018x}", varnode.name(), space.space_type, varnode.space.0, varnode.address);
        }
        for foo in sleigh.inner.attach_numbers() {
            dbg!(foo);
        }
    }
    Ok(())
}