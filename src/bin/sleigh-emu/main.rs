#![allow(unused_variables)]
//#![allow(dead_code)]

use std::{fs::File, path::PathBuf};

use clap::Parser;

use anyhow::{bail, Context as AnyhowContext, Result};

use sleigh_rs::SpaceId;

use sleigher::emulator::{Cpu, State};
use sleigher::space::{FileRegion, HashSpace, MappedSpace};
use sleigher::value::{Address, Ref};

#[derive(Debug, Clone)]
struct FileMap {
    address: u64,
    path: PathBuf,
}

fn parse_int(s: &str) -> std::result::Result<u64, std::num::ParseIntError> {
    if let Some(s) = s.strip_prefix("0x") {
        u64::from_str_radix(s, 16)
    } else {
        u64::from_str_radix(s, 10)
    }
}

impl std::str::FromStr for FileMap {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let (address, path) = s
            .split_once(":")
            .context("File mapping does not contain ':'")?;
        let address = parse_int(address)?;
        let path = PathBuf::from_str(path)?;
        Ok(FileMap { address, path })
    }
}

#[derive(Debug, Clone)]
struct RamMap {
    address: u64,
    length: u64,
}

impl std::str::FromStr for RamMap {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let (address, length) = s
            .split_once(":")
            .context("Ram mapping does not contain ':'")?;
        let address = parse_int(address)?;
        let length = parse_int(length)?;
        Ok(RamMap { address, length })
    }
}

#[derive(Debug, Clone)]
struct RegAssignment {
    name: String,
    value: u64,
}

impl std::str::FromStr for RegAssignment {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let (name, value) = s
            .split_once("=")
            .context("register assignemnt must contain '='")?;
        let name = name.to_string();
        let value = parse_int(value)?;
        Ok(RegAssignment { name, value })
    }
}

#[derive(Debug, Clone)]
struct Breakpoint {
    address: u64,
    name: String,
    regs: Vec<String>,
}

impl std::str::FromStr for Breakpoint {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let (address, name_and_regs) = s
            .split_once(':')
            .context("Breakpoint must be on the form: address:name:reglist")?;
        let (name, regs) = name_and_regs
            .split_once(":")
            .context("Breakpoint must be on the form: address:name:reglist")?;
        let address = parse_int(address)?;
        let name = name.to_string();
        let regs = regs
            .split(',')
            .map(|reg| reg.to_string())
            .collect::<Vec<String>>();

        Ok(Self {
            address,
            name,
            regs,
        })
    }
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    slaspec: PathBuf,

    #[arg(short = 'f', long = "map")]
    file_mappings: Vec<FileMap>,

    #[arg(short = 'm', long = "ram")]
    ram_mappings: Vec<RamMap>,

    #[arg(short='e', long="entrypoint", value_parser=parse_int)]
    entrypoint: u64,

    #[arg(short = 'r', long = "reg")]
    registers: Vec<RegAssignment>,

    #[arg(short = 'b', long = "breakpoint")]
    breakpoints: Vec<Breakpoint>,

    #[arg(short='s', long="steps", value_parser=parse_int)]
    steps: Option<u64>,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();
    let sleigh = match sleigh_rs::file_to_sleigh(&args.slaspec) {
        Ok(sleigh) => sleigh,
        Err(err) => {
            println!("Could not open or parse slaspec: {:#?}", err);
            bail!("Could not open or pasrse slaspec");
        }
    };

    let mut memory = MappedSpace::new();
    for file_map in args.file_mappings {
        let file = File::open(&file_map.path)
            .context(format!("Could not open file: {:?}", &file_map.path))?;
        let size = file.metadata()?.len();
        memory.add_mapping(Address(file_map.address), size, Box::new(FileRegion(file)))
    }

    for ram_map in args.ram_mappings {
        memory.add_mapping(
            Address(ram_map.address),
            ram_map.length,
            Box::new(HashSpace::new()),
        )
    }

    let mut state = State::new();
    state.spaces.insert(SpaceId(0), Box::new(memory));
    state.pc = args.entrypoint;

    log::info!("=== Initializing registers ===");
    for reg in args.registers {
        let varnode = sleigh
            .varnodes()
            .iter()
            .find(|varnode| varnode.name() == reg.name)
            .unwrap();
        state.write_ref(varnode.into(), &reg.value.to_be_bytes())?;
    }

    let mut cpu = Cpu::new(&sleigh, state);

    fn read_reg_u32(cpu: &mut Cpu, name: &str) -> u32 {
        let varnode = cpu
            .sleigh
            .varnodes()
            .iter()
            .find(|varnode| varnode.name() == name)
            .unwrap();
        cpu.state.read_ref_u32be(varnode.into()).unwrap()
    }

    for step in args.steps.map(|steps| 0..steps).unwrap_or(0..u64::MAX) {
        let pc = cpu.state.pc;

        if pc == 0u64 {
            log::info!("PC = null! Exiting!");
            break;
        }

        for bp in &args.breakpoints {
            if bp.address == pc {
                let regs = bp
                    .regs
                    .iter()
                    .map(|reg| format!("{}={:#010x}", reg, read_reg_u32(&mut cpu, reg.as_str())))
                    .collect::<Vec<_>>()
                    .join(" ");
                log::info!("Breakpoint {:#010x}: {} {}", bp.address, bp.name, regs);
            }
        }

        cpu.step()?;
    }

    let mut hash_bytes = [0u8; 32];
    cpu.state
        .read_ref(Ref(SpaceId(0), 32, Address(0xffffff80u64)), &mut hash_bytes)?;
    log::info!("{:02x?}", hash_bytes);

    Ok(())
}
