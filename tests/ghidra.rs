//! Known-good and known-bad examples checked against real Ghidra slaspecs.
//!
//! Needs a Ghidra source tree (only `Ghidra/Processors/*/data/languages` is used):
//!
//!     GHIDRA_SRC=/path/to/ghidra cargo test --test ghidra -- --ignored --nocapture
//!
//! `GHIDRA_SPEC=<substring>` restricts the run to matching specs.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::Command;

use sleigher::assembler::{AsmError, InstructionAssembler};
use sleigher::context::Context;
use sleigher::disassembler::Disassembler;

enum Case {
    /// Bytes and the text Ghidra displays for them, checked in both directions
    Good(&'static [u8], &'static str),
    /// Bytes that must not disassemble
    BadBytes(&'static [u8]),
    /// Text that must not assemble
    BadAsm(&'static str),
}
use Case::*;

/// `Good` and `BadBytes` cases were checked with Ghidra's own C++ SLEIGH engine (Ghidra 918d44ebc2), using the
/// default context from each processor's `.pspec`. `BadAsm` cases are malformed or not encodable in that mode.
#[rustfmt::skip]
const CORPUS: &[(&str, &[Case])] = &[
    ("x86/data/languages/x86.slaspec", &[
        Good(&[0x90], "NOP"),
        Good(&[0xc3], "RET"),
        Good(&[0x55], "PUSH EBP"),
        Good(&[0x50], "PUSH EAX"),
        Good(&[0x66, 0x50], "PUSH AX"),
        Good(&[0x06], "PUSH ES"),
        Good(&[0x89, 0xe5], "MOV EBP,ESP"),
        Good(&[0x31, 0xc0], "XOR EAX,EAX"),
        BadBytes(&[0x0f, 0x04]),
        BadAsm("MOV EAX,"),
        BadAsm("FOO EAX"),
    ]),
    ("x86/data/languages/x86-64.slaspec", &[
        Good(&[0x48, 0x89, 0xe5], "MOV RBP,RSP"),
        Good(&[0xc3], "RET"),
        Good(&[0x50], "PUSH RAX"),
        Good(&[0x41, 0x50], "PUSH R8"),
        BadBytes(&[0x06]),
        BadAsm("PUSH EAX"),
    ]),
    ("ARM/data/languages/ARM8_le.slaspec", &[
        Good(&[0x1e, 0xff, 0x2f, 0xe1], "bx lr"),
        Good(&[0x04, 0xe0, 0x2d, 0xe5], "str lr,[sp,#-0x4]!"),
        Good(&[0xf0, 0x00, 0xf0, 0xe7], "udf #0x0"),
        BadBytes(&[0xff, 0xff, 0xff, 0xff]),
        BadAsm("bx"),
    ]),
    ("AARCH64/data/languages/AARCH64.slaspec", &[
        Good(&[0xc0, 0x03, 0x5f, 0xd6], "ret"),
        Good(&[0x1f, 0x20, 0x03, 0xd5], "nop"),
        Good(&[0x00, 0x00, 0x00, 0x00], "udf 0x0"),
        BadBytes(&[0xff, 0xff, 0xff, 0xff]),
    ]),
    ("MIPS/data/languages/mips32be.slaspec", &[
        Good(&[0x00, 0x00, 0x00, 0x00], "nop"),
        Good(&[0x24, 0x02, 0x00, 0x01], "li v0, 0x1"),
        Good(&[0x03, 0xe0, 0x00, 0x08], "jr ra"),
        BadBytes(&[0xfc, 0x00, 0x00, 0x00]),
    ]),
    ("RISCV/data/languages/riscv.lp64d.slaspec", &[
        Good(&[0x67, 0x80, 0x00, 0x00], "ret"),
        Good(&[0x93, 0x02, 0x10, 0x00], "li t0,0x1"),
        BadBytes(&[0x00, 0x00, 0x00, 0x00]),
        BadBytes(&[0xff, 0xff, 0xff, 0xff]),
    ]),
    ("PowerPC/data/languages/ppc_32_be.slaspec", &[
        Good(&[0x4e, 0x80, 0x00, 0x20], "blr"),
        Good(&[0x60, 0x00, 0x00, 0x00], "ori r0,r0,0x0"),
        BadBytes(&[0x00, 0x00, 0x00, 0x00]),
    ]),
];

thread_local! {
    /// Location of the last panic on this thread, recorded by the hook installed in `run_corpus`
    static PANIC_LOCATION: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Runs `f`, turning a panic into an error carrying the panic message and location
fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        let location = PANIC_LOCATION.with(|l| l.take());
        Err(format!("PANIC at {}: {}", location, msg))
    })
}

fn check_disassembly(asm: &InstructionAssembler, bytes: &[u8], text: &str) -> Result<(), String> {
    let disasm = Disassembler::new(asm);
    match disasm.disassemble(0, &Context::new(asm), bytes) {
        Ok(inst) if inst.to_string() == text => Ok(()),
        Ok(inst) => Err(format!("got {:?}", inst.to_string())),
        Err(err) => Err(format!("error: {}", err)),
    }
}

fn check_assembly(asm: &InstructionAssembler, text: &str, bytes: &[u8]) -> Result<(), String> {
    match asm.assemble_instruction(text) {
        Ok(constraints) => match constraints.to_bytes() {
            Some(got) if got == bytes => Ok(()),
            got => Err(format!("got {:02x?}", got)),
        },
        Err(AsmError::Ambiguous(candidates)) if candidates.iter().any(|c| c.bytes == bytes) => {
            Ok(())
        }
        Err(err) => Err(format!("error: {}", err)),
    }
}

fn check_bad_bytes(asm: &InstructionAssembler, bytes: &[u8]) -> Result<(), String> {
    match Disassembler::new(asm).disassemble(0, &Context::new(asm), bytes) {
        Ok(inst) => Err(format!("disassembled to {:?}", inst.to_string())),
        Err(_) => Ok(()),
    }
}

fn check_bad_asm(asm: &InstructionAssembler, text: &str) -> Result<(), String> {
    match asm.assemble_instruction(text) {
        Err(AsmError::NoMatch) => Ok(()),
        Ok(constraints) => Err(format!("assembled to {:02x?}", constraints.to_bytes())),
        Err(err) => Err(format!("error: {}", err)),
    }
}

/// Checks every case of one spec, printing a `RUN` line before and a `PASS`/`FAIL` line after
/// each check so the parent can attribute a crash (e.g. a stack overflow) to the check causing it
fn run_spec(ghidra: &Path, spec: &str, cases: &[Case]) {
    // Panics are reported per check, so record the location instead of printing to stderr
    std::panic::set_hook(Box::new(|info| {
        let location = info.location().map(|l| l.to_string()).unwrap_or_default();
        PANIC_LOCATION.with(|l| *l.borrow_mut() = location);
    }));
    let report = |kind: &str, input: &str, check: &mut dyn FnMut() -> Result<(), String>| {
        println!("RUN\t{kind}\t{input}");
        match guarded(check) {
            Ok(()) => println!("PASS\t{kind}\t{input}"),
            Err(err) => {
                let err = err.replace(&format!("{}/", ghidra.display()), "");
                println!(
                    "FAIL\t{kind}\t{input}\t{}",
                    err.lines().next().unwrap_or_default()
                );
            }
        }
    };

    let path = ghidra.join("Ghidra/Processors").join(spec);
    let mut asm = None;
    report("load", "", &mut || {
        let sleigh = sleigh_rs::file_to_sleigh(&path).map_err(|err| err.to_string())?;
        asm = Some(InstructionAssembler::new(sleigh));
        Ok(())
    });
    let Some(asm) = asm else { return };

    for case in cases {
        match case {
            Good(bytes, text) => {
                report("disasm", &format!("{:02x?}", bytes), &mut || {
                    check_disassembly(&asm, bytes, text)
                });
                report("asm", text, &mut || check_assembly(&asm, text, bytes));
            }
            BadBytes(bytes) => report("bad-bytes", &format!("{:02x?}", bytes), &mut || {
                check_bad_bytes(&asm, bytes)
            }),
            BadAsm(text) => report("bad-asm", text, &mut || check_bad_asm(&asm, text)),
        }
    }
}

/// Runs one spec in a child process, returning its result lines as `(status, kind, input, detail)`
fn spawn_spec(spec: &str) -> Vec<[String; 4]> {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["ghidra_corpus", "--exact", "--ignored", "--nocapture"])
        .env(CHILD_ENV, spec)
        .output()
        .expect("could not run child test process");

    let mut results: Vec<[String; 4]> = vec![];
    let mut running = None;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let mut fields = line.splitn(4, '\t').map(str::to_string);
        let (Some(status), Some(kind), Some(input)) = (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let detail = fields.next().unwrap_or_default();
        match status.as_str() {
            "RUN" => running = Some([status, kind, input, detail]),
            "PASS" | "FAIL" => {
                running = None;
                results.push([status, kind, input, detail]);
            }
            _ => {}
        }
    }
    if let Some([_, kind, input, _]) = running {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let reason = stderr.lines().find(|l| !l.is_empty()).unwrap_or_default();
        let detail = format!("CRASH ({}): {}", output.status, reason);
        results.push(["FAIL".to_string(), kind, input, detail]);
    }
    results
}

const CHILD_ENV: &str = "GHIDRA_CORPUS_SPEC";

#[test]
#[ignore = "needs GHIDRA_SRC"]
fn ghidra_corpus() {
    let Some(ghidra) = std::env::var_os("GHIDRA_SRC").map(PathBuf::from) else {
        eprintln!("GHIDRA_SRC not set, skipping");
        return;
    };

    if let Ok(spec) = std::env::var(CHILD_ENV) {
        let (spec, cases) = CORPUS.iter().find(|(s, _)| *s == spec).unwrap();
        // Parsing real specs recurses deeper than the default test thread stack allows
        std::thread::Builder::new()
            .stack_size(1 << 30)
            .spawn(move || run_spec(&ghidra, spec, cases))
            .unwrap()
            .join()
            .unwrap();
        return;
    }

    let filter = std::env::var("GHIDRA_SPEC").unwrap_or_default();
    let (mut checks, mut failed) = (0, 0);
    for (spec, _) in CORPUS.iter().filter(|(spec, _)| spec.contains(&filter)) {
        let path = ghidra.join("Ghidra/Processors").join(spec);
        assert!(path.exists(), "missing slaspec: {:?}", path);
        let name = path.file_name().unwrap().to_string_lossy();
        for [status, kind, input, detail] in spawn_spec(spec) {
            checks += 1;
            failed += (status == "FAIL") as usize;
            println!("{status} {name:<24} {kind:<9} {input:<28} {detail}");
        }
    }
    println!("{} of {} checks failed", failed, checks);
    assert_eq!(failed, 0, "{} of {} checks failed", failed, checks);
}
