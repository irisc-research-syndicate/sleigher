//! Known-good and known-bad examples checked against real Ghidra slaspecs.
//!
//! Needs a Ghidra source tree (only `Ghidra/Processors/*/data/languages` is used):
//!
//!     GHIDRA_SRC=/path/to/ghidra cargo test --release --test ghidra -- --ignored
//!
//! Every check is its own test, e.g. `x86_64::ret::asm`, so the usual test filters select specs or
//! checks. Each runs in a child process, so a crash or a check running longer than
//! `GHIDRA_TIMEOUT` seconds (default 60) or more than `GHIDRA_MEMORY_MB` of address space
//! (default 1024) fails only that test.
//!
//! `good!` and `bad_bytes!` cases were checked with Ghidra's own C++ SLEIGH engine (Ghidra
//! 918d44ebc2), using the default context from each processor's `.pspec`. `bad_asm!` cases are
//! malformed or not encodable in that mode.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use sleigher::assembler::{AsmError, InstructionAssembler, Labels};
use sleigher::context::Context;
use sleigher::disassembler::Disassembler;

struct Spec {
    /// Path under `Ghidra/Processors`
    path: &'static str,
    /// Initial context, the `context_set` defaults from the processor's `.pspec`
    context: &'static [(&'static str, i64)],
}

#[derive(Clone, Copy)]
enum Check {
    Load,
    Disasm(&'static [u8], &'static str),
    Asm(&'static str, &'static [u8]),
    BadBytes(&'static [u8]),
    BadAsm(&'static str),
}

/// A module of tests for one spec, with a `load` test and the cases given in its body
macro_rules! spec {
    ($name:ident, $path:literal, [$(($var:literal, $val:literal)),*], { $($cases:tt)* }) => {
        mod $name {
            use super::*;

            const SPEC: Spec = Spec { path: $path, context: &[$(($var, $val)),*] };

            #[test]
            #[ignore = "needs GHIDRA_SRC"]
            fn load() {
                run(&SPEC, concat!(module_path!(), "::load"), Check::Load);
            }

            $($cases)*
        }
    };
}

/// Bytes and the text Ghidra displays for them: a `disasm` and an `asm` test
macro_rules! good {
    ($name:ident, [$($byte:literal),*], $text:literal) => {
        mod $name {
            use super::*;

            #[test]
            #[ignore = "needs GHIDRA_SRC"]
            fn disasm() {
                let check = Check::Disasm(&[$($byte),*], $text);
                run(&SPEC, concat!(module_path!(), "::disasm"), check);
            }

            #[test]
            #[ignore = "needs GHIDRA_SRC"]
            fn asm() {
                let check = Check::Asm($text, &[$($byte),*]);
                run(&SPEC, concat!(module_path!(), "::asm"), check);
            }
        }
    };
}

/// Bytes that must not disassemble
macro_rules! bad_bytes {
    ($name:ident, [$($byte:literal),*]) => {
        #[test]
        #[ignore = "needs GHIDRA_SRC"]
        fn $name() {
            let check = Check::BadBytes(&[$($byte),*]);
            run(&SPEC, concat!(module_path!(), "::", stringify!($name)), check);
        }
    };
}

/// Text that must not assemble
macro_rules! bad_asm {
    ($name:ident, $text:literal) => {
        #[test]
        #[ignore = "needs GHIDRA_SRC"]
        fn $name() {
            let check = Check::BadAsm($text);
            run(
                &SPEC,
                concat!(module_path!(), "::", stringify!($name)),
                check,
            );
        }
    };
}

spec!(
    x86,
    "x86/data/languages/x86.slaspec",
    [("addrsize", 1), ("opsize", 1)],
    {
        good!(nop, [0x90], "NOP");
        good!(ret, [0xc3], "RET");
        good!(push_ebp, [0x55], "PUSH EBP");
        good!(push_eax, [0x50], "PUSH EAX");
        good!(push_ax, [0x66, 0x50], "PUSH AX");
        good!(push_es, [0x06], "PUSH ES");
        good!(mov_ebp_esp, [0x89, 0xe5], "MOV EBP,ESP");
        good!(xor_eax_eax, [0x31, 0xc0], "XOR EAX,EAX");
        bad_bytes!(bad_bytes_0f04, [0x0f, 0x04]);
        bad_asm!(bad_asm_missing_operand, "MOV EAX,");
        bad_asm!(bad_asm_unknown_mnemonic, "FOO EAX");
    }
);

spec!(
    x86_64,
    "x86/data/languages/x86-64.slaspec",
    [
        ("addrsize", 2),
        ("opsize", 1),
        ("rexprefix", 0),
        ("longMode", 1)
    ],
    {
        good!(mov_rbp_rsp, [0x48, 0x89, 0xe5], "MOV RBP,RSP");
        good!(ret, [0xc3], "RET");
        good!(push_rax, [0x50], "PUSH RAX");
        good!(push_r8, [0x41, 0x50], "PUSH R8");
        bad_bytes!(bad_bytes_push_es, [0x06]);
        bad_asm!(bad_asm_push_eax, "PUSH EAX");
    }
);

spec!(
    arm8_le,
    "ARM/data/languages/ARM8_le.slaspec",
    [("TMode", 0), ("LRset", 0)],
    {
        good!(bx_lr, [0x1e, 0xff, 0x2f, 0xe1], "bx lr");
        good!(
            str_lr_pre_index,
            [0x04, 0xe0, 0x2d, 0xe5],
            "str lr,[sp,#-0x4]!"
        );
        good!(udf, [0xf0, 0x00, 0xf0, 0xe7], "udf #0x0");
        bad_bytes!(bad_bytes_ffffffff, [0xff, 0xff, 0xff, 0xff]);
        bad_asm!(bad_asm_missing_operand, "bx");
    }
);

spec!(aarch64, "AARCH64/data/languages/AARCH64.slaspec", [], {
    good!(ret, [0xc0, 0x03, 0x5f, 0xd6], "ret");
    good!(nop, [0x1f, 0x20, 0x03, 0xd5], "nop");
    good!(udf, [0x00, 0x00, 0x00, 0x00], "udf 0x0");
    bad_bytes!(bad_bytes_ffffffff, [0xff, 0xff, 0xff, 0xff]);
});

spec!(
    mips32be,
    "MIPS/data/languages/mips32be.slaspec",
    [("PAIR_INSTRUCTION_FLAG", 0), ("RELP", 1)],
    {
        good!(nop, [0x00, 0x00, 0x00, 0x00], "nop");
        good!(li_v0, [0x24, 0x02, 0x00, 0x01], "li v0, 0x1");
        good!(jr_ra, [0x03, 0xe0, 0x00, 0x08], "jr ra");
        bad_bytes!(bad_bytes_fc000000, [0xfc, 0x00, 0x00, 0x00]);
    }
);

spec!(
    riscv_lp64d,
    "RISCV/data/languages/riscv.lp64d.slaspec",
    [("MXL", 2)],
    {
        good!(ret, [0x67, 0x80, 0x00, 0x00], "ret");
        good!(li_t0, [0x93, 0x02, 0x10, 0x00], "li t0,0x1");
        bad_bytes!(bad_bytes_00000000, [0x00, 0x00, 0x00, 0x00]);
        bad_bytes!(bad_bytes_ffffffff, [0xff, 0xff, 0xff, 0xff]);
    }
);

spec!(
    ppc_32_be,
    "PowerPC/data/languages/ppc_32_be.slaspec",
    [("linkreg", 0)],
    {
        good!(blr, [0x4e, 0x80, 0x00, 0x20], "blr");
        good!(ori_r0, [0x60, 0x00, 0x00, 0x00], "ori r0,r0,0x0");
        bad_bytes!(bad_bytes_00000000, [0x00, 0x00, 0x00, 0x00]);
    }
);

fn check_disassembly(
    asm: &InstructionAssembler,
    ctx: &Context,
    bytes: &[u8],
    text: &str,
) -> Result<(), String> {
    let disasm = Disassembler::new(asm);
    match disasm.disassemble(0, ctx, bytes) {
        Ok(inst) if inst.to_string() == text => Ok(()),
        Ok(inst) => Err(format!("got {:?}", inst.to_string())),
        Err(err) => Err(format!("error: {}", err)),
    }
}

fn check_assembly(
    asm: &InstructionAssembler,
    ctx: &Context,
    text: &str,
    bytes: &[u8],
) -> Result<(), String> {
    match asm.assemble_instruction_at(text, 0, ctx, &Labels::new()) {
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

fn check_bad_bytes(asm: &InstructionAssembler, ctx: &Context, bytes: &[u8]) -> Result<(), String> {
    match Disassembler::new(asm).disassemble(0, ctx, bytes) {
        Ok(inst) => Err(format!("disassembled to {:?}", inst.to_string())),
        Err(_) => Ok(()),
    }
}

fn check_bad_asm(asm: &InstructionAssembler, ctx: &Context, text: &str) -> Result<(), String> {
    match asm.assemble_instruction_at(text, 0, ctx, &Labels::new()) {
        Err(AsmError::NoMatch) => Ok(()),
        Ok(constraints) => Err(format!("assembled to {:02x?}", constraints.to_bytes())),
        Err(err) => Err(format!("error: {}", err)),
    }
}

fn run_check(ghidra: &Path, spec: &Spec, check: Check) -> Result<(), String> {
    let path = ghidra.join("Ghidra/Processors").join(spec.path);
    let sleigh = sleigh_rs::file_to_sleigh(&path).map_err(|err| format!("load: {}", err))?;
    let ctx = Context::from_values(&sleigh, spec.context).map_err(|err| err.to_string())?;
    let asm = InstructionAssembler::new(sleigh);
    match check {
        Check::Load => Ok(()),
        Check::Disasm(bytes, text) => check_disassembly(&asm, &ctx, bytes, text),
        Check::Asm(text, bytes) => check_assembly(&asm, &ctx, text, bytes),
        Check::BadBytes(bytes) => check_bad_bytes(&asm, &ctx, bytes),
        Check::BadAsm(text) => check_bad_asm(&asm, &ctx, text),
    }
}

/// Runs the test `name` in a child process, so a crash, a check running longer than `timeout` or
/// one using more than `memory_mb` of address space fails only this test
fn run_in_child(
    ghidra: &Path,
    name: &str,
    timeout: Duration,
    memory_mb: u64,
) -> Result<(), String> {
    // std can't set rlimits, so let the shell do it before running the test binary
    let mut child = Command::new("sh")
        .args(["-c", r#"ulimit -v "$GHIDRA_CORPUS_KB" && exec "$0" "$@""#])
        .arg(std::env::current_exe().unwrap())
        .args([name, "--exact", "--ignored", "--nocapture"])
        .env("GHIDRA_CORPUS_KB", (memory_mb * 1024).to_string())
        .env(CHILD_ENV, "1")
        // Parsing real specs recurses deeper than the default test thread stack allows
        .env("RUST_MIN_STACK", (256 << 20).to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("could not run child process: {}", err))?;
    let mut stderr = child.stderr.take().unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        let _ = tx.send(s);
    });

    let stderr = rx.recv_timeout(timeout);
    let _ = child.kill();
    let status = child.wait().map_err(|err| err.to_string())?;
    let stderr = stderr.map_err(|_| format!("TIMEOUT after {:?}", timeout))?;
    if status.success() {
        return Ok(());
    }
    // The child's panic message, or how it crashed
    let reason = stderr
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with("note: ") && !l.starts_with("error: test"))
        .collect::<Vec<_>>()
        .join("\n")
        .replace(&format!("{}/", ghidra.display()), "");
    Err(format!("{} ({})", reason, status))
}

const CHILD_ENV: &str = "GHIDRA_CORPUS_CHILD";

/// Runs `check` of `spec` for the test at `path` (its `module_path!()` and name)
fn run(spec: &'static Spec, path: &str, check: Check) {
    let Some(ghidra) = std::env::var_os("GHIDRA_SRC").map(PathBuf::from) else {
        eprintln!("GHIDRA_SRC not set, skipping");
        return;
    };
    let result = if std::env::var_os(CHILD_ENV).is_some() {
        run_check(&ghidra, spec, check)
    } else {
        // The test's name is its path without the crate
        let name = path.split_once("::").unwrap().1;
        let timeout = std::env::var("GHIDRA_TIMEOUT").map_or(60, |s| s.parse().unwrap());
        let memory_mb = std::env::var("GHIDRA_MEMORY_MB").map_or(1024, |s| s.parse().unwrap());
        run_in_child(&ghidra, name, Duration::from_secs(timeout), memory_mb)
    };
    if let Err(err) = result {
        panic!("{}", err);
    }
}
