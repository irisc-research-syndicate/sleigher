Sleigher: Toolbox for working with Sleigh from Ghidra
======================================================
Sleigh is a DSL used in the reverse engineering tool Ghidra,
it is a language that discribes an ISA(such as x86, ARM, ...),
is is premarily used for disassembling and lifting code for use in Ghidra's decompiler.

This project aims to liberate the language from Ghidra such that it can be used programmatically in other rust programs,
this enables rapid prototyping of Sleigh code for new and unknown architectures, and well as enabling emulation, disassembling and assembling without the use of the whole framework of Ghidra.


Overview of the project
=======================
This project consists of 4 primary parts: A library `sleigher`, and 3 tools `sleigh-disasm`, `sleigh-asm`, `sleigh-emu` using that library.

The 3 tools should be considered example code for using the library as well as being simple but moderately powerful tools for working with Sleigh.

Context variables are supported, and each tool takes initial values with `--context name=value`.
Unlike Ghidra, which keeps the context by address, the context flows from instruction to instruction in the order they are visited: execution order in the emulator, line order in the assembler and a linear sweep in the disassembler.
In the emulator this matches Ghidra's, which carries the context along execution too.

`sleigh-disasm --flow` follows control flow from the start address instead, keeping the context by address as Ghidra's disassembler does: an address is decoded with the context that flowed to it first plus what `globalset`s committed to it, so a `globalset` to a branch destination is seen even where a linear sweep would have passed it already.
Each address is decoded once, so a commit to an address decoded earlier is not seen there, nor further along its flow.
`sleigh-asm --all-commits` likewise assembles each line with what any line commits to it, not only the lines before it, while the context still flows on in line order.

Delay slots follow Ghidra: an instruction's p-code includes that of the instructions in its delay slot at its `delayslot`, and `inst_next` in its p-code is the address after them.


Testing against Ghidra's slaspecs
=================================
`tests/ghidra.rs` checks a small corpus of known-good and known-bad instructions against the real slaspecs for
x86, x86-64, ARM, AArch64, MIPS, RISC-V and PowerPC. It needs a Ghidra source tree and is ignored by default:

```
git clone --depth 1 --filter=blob:none --sparse https://github.com/NationalSecurityAgency/ghidra.git
git -C ghidra sparse-checkout set --no-cone '/Ghidra/Processors/*/data/languages/'
GHIDRA_SRC=$PWD/ghidra cargo test --release --test ghidra -- --ignored
```

Every check is its own test, e.g. `x86_64::ret::asm` or `riscv_lp64d::bad_bytes_ffffffff`, so the usual test
filters select specs or checks (`-- --ignored riscv`). Each test runs in its own process, so a crash, a check running
longer than `GHIDRA_TIMEOUT` seconds (default 60) or using more than `GHIDRA_MEMORY_MB` of address space (default 1024)
fails only that test.


Future work
===========
- Instruction lifter to a simpler IR than raw sleigh execution semantics
- Symbolic execution engine
- JIT emulator for faster execution of lifted instructions
- Better testing, more toy examples.
- Debugger?