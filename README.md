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

Delay slots follow Ghidra: an instruction's p-code includes that of the instructions in its delay slot at its `delayslot`, and `inst_next` in its p-code is the address after them.


Future work
===========
- Instruction lifter to a simpler IR than raw sleigh execution semantics
- Symbolic execution engine
- JIT emulator for faster execution of lifted instructions
- Better testing, more toy examples.
- Debugger?