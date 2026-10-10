// CRC-32 (reflected, polynomial 0xedb88320), one bit at a time
// in:  r1 = data, r2 = length    out: r0 = crc
// Small immediates are ambiguous in this ISA, so constants are built in registers.
        xor r9, r9, r9
        sub r9, r9, -0x1        // r9 = 1
        xor r7, r7, r7
        add r7, r7, -0x1        // r7 = 0xffffffff
        or r0, r7, r7           // crc
        xor r6, r6, r6
        xor r6, r6, 0xedb80000
        or r6, r6, 0x8320       // r6 = polynomial
        add r8, r9, r9
        add r8, r8, r8
        add r8, r8, r8          // r8 = 8
        bz r2, done
byte:   lbu r3, r1
        xor r0, r0, r3
        add r1, r1, r9
        or r4, r8, r8           // bits left
bit:    and r5, r0, r9
        shr r0, r0, r9
        bz r5, skip
        xor r0, r0, r6
skip:   sub r4, r4, r9
        bnz r4, bit
        sub r2, r2, r9
        bnz r2, byte
done:   xor r0, r0, r7
end:
