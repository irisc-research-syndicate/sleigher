// CRC-32 (reflected, polynomial 0xedb88320), one bit at a time
// in:  r1 = data, r2 = length    out: r0 = crc
        mov r0, #0xffffffff
        mov r6, #0xedb88320
        cmp r2, #0x0
        jz done
byte:   ldb r3, [r1]
        xor r0, r3
        add r1, #0x1
        mov r4, #0x8
bit:    mov r5, r0
        shr r0, #0x1
        and r5, #0x1        // ZF = low bit was clear
        jz skip
        xor r0, r6
skip:   sub r4, #0x1
        jnz bit
        sub r2, #0x1
        jnz byte
done:   xor r0, #0xffffffff
end:
