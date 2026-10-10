// CRC-32 (reflected, polynomial 0xedb88320), one bit at a time
// in:  r1 = data, r2 = length    out: r3 = crc
// The bit loop is branch free: crc = (crc >> 1) ^ (poly & -(crc & 1))
        { and r0, r0, 0x0 }
        { or r3, r0, -0x1 }
        { or r6, r0, -0x12477ce0 }              // polynomial
        { bz r2, done }
byte:   { ldb r4, r1 }
        { xor r3, r4 ; add r1, r1, 0x1 }
        { or r5, r0, 0x8 }                      // bits left
bit:    { and r4, r3, 0x1 }
        { neg r4, r4 ; shr1 r3, r3 }
        { and r4, r4, -0x12477ce0 }
        { xor r3, r4 ; sub r5, r5, 0x1 }
        { bnz r5, bit }
        { sub r2, r2, 0x1 }
        { bnz r2, byte }
done:   { xor r3, r3, -0x1 }
end:
