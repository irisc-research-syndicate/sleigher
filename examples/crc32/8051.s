// CRC-32 (reflected, polynomial 0xedb88320), one bit at a time
// in:  DPTR = data (in CODE), R2 = length    out: R4R5R6R7 = crc, R4 lowest
        mov SP, #0x2f
        mov R4, #0xff
        mov R5, #0xff
        mov 0x6, #0xff      // R6 by its direct address
        mov R7, #0xff
        mov A, R2
        jz done
byte:   clr A
        movc A, @A+DPTR
        inc DPTR
        xrl 0x4, A          // R4 ^= data
        mov R3, #0x8
bit:    acall shift
        jnc skip
        xrl 0x7, #0xed
        xrl 0x6, #0xb8
        mov A, R5
        xrl A, #0x83
        mov R5, A
        mov A, #0x20
        xrl A, R4
        mov R4, A
skip:   djnz R3, bit
        djnz R2, byte
done:   mov A, R4
        cpl A
        mov R4, A
        mov A, R5
        cpl A
        mov R5, A
        mov A, R6
        cpl A
        mov R6, A
        mov A, R7
        cpl A
        mov R7, A
        ljmp end

// Shift R7..R4 right by one, the bit shifted out ends up in C
shift:  clr C
        mov A, R7
        rrc A
        mov R7, A
        mov A, R6
        rrc A
        mov R6, A
        mov A, R5
        rrc A
        mov R5, A
        mov A, R4
        rrc A
        mov R4, A
        ret
end:
