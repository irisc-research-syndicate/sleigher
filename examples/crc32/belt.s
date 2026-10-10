// CRC-32 (reflected, polynomial 0xedb88320), one bit at a time
// in:  b0 = data, b1 = length    out: b0 = crc
// Both loops keep their state at the front of the belt and conform it back in place:
//   b0 = crc, b1 = bits left, b2 = polynomial, b3 = data pointer, b4 = bytes left
        conw 0xedb88320
        con 0x0
        con -0x1
        brz b4, done
byte:   ldb b3
        xor b0, b1              // crc ^ data byte
        addi b5, 0x1            // next data byte
        addi b7, -0x1           // one byte less
        con 0x8                 // bits left
        conform b3, b0, b7, b2, b1
bit:    andi b0, 0x1
        mul b0, b3              // polynomial or 0
        shri b2, 0x1
        xor b0, b1              // new crc
        addi b5, -0x1           // one bit less
        conform b1, b0, b7, b8, b9
        br b1, bit
        br b4, byte
done:   xori b0, -0x1
end:
