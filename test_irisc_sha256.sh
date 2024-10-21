#!/bin/sh
clear;
export RUST_LOG=${RUST_LOG:-info}
cargo run --bin sleigh-emu -- \
    --slaspec iRISC.slaspec \
    --map 0x00710000:00007000_0001c494_IRON_PREP_CODE \
    --ram 0xffff0000:0x10000 \
    --entrypoint 0x007230cc \
    -r r1=0xfffff000 -r r4=0x00710000 -r r5=0x37 -r r6=0xffffff80 \
    -b 0x007230cc:sha256:r1l,r4l,r5l,r6l \
    -b 0x00722d8c:sha256_transform:r1l,r4l \
    -b 0x00722cf8:sha256_update:r1l,r4l,r5l,r6l \
    -b 0x00722fb0:sha256_finalize:r1l,r4l,r5l \
    -b 0x00722e7c:sha256_round:r21l,r3l,r26l,r23l,r20l,r18l,r19l,r22l