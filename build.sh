#!/bin/zsh
# build the neural boot sector
set -e
.venv/bin/python corpus_gen.py      # 3.2 MB of synthetic dmesg dreams
.venv/bin/python train.py           # train, quantize, emit params.inc + weights.bin
nasm -f bin boot.asm -o boot.bin    # 512-byte stage 1 (asserted below)
[ $(stat -f%z boot.bin) -le 512 ] || { echo "BOOT SECTOR TOO BIG"; exit 1; }
cat boot.bin weights.bin > disk.img
echo "disk.img ready: $(stat -f%z disk.img) bytes (boot sector + int8 weights)"
