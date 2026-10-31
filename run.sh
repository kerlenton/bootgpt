#!/bin/zsh
# boot the dream machine in a QEMU window
# (screen-record this window for the tweet video; any key reboots)
qemu-system-i386 -drive format=raw,file=disk.img,if=ide -boot c
