#!/usr/bin/env python3
"""Generate a realistic Linux dmesg-style corpus for bootgpt.

Real boot logs repeat themselves: one kernel version, one disk, the same
usb ports, monotonic timestamps. Keeping that structure makes the tiny
model's dreams look far more coherent. Charset is restricted to 62+ chars.
"""
import random

random.seed(1337)

ALLOWED = (" abcdefghijklmnopqrstuvwxyz"
           "0123456789"
           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "[].:-_/=+,")
ALLOWED_SET = set(ALLOWED)


def t(spec):
    return "[{:>10}] ".format(spec)


class Boot:
    """Per-boot constants: the same machine dreams the same devices."""

    def __init__(self):
        self.kern = "%d.%d.%d-%d-generic" % (random.choice([5, 6]),
                                             random.randint(4, 19),
                                             random.randint(0, 30),
                                             random.randint(1, 25))
        self.gcc = "%d.%d.%d" % (random.choice([11, 12, 13]),
                                 random.randint(0, 4), 0)
        self.disk = random.choice(["sda", "sda", "sda", "nvme0n1"])
        self.usb_port = random.randint(1, 3)
        self.usb_dev = random.randint(2, 9)
        self.bus = random.randint(0, 2)
        self.ata = random.randint(1, 6)
        self.iface = random.choice(["eth0", "enp3s0"])
        self.mem_total = random.randint(16000000, 17000000)
        self.cpus = random.choice([4, 8, 8, 12, 16])
        self.root = random.choice(["root=/dev/sda2", "root=/dev/sda2",
                                   "root=/dev/nvme0n1p2"])

    def slot(self, i):
        return "%02x:%02x.%d" % (self.bus, random.randint(0, 31),
                                 random.randint(0, 7))


def hexaddr():
    return "0x%08x" % random.getrandbits(32)


def line_linux_version(b, ts):
    return t(ts) + "Linux version %s (gcc version %s, GNU ld ...) bootgpt" % (
        b.kern, b.gcc)


def line_command_line(b, ts):
    opts = ["BOOT_IMAGE=/vmlinuz-%s" % b.kern, b.root, "quiet", "splash", "rw"]
    return t(ts) + "Command line: " + " ".join(opts[:random.randint(3, 5)])


def line_ram_map(b, ts):
    out = [t(ts) + "BIOS-e820: [mem %#010x-%#010x] %s" % (
        random.getrandbits(32), random.getrandbits(32),
        random.choice(["usable", "reserved", "ACPI data"]))
        for _ in range(random.randint(2, 4))]
    return "\n".join(out)


def line_x86(b, ts):
    avail = b.mem_total - random.randint(2000000, 4000000)
    picks = [
        "Memory: %dK/%dK available (%dK kernel code, %dK rwdata)" % (
            avail, b.mem_total, random.randint(8000, 20000),
            random.randint(2000, 6000)),
        "x86/fpu: Supporting XSAVE feature 0x%02x: '%s'" % (
            random.getrandbits(8),
            random.choice(["x87 floating point", "SSE", "AVX"])),
        "x86: Booted up %d CPUs in %d ms" % (b.cpus, random.randint(50, 900)),
    ]
    return t(ts) + random.choice(picks)


def line_smp(b, ts):
    return t(ts) + "smpboot: CPU%d is now online" % random.randrange(b.cpus)


def line_cpu(b, ts):
    picks = [
        "microcode: updated early: 0x%x -> 0x%x" % (
            random.getrandbits(16), random.getrandbits(24)),
        "mce: CPU%d: Machine check events logged" % random.randrange(b.cpus),
    ]
    return t(ts) + random.choice(picks)


def line_acpi(b, ts):
    picks = [
        "ACPI: Early table checksum verification disabled",
        "ACPI: Interpreter enabled",
        "ACPI: Using IOAPIC for irq balancing",
        "ACPI: Power Button [PWRF]",
        "ACPI: PCI Interrupt Link [LNKA] enabled at IRQ %d" % random.randint(9, 22),
        "ACPI: Added _OSI(Module Device)",
    ]
    return t(ts) + random.choice(picks)


def line_pci(b, ts):
    vendors = ["8086:1237", "10de:0a65", "1022:1483", "8086:29c0", "1b36:000c"]
    slot = b.slot(0)
    picks = [
        "pci 0000:%s: [%s] type %02x class 0x%06x" % (
            slot, random.choice(vendors), random.getrandbits(8),
            random.getrandbits(24)),
        "pci 0000:%s: reg 0x10: [mem %#010x-%#010x 64bit]" % (
            slot, random.getrandbits(32), random.getrandbits(32)),
        "pci 0000:%s: BAR 0: assigned [mem %#010x 64bit]" % (
            slot, random.getrandbits(32)),
    ]
    return t(ts) + random.choice(picks)


def line_usb(b, ts):
    picks = [
        "usb %d-%d: new %s-speed USB device number %d using xhci_hcd" % (
            b.bus + 1, b.usb_port,
            random.choice(["high", "full"]),
            b.usb_dev + random.randrange(3)),
        "usb %d-%d: New USB device found, idVendor=%04x, idProduct=%04x" % (
            b.bus + 1, b.usb_port,
            random.getrandbits(16), random.getrandbits(16)),
        "usb-storage %d-%d:1.0: USB Mass Storage device detected" % (
            b.bus + 1, b.usb_port),
        "usbcore: registered new interface driver usbfs",
    ]
    return t(ts) + random.choice(picks)


def line_block(b, ts):
    picks = [
        "sd %d:0:0:0: [%s] 512-byte logical blocks: (%d GB)" % (
            random.randint(0, 3), b.disk,
            random.randint(100, 2000)),
        "sd %d:0:0:0: [%s] Write Protect is off" % (random.randint(0, 3), b.disk),
        "%s: %s1 %s2 %s3" % (b.disk, b.disk, b.disk, b.disk),
        "ata%d: SATA link up %s (SStatus %d)" % (
            b.ata, random.choice(["3.0 Gbps", "6.0 Gbps"]),
            random.getrandbits(8)),
        "EXT4-fs (%s2): mounted filesystem with ordered data mode" % b.disk,
        "random: crng init done",
        "clocksource: tsc: mask: %#010x max_cycles: %#010x" % (
            random.getrandbits(32), random.getrandbits(32)),
    ]
    return t(ts) + random.choice(picks)


def line_net(b, ts):
    picks = [
        "e1000: %s: IRQ %d" % (b.iface, random.randint(10, 40)),
        "NET: Registered protocol family %d" % random.choice([2, 16, 26]),
    ]
    return t(ts) + random.choice(picks)


def line_misc(b, ts):
    picks = [
        "clocksource: Switched to clocksource %s" % random.choice(
            ["tsc", "hpet", "refined-tsc"]),
        "Freeing unused kernel image memory: %dK" % random.randint(1000, 4000),
        "Write protecting the kernel read-only data: %dK" % (
            random.randint(1000, 4000)),
        "i915: Found %s VGA controller at %s" % ("0x1237", hexaddr()),
    ]
    return t(ts) + random.choice(picks)


BOOT_SEQ = [
    (line_linux_version, 1), (line_command_line, 1),
    (line_ram_map, 2), (line_x86, 2), (line_smp, 2),
    (line_cpu, 2), (line_acpi, 4), (line_pci, 4),
    (line_usb, 3), (line_block, 4),
    (line_net, 1), (line_misc, 2),
]


def sanitize(s):
    out = []
    for ch in s:
        if ch in ALLOWED_SET or ch == "\n":
            out.append(ch)
        elif ch == "\t":
            out.append(" ")
        else:
            out.append("")
    return "".join(out)


def main():
    all_lines = []
    for i in range(1600):
        b = Boot()
        ts = 0.0
        for fn, cnt in BOOT_SEQ:
            for _ in range(cnt):
                ts += random.random() * 0.25
                all_lines.append(fn(b, "%.6f" % ts))
    text = "\n".join(sanitize(l) for l in all_lines) + "\n"
    bad = set(text) - ALLOWED_SET - {"\n"}
    assert not bad, bad
    open("corpus.txt", "w").write(text)
    print("corpus.txt:", len(text), "bytes,", len(all_lines), "lines")
    print("\n".join(text.splitlines()[:6]))


if __name__ == "__main__":
    main()
