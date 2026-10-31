#!/usr/bin/env python3
"""Automated demo recording for bootgpt (screendump pipeline).

Drives headless QEMU through the monitor socket: captures VGA
screendumps at 15 fps, presses a key at KEY_AT so the machine reboots
in frame, then assembles mp4 (2x crisp) + looping gif.

Timeline (~34s): boot -> neural dream -> keypress -> reboot -> dream again.
"""
import os
import shutil
import socket
import subprocess
import time

TAKE = 20          # total seconds
KEY_AT = 15        # keypress moment (reboot on camera)
FPS = 15
FRAMES = "/tmp/bootgpt_frames"


def main():
    shutil.rmtree(FRAMES, ignore_errors=True)
    os.makedirs(FRAMES)

    qemu = subprocess.Popen(
        ["qemu-system-i386",
         "-drive", "format=raw,file=disk.img,if=ide", "-boot", "c",
         "-monitor", "unix:rec.sock,server,nowait",
         "-display", "none", "-serial", "none"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)

    mon = socket.socket(socket.AF_UNIX)
    mon.connect("rec.sock")
    time.sleep(0.3)
    mon.settimeout(1.0)
    try:
        mon.recv(65536)
    except OSError:
        pass

    def cmd(c, wait_reply=True):
        mon.send(c.encode() + b"\n")
        if wait_reply:
            while True:
                d = mon.recv(65536)
                if b"(qemu)" in d:
                    return

    print(f"recording {TAKE}s @ {FPS}fps (screendump pipeline)")
    t0 = time.time()
    n = sent = 0
    tick = 1.0 / FPS
    next_t = t0
    while True:
        now = time.time()
        if now - t0 >= TAKE:
            break
        if not sent and now - t0 >= KEY_AT:
            cmd("sendkey spc")
            print("  [key pressed -> machine reboots in-frame]")
            sent = 1
        cmd(f"screendump {FRAMES}/f{n:05d}.ppm")
        n += 1
        next_t += tick
        d = next_t - time.time()
        if d > 0:
            time.sleep(d)

    time.sleep(0.5)                      # let the last dumps flush
    mon.send(b"quit\n")
    qemu.terminate()
    frames = len(os.listdir(FRAMES))
    print(f"{frames} frames captured")

    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", str(FPS), "-i", f"{FRAMES}/f%05d.ppm",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "bootgpt_raw.mp4"], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", "bootgpt_raw.mp4",
                    "-vf", "scale=1440:800:flags=neighbor",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "bootgpt_demo.mp4"], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", "bootgpt_raw.mp4",
                    "-vf", ("fps=12,scale=720:400:flags=neighbor,"
                            "split[a][b];[a]palettegen[p];"
                            "[b][p]paletteuse=dither=bayer:bayer_scale=3"),
                    "-loop", "0", "bootgpt_demo.gif"], check=True)
    for f in ("bootgpt_raw.mp4", "bootgpt_demo.mp4", "bootgpt_demo.gif"):
        print(f, f"{os.path.getsize(f)/1024:.0f} KB")
    shutil.rmtree(FRAMES, ignore_errors=True)
    print("done: bootgpt_demo.mp4 / bootgpt_demo.gif ready")


if __name__ == "__main__":
    main()
