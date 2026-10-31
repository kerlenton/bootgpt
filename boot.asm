; bootgpt: a neural network that boots your PC before any OS does.
; Stage 1 (this, <=510 bytes): loads int8 weights from sectors 2+ and
; runs char-level LM inference in 16-bit real mode on a 386+.
; Weights (FS=0x0800 segment, untouched by BIOS):
;   +0    vocab[64]          ASCII, '\n'=10
;   +64   emb[64*6]          int8, [v*6+r]
;   +448  W1[48*36]          int8, [j*36+i], i = k*6+r
;   +2176 W2[64*48]          int8, [v*48+h]
;   +5248 b1[48]             int16 Q12
;   +5344 b2[64]             int16 Q8
;   +5472 strings (hdr, byemsg), emitted by train.py
; Runtime (DS=0): ctx 0x7A00[6], y 0x7A10[48], logits 0x7A60[128],
;                 xin 0x7B00[36], rnd 0x7B40, bootdrv 0x7B42

[org 0x7C00]
[cpu 386]
[bits 16]
%include "params.inc"

EMB_OFF     equ 64          ; vocab[64] first, rest from params.inc

CTXBUF      equ 0x7A00
YBUF        equ 0x7A10
LOGBUF      equ 0x7A80
XINBUF      equ 0x7B00
RND         equ 0x7B40
BOOTDRV     equ 0x7B42
DBGCTR      equ 0x7B44

start:
        cli
        xor ax, ax
        mov ds, ax
        mov ss, ax
        mov sp, 0x7000
        sti
        mov [BOOTDRV], dl
        mov ax, 0x0800
        mov fs, ax
        mov es, ax

; ---- load SECTORS sectors (weights) from CHS 0/0/2 to 0x0800:0000 ----
.load:
        mov ax, 0x0200 | SECTORS
        mov cx, 0x0002          ; C=0, S=2
        xor dx, dx              ; H=0
        mov dl, [BOOTDRV]
        xor bx, bx
        int 0x13
        jc .load

%ifndef DBGPORT
        mov si, HDRADDR
        call puts
%endif

; ---- seed the LCG from PIT ticks (SEEDFIX for reproducible debug runs) ----
%ifdef SEEDFIX
        mov dx, SEEDFIX
%else
        xor ah, ah
        int 0x1A
%ifndef DBGPORT
        or dx, 1
%endif
%endif
        mov [RND], dx

; ---- ctx = seed indices (data at the bottom of this sector) ----
        mov si, seedctx
        xor bx, bx
.ctxin:
        mov al, [si+bx]
        mov [CTXBUF+bx], al
        inc bx
        cmp bx, CTX
        jb .ctxin

mainloop:
        call infer              ; -> bl = next char index
%ifdef DBGSTOP
        inc byte [DBGCTR]
        cmp byte [DBGCTR], DBGSTOP
        jb .go
        jmp $                   ; park: buffers hold last inference state
.go:
%endif
        push bx

        mov al, [fs:bx]         ; vocab[idx]
        cmp al, 10
        jne .noCR
        mov al, 13
        call putc
        mov al, 10
.noCR:
        call putc
        pop bx

        ; shift ctx, append prediction
        mov si, CTXBUF
        mov cx, CTX-1
.sh:
        mov al, [si+1]
        mov [si], al
        inc si
        loop .sh
        mov [si], bl

%ifndef DBGPORT
        mov ah, 1               ; key pressed?
        int 0x16
        jnz keyexit

        mov ah, 0x86            ; wait ~10 ms
        xor cx, cx
        mov dx, 10000
        int 0x15
%endif
        jmp mainloop

keyexit:
        xor ah, ah              ; consume the keypress so the buffer
        int 0x16                ; is empty after the reboot
        int 0x19                ; reboot

; ---------------- inference (mirrors BootSim in train.py) ----------------
infer:
        ; xin[i] = emb[EMB_OFF + ctx[k]*6 + r], i = k*6+r
        xor ebx, ebx
        xor ebp, ebp
.kl:
        movzx eax, byte [CTXBUF+ebp]
        mov ecx, eax
        shl eax, 1
        shl ecx, 2
        add eax, ecx            ; ctx[k]*6
        add eax, EMB_OFF
        mov esi, eax
        mov ecx, EMBDIM
.rl:
        mov al, [fs:esi]
        mov [XINBUF+ebx], al
        inc esi
        inc ebx
        loop .rl
        inc ebp
        cmp ebp, CTX
        jb .kl

        ; hidden: y[j] = trunc(p*64 / (4096+|p|))   (always < 64, no clip)
        ;          p = (Σ xin*W1[j]) [<</>]A1SHIFT + b1[j]   (Q12)
        mov edi, W1_OFF
        xor ebp, ebp
.jl:
        xor esi, esi
        xor ebx, ebx
.mac1:
        movsx eax, byte [XINBUF+ebx]
        movsx ecx, byte [fs:edi+ebx]
        imul eax, ecx
        add esi, eax
        inc ebx
        cmp ebx, CTX*EMBDIM
        jb .mac1
%if A1SHIFT >= 0
        sar esi, A1SHIFT
%else
        shl esi, -A1SHIFT
%endif
        movsx eax, word [fs:ebp*2+B1_OFF]
        add esi, eax            ; esi = p
        mov ebx, esi
        test ebx, ebx
        jns .p1
        neg ebx
.p1:
        add ebx, 4096           ; ebx = 4096+|p|
        mov eax, esi
        shl eax, 6
        cdq
        idiv ebx                ; eax = trunc(p*64 / (4096+|p|))
        mov [YBUF+ebp], al
%ifdef DBGPORT
        mov eax, esi
        out 0x80, al
        shr eax, 8
        out 0x80, al
        shr eax, 8
        out 0x80, al
        shr eax, 8
        out 0x80, al
%endif
        add edi, CTX*EMBDIM
        inc ebp
        cmp ebp, HID
        jb .jl

        ; logits: L[v] = (Σ y*W2[v]) [<</>]A2SHIFT + b2[v]  (Q8, int16)
        mov edi, W2_OFF
        xor ebp, ebp
.vl:
        xor esi, esi
        xor ebx, ebx
.mac2:
        movsx eax, byte [YBUF+ebx]
        movsx ecx, byte [fs:edi+ebx]
        imul eax, ecx
        add esi, eax
        inc ebx
        cmp ebx, HID
        jb .mac2
%if A2SHIFT >= 0
        sar esi, A2SHIFT
%else
        shl esi, -A2SHIFT
%endif
        movsx eax, word [fs:ebp*2+B2_OFF]
        add esi, eax
        mov [LOGBUF+ebp*2], si
        add edi, HID
        inc ebp
        cmp ebp, VOCAB
        jb .vl

        ; sample: argmax( L[v] + int8(rnd>>8)*NOISE ), first index wins ties
        xor edi, edi
        not edi
        xor ebx, ebx
        xor ebp, ebp
.sl:
        mov ax, [RND]
        imul ax, ax, 25173
        add ax, 13849
        mov [RND], ax
        mov al, ah
        cbw
        imul ax, ax, NOISE
        movsx edx, ax
        movsx ecx, word [LOGBUF+ebp*2]
        add ecx, edx
        cmp ecx, edi
        jle .nx
        mov edi, ecx
        mov ebx, ebp
.nx:
        inc ebp
        cmp ebp, VOCAB
        jb .sl
        ret                     ; bl = argmax index

; ---------------- bios helpers ----------------
putc:                           ; AL = char (clobbers AX/BX, FS untouched)
        mov ah, 0x0E
        xor bx, bx
        int 0x10
        ret

puts:                           ; DS:SI -> asciiz
        lodsb
        or al, al
        jz .d
        call putc
        jmp puts
.d:
        ret

seedctx: SEEDBYTES              ; vocab indices (train.py)

        times 510-($-$$) db 0
        dw 0xAA55
