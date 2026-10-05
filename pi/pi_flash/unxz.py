import lzma

SRC = r"C:\Users\Jan\pi_flash\2026-09-15-raspios-trixie-arm64-lite.img.xz"
DST = r"C:\Users\Jan\pi_flash\raspios-trixie-arm64-lite.img"

s = lzma.open(SRC, "rb")
d = open(DST, "wb")
n = 0
while True:
    b = s.read(64 * 1024 * 1024)
    if not b:
        break
    d.write(b)
    n += len(b)
    print(f"{n // (1024*1024)} MB", flush=True)
d.close()
print(f"DONE {n // (1024*1024)} MB")
