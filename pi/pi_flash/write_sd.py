import ctypes
import sys
from ctypes import wintypes

IMG = r"C:\Users\Jan\pi_flash\raspios-trixie-arm64-lite.img"
DRV = r"\\.\PhysicalDrive1"


class DISK_GEOMETRY(ctypes.Structure):
    _fields_ = [("Cylinders", ctypes.c_longlong), ("MediaType", wintypes.DWORD),
                ("TracksPerCylinder", wintypes.DWORD), ("SectorsPerTrack", wintypes.DWORD),
                ("BytesPerSector", wintypes.DWORD)]


class DISK_GEOMETRY_EX(ctypes.Structure):
    _fields_ = [("Geometry", DISK_GEOMETRY), ("DiskSize", ctypes.c_ulonglong),
                ("Data", ctypes.c_byte * 1)]


# identity guard: the target physical drive must be the ~59.48 GB SD card
handle = ctypes.windll.kernel32.CreateFileW(
    DRV, 0, 0x3, None, 3, 0x2000000, None)  # OPEN_EXISTING, FILE_SHARE_RW
if handle == -1:
    print("FAIL: cannot open physical drive")
    sys.exit(1)
buf = DISK_GEOMETRY_EX()
ret = ctypes.c_ulong(0)
ok = ctypes.windll.kernel32.DeviceIoControl(
    handle, 0x700A0, None, 0, ctypes.byref(buf), ctypes.sizeof(buf),
    ctypes.byref(ret), None)  # IOCTL_DISK_GET_DRIVE_GEOMETRY_EX
ctypes.windll.kernel32.CloseHandle(handle)
if not ok:
    print("FAIL: DeviceIoControl geometry query")
    sys.exit(1)
gb = buf.DiskSize / (1024 ** 3)
if not (50 < gb < 70):
    print(f"FAIL: drive size {gb:.2f} GB is not the SD card — aborting")
    sys.exit(1)
print(f"drive identity OK: {gb:.2f} GB")

img = open(IMG, "rb")
drv = open(DRV, "r+b", buffering=0)
n = 0
while True:
    b = img.read(64 * 1024 * 1024)
    if not b:
        break
    drv.write(b)
    n += len(b)
    print(f"{n // (1024 * 1024)} MB", flush=True)
drv.flush()
drv.close()
print(f"DONE {n // (1024 * 1024)} MB")

# verify: first 1 MiB round-trip
with open(IMG, "rb") as f:
    ref = f.read(1024 * 1024)
h = open(DRV, "rb", buffering=0)
got = h.read(1024 * 1024)
h.close()
print("VERIFY:", "PASS" if got == ref else "FAIL")
sys.exit(0 if got == ref else 1)
