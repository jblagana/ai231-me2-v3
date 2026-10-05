# Flash kit — AI231 ME2 V2 (Pi 5, RPi OS Trixie lite)

Working reader needed. The original reader on Jan's laptop (GENERIC STORAGE
DEVICE) died mid-task: it enumerates a 59.48 GB disk but refuses ALL I/O
(verified: elevated raw opens, diskpart, even after a full PC reboot —
identification works, data path dead). Card itself is probably fine.

## What's in this folder
- `raspios-trixie-arm64-lite.img` — 2.85 GB extracted image (any card >= 8 GB
  works; 64 GB recommended).
- `sha256.txt` — verify on the machine that downloaded it (skip if copied from
  a verified machine).
- `write_sd.py` — Windows writer with drive-identity guard.

## Windows (any PC with a working reader)
    python write_sd.py
Guard: only writes to a disk whose FriendlyName contains STORAGE/USB/SD and
whose size is 50-70 GB. Adjust the size range in the script if using a
different card size (e.g. 8-10 GB card → change to 7-11).

## Linux / macOS
    # find the device: /dev/sdX (Linux) or /dev/disk2 (macOS) — check with
    # lsblk -o NAME,SIZE,MODEL / diskutil list FIRST
    sudo dd if=raspios-trixie-arm64-lite.img of=/dev/sdX bs=4M conv=fsync status=progress
    sync

## Android phone (no PC at all)
1. OTG adapter + a spare microSD card (>= 8 GB) in it.
2. Copy this .img to the phone (OneDrive app).
3. Install "USB Drive Dumper" (F-Droid or Play Store).
4. Write the .img to the card, safe-remove, put card in the Pi.

## After flashing (on whatever machine has the card) — bake WiFi
The card boots with partition `sda1` (FAT, "BOOT") + `sda2` (ext4, "ROOT").
Mount the FAT partition and drop in `/boot/firmware/user-data`:

    cloud-init:
      datasource_list: [ None ]
    hostname: me2pi
    ssh:
      install: openssh-server
      enable: true
    network:
      version: 2
      ethernets:
        eth0:
          optional: true
          access-points:
            hotspot_jan:
              password: "abcdefghij"

    # Linux (mount the FAT partition; on trixie the FAT root == /boot/firmware,
    # so user-data goes at the FAT ROOT):
    sudo mkdir -p /mnt/flash && sudo mount /dev/sdX1 /mnt/flash
    sudo cp user-data /mnt/flash/user-data && sudo umount /mnt/flash && sync
    # Windows: after the write the FAT partition gets a letter (e.g. D:,
    # label RPI-BOOT/BOOT) — write the SAME YAML to  D:\user-data  (FAT root,
    # NOT D:\firmware\user-data). Then dismount + unplug.
    # Insert into Pi 5, power on, first boot ~2-3 min.

    # NOTE: the YAML above must create the user headlessly (raw image has no
    # default user) — add:
    # users:
    #   - name: jan
    #     groups: [sudo, audio]
    #     shell: /bin/bash
    #     sudo: ['ALL=(ALL) NOPASSWD:ALL']
    # chpasswd: { list: ['jan:abcdefghij'], expire: false }
    # (password = same as WiFi; this is a lab device)

Then: ssh jan@me2pi.local (or the IP from the router), copy
`pi/bootstrap.sh` from the repo, run it, re-login (audio group), test
`python3 pi_demo.py --once` with the USB mic.