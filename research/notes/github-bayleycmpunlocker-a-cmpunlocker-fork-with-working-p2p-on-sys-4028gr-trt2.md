---
title: 'GitHub - bayley/cmpunlocker: A cmpunlocker fork with working p2p on SYS-4028GR-TRT2
  · GitHub'
id: github-bayleycmpunlocker-a-cmpunlocker-fork-with-working-p2p-on-sys-4028gr-trt2
tags:
- cmp170hx-training-feasibility-a0d4fc
- hardware
- cmp170hx
- peer-to-peer
- pcie-link
- unlock
- measured
- primary-source
created: '2026-10-27T10:51:32.853821Z'
updated: '2026-10-27T19:00:31.331724Z'
source: https://github.com/bayley/cmpunlocker
source_domain: github.com
fetched_at: '2026-10-27T10:51:32.230509Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'bayley/cmpunlocker (56 commits, latest 5a7bb4b 2026-08-08, 26 stars) — a
  fork of the CMP 170HX unlock tool that gets GPU-to-GPU P2P working, and the single
  most load-bearing source for the query''s ''no P2P'' premise. Directly falsifies
  the repo''s claim that P2P is impossible: P2P works over BAR1 P2P (not the mailbox),
  measured 1.68 GB/s each way on PCIe Gen2 x4 with a full 4-GPU mesh (12/12 directed
  pairs), verified with real transfers rather than cudaDeviceCanAccessPeer(). The
  mailbox path (P2P_CONNECTIVITY_PCIE_PROPRIETARY) is a documented dead end — driver
  configures it correctly, both GSP RPCs return NV_OK, transfers move exactly zero
  bytes with no error and no Xid. BAR1 P2P sidesteps it by rewriting peer PTEs from
  GMMU_APERTURE_PEER to SYS_COH/SYS_NONCOH pointing at the peer''s BAR1 bus address.
  Requires together: patches 0011 (BAR1 P2P HALs forced on + peer-PTE rewrite), 0013
  (skip mailbox peer pre-registration), 0015 (restore BAR1-P2P read cap), kernel-patches
  for 64 GB BAR1 from a normal boot, and RMForceStaticBar1=1 / RMPcieP2PType=1 in
  modprobe.d plus update-initramfs. Scaling to eight GPUs across two PLX switches
  on one Xeon E5 v4 root complex: all 56 directed pairs pass the alias-proof test
  including all 32 cross-switch pairs, no driver change needed — even though NVIDIA''s
  own _p2pCapsGetHostSystemStatusOverPcieBar1 grants the read cap only for bCommonPciSwitchFound
  || RYZEN || XEON_SPR, so the gate is conservative rather than descriptive. Steady
  state: same-switch 18 pairs all 1.68 GB/s; cross-switch 24 pairs 1.07-1.45 GB/s
  (~78% of same-switch, asymmetric A->B 1.45 vs B->A 1.07-1.31, a root-complex property).
  ACS redirection must be disabled on every port of both switches — leaving switch
  A with ReqRedir+/CmpltRedir+ dropped same-switch from 1.68 to 1.20 GB/s (correct
  but slow, fails silently). Eight 64 GB BAR1s at 128 GB stride span 512 GB across
  two root ports, needing pci=hpmmioprefsize=2T plus kernel patch 0001. CRITICAL for
  the query''s link-width premise: ''All CMP 170HX report x4 (downgraded) from an
  x16 capability; that x4 is normal, x2 is not'' — one card trained at x2 and ran
  exactly half speed everywhere (0.84 GB/s same-switch, ~0.58 cross) with zero AER
  errors, i.e. physical (riser/seating) not signal integrity. Also documents two ways
  to hang the machine (ForceP2P=0x11 registry dword; live module reload) and an anti-rollback
  kernel-update rebuild system with pinned NVIDIA packages. Unlock scope: full SM
  compute, 64 GB on 8 GB cards / 40 GB on 10 GB cards, PCIe Gen2, BAR1 64 MB->64 GB,
  HBM2e NDIV overclock (N x 27 MHz; NDIV 70 = 1890 MHz works on ~60% of 8 GB cards,
  73 usually only on lucky ones).'
utility_score: 16.0
---

[Skip to content](https://github.com/bayley/cmpunlocker#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/bayley/cmpunlocker) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/bayley/cmpunlocker) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/bayley/cmpunlocker) to refresh your session. Dismiss alert
[**1** Branch](https://github.com/bayley/cmpunlocker/branches)[](https://github.com/bayley/cmpunlocker/tags)
Go to file
Open more actions menu
## Latest commit
[README: P2P across two PCIe switches, and the ACS gap it exposed](https://github.com/bayley/cmpunlocker/commit/5a7bb4b7e5056306fe49e8b824787659abb19914)
Open commit details
Aug 8, 2026
[5a7bb4b](https://github.com/bayley/cmpunlocker/commit/5a7bb4b7e5056306fe49e8b824787659abb19914) · Aug 8, 2026
## History
[56 Commits](https://github.com/bayley/cmpunlocker/commits/master/)
Open commit details
56 Commits
## Folders and files  
| Name  | Name  | Last commit message  | Last commit date  |  
| --- | --- | --- | --- |  
|   |   |   | Jul 29, 2026  |  
|   |   | [added compatibility with cuda11](https://github.com/bayley/cmpunlocker/commit/83f7199aab49cc6b5d5cd9d4463094f380e343ee "added compatibility with cuda11")  | Jul 31, 2026  |  
|   | Aug 1, 2026  |  
|   |   | [p2p: keep the debug instrumentation that found the BAR1 P2P gates](https://github.com/bayley/cmpunlocker/commit/47f6d678bada64f0b1f0f3645990b88da7565acb "p2p: keep the debug instrumentation that found the BAR1 P2P gates

Not applied by default -- build.sh globs driver/patches/*.patch, so these are
inert where they sit. Move one back into driver/patches/ to re-enable it.

  0010  prints the swallowed status of the two GSP mailbox setup RPCs, which are
        checked with NV_ASSERT\(\) and therefore silent in a release build, plus
        the selected connection type. Note the enum is 1=PCIE_PROPRIETARY
        \(mailbox\), 2=PCIE_BAR1 -- the header comment in the patch has it
        backwards.

  0012  dumps all four gates of kbusIsPcieBar1P2PMappingSupported_GH100 and tags
        which one rejects. This is what found the mailbox peer pre-registration.

  0014  logs the host-system read-cap decision, including bCommonPciSwitchFound
        and the CPU type. This is what found clFindCommonDownstreamBR\(\) failing
        to see the PLX switch.

Between them these turn "P2P silently does nothing" into a specific rejecting
line of code, which is most of the work.

Co-Authored-By: Claude Opus 5 \(1M context\) <noreply@anthropic.com>")  | Aug 8, 2026  |  
|   |   | [README: P2P across two PCIe switches, and the ACS gap it exposed](https://github.com/bayley/cmpunlocker/commit/5a7bb4b7e5056306fe49e8b824787659abb19914 "README: P2P across two PCIe switches, and the ACS gap it exposed

Scaled from four GPUs behind one PLX switch to eight across two switches on
the same CPU root complex. All 56 directed pairs pass the alias-proof test,
including all 32 cross-switch pairs, with no driver change.

Worth recording because the driver predicts otherwise: the read cap is gated
on bCommonPciSwitchFound || RYZEN || XEON_SPR, and this Xeon E5 v4 matches
none of them. The gate is conservative, not descriptive. Drops the old README
warning that 0015 might be claiming something untrue off-switch.

Two things did need attention:

- disable_acs_redir listed only the first switch. The second came up with
  ReqRedir+ CmpltRedir+, bouncing same-switch peer traffic through the root
  complex: 1.20 vs 1.68 GB/s. Correct but slow, so nothing fails loudly.
  Now lists every port of both switches including empty slots.
- Eight 64GB BAR1s span 1TB, 512GB per root port. Needs hpmmioprefsize=2T
  and patch 0001, or some GPUs silently get no Region 1.

Also note to check LnkSta when a pair looks slow: one card trained at x2 and
ran exactly half speed everywhere, with zero AER errors.")  | Aug 8, 2026  |  
|   |   | [add docs, dumps, utils for timings](https://github.com/bayley/cmpunlocker/commit/97d60f05e9bbf9c6fbcb9e848574fe82708ec5d3 "add docs, dumps, utils for timings")  | Aug 1, 2026  |  
|   |   | [survive kernel updates instead of falling back to stock](https://github.com/bayley/cmpunlocker/commit/6401e03fc4b35b1131e6b29a2268cdf5b38a9ae4 "survive kernel updates instead of falling back to stock

A kernel update left the card on whatever the distro rebuilt - the stock
driver in extra/ \(akmod\) or updates/dkms/ \(dkms\), reporting 8GB - or on
nouveau via nvidia-fallback.service. The patched modules were only ever
built for the kernel that was running at install time.

Rebuild them from the kernel update path instead:

  - build.sh takes CMPUNLOCKER_KVER, so it can build for a kernel that is
    not running, and skips the module reload when it does.
  - persist/ carries the hook for each distro family \(kernel-install,
    postinst.d, pacman\), all calling one rebuild.sh.
  - The build payload and the compile-time flags are copied to
    /var/lib/cmpunlocker and /etc/cmpunlocker, so a rebuild reproduces the
    user's overclock and keeps working if the clone is deleted.
  - cmpunlocker-rebuild.service holds boot until the modules exist, for
    the cases a hook cannot cover. Gives up after three failures rather
    than delaying every boot forever.
  - A depmod override makes the patched modules outrank the stock ones,
    which are rebuilt into a sibling directory on every kernel update.
  - nvidia-fallback.service is masked and nouveau blacklisted.
  - NVIDIA packages are pinned: a driver upgrade past driver/VERSION makes
    every later rebuild fail, which is the rollback this prevents. GPU
    firmware packages are excluded - they belong to linux-firmware and
    holding them back can wedge system upgrades.

Pinning is verified rather than assumed. dnf reports success even when it
could not reach the mirrors, so the result is read back before claiming
it, and the installer summary reports what actually happened.

Opt out with --no-persist / --no-pin. remove.sh undoes all of it.")  | Aug 5, 2026  |  
|   |   |   | Jul 18, 2026  |  
|   |   |   | Jul 15, 2026  |  
|   |   | [README: P2P across two PCIe switches, and the ACS gap it exposed](https://github.com/bayley/cmpunlocker/commit/5a7bb4b7e5056306fe49e8b824787659abb19914 "README: P2P across two PCIe switches, and the ACS gap it exposed

Scaled from four GPUs behind one PLX switch to eight across two switches on
the same CPU root complex. All 56 directed pairs pass the alias-proof test,
including all 32 cross-switch pairs, with no driver change.

Worth recording because the driver predicts otherwise: the read cap is gated
on bCommonPciSwitchFound || RYZEN || XEON_SPR, and this Xeon E5 v4 matches
none of them. The gate is conservative, not descriptive. Drops the old README
warning that 0015 might be claiming something untrue off-switch.

Two things did need attention:

- disable_acs_redir listed only the first switch. The second came up with
  ReqRedir+ CmpltRedir+, bouncing same-switch peer traffic through the root
  complex: 1.20 vs 1.68 GB/s. Correct but slow, so nothing fails loudly.
  Now lists every port of both switches including empty slots.
- Eight 64GB BAR1s span 1TB, 512GB per root port. Needs hpmmioprefsize=2T
  and patch 0001, or some GPUs silently get no Region 1.

Also note to check LnkSta when a pair looks slow: one card trained at x2 and
ran exactly half speed everywhere, with zero AER errors.")  | Aug 8, 2026  |  
|   |   | [unlock BAR1 resize to 64 GB](https://github.com/bayley/cmpunlocker/commit/3c176d7b8644e02a8e6f7a95c0254f5aa22c5b5f "unlock BAR1 resize to 64 GB")  | Aug 6, 2026  |  
|   |   | [close two rollback paths: firmware-owned dmem.bin, and remove.sh hanging](https://github.com/bayley/cmpunlocker/commit/89df12f6bf53b4e56cb6c3f53279361de9055e1e "close two rollback paths: firmware-owned dmem.bin, and remove.sh hanging

Booter payload override moved out of the firmware package's namespace.
cmpunlock.c read it from /lib/firmware/nvidia/ga100/gsp/dmem.bin, which
belongs to nvidia-gpu-firmware \(linux-firmware on Fedora\). That directory
already accumulates per-driver blobs on every firmware release, so an
update landing a dmem.bin there would have been preferred over the
built-in payload and quietly broken the unlock. Pinning that package is
not an option - it is a linux-firmware subpackage and holding it back
wedges system upgrades on a dependency conflict. Reading from
/var/lib/cmpunlocker/dmem.bin instead makes the question moot. The GSP
firmware the driver actually loads lives in the versioned directory from
the driver package, which is pinned.

remove.sh no longer swaps the running driver by default. Loading the
stock nvidia-drm against a CMP wedges the machine: traced it to that
exact modprobe, after which the kernel still answers pings while
userspace stops making progress. Nothing needed the swap - the modules
are already off disk, so the next boot picks up the stock driver on its
own, and an uninstall ends in a reboot anyway. Behind --reload for anyone
who wants it, now guarded with timeouts. Dropped the rmmod -f fallback:
forcing out a module that is still referenced can take the kernel with
it, which is a bad trade when a reboot finishes the job.

remove.sh also syncs after depmod, matching build.sh. Without it a power
cut between depmod and the next flush leaves a zero-length modules.dep,
and then nothing resolves on the next boot - not the NIC driver, not
storage. Hit this three times in testing via hard power-off.")  | Aug 6, 2026  |  
| View all files |  
## Repository files navigation
# cmpunlocker
Unlock tool for the NVIDIA CMP 170HX (GA100). Restores full SM compute, unlocked HBM2e memory geometry, PCIe Gen2 and GPU-to-GPU P2P — all clamped in firmware on the stock card.
**[Join our Discord community](https://discord.gg/CdHSakKSFv)** for support and discussions.
## Proof of Concept
Below are memory and performance results after applying the unlock:
### Unlock Results
## Requirements
  * Linux (x86-64 or aarch64)
  * Root access
  * NVIDIA CMP 170HX (8GB or 10GB)
  * **nvidia-open 610.43.03 or 610.43.02 already installed** (libs + firmware)
  * Kernel headers matching the running kernel (`linux-headers-$(uname -r)` / `kernel-devel`)
  * Secure Boot disabled (patched modules are unsigned)
  * Network access on first install (downloads matching stock `open-gpu-kernel-modules` sources)


## Install

```
sudo ./install.sh
```

Then perform a cold reboot (full power off, then boot). The correct memory geometry is selected automatically from the PCI device ID (`0x20C2` = 8GB -> 64GB, `0x2082` = 10GB -> 40GB).
### HBM Memory overclock
HBM Memory overclock 
`--mclk-ndiv=N` sets the FBPA PLL multiplier; the resulting clock is `N * 27` MHz. Any VBIOS works, on both `0x20C2` (8GB) and `0x2082` (10GB).

```
sudo ./install.sh --mclk-ndiv=70   # 1890 MHz
```
  
| NDIV  | Frequency  | Notes  |  
| --- | --- | --- |  
| 45  | 1215 MHz  | Stock 10gb  |  
| 60  | 1620 MHz  | Works on ~60% of 10gb cards  |  
| 54  | 1458 MHz  | Stock 8gb 250w vbios  |  
| 64  | 1728 MHz  | Stock 8gb 300w vbios  |  
| 70  | 1890 MHz  | Works on ~60% of 8gb cards  |  
| 73  | 1971 MHz  | Usually only on lucky 8gb cards  |  
Values below stock downclock the card, which is the way to stabilise a card that fails at stock.
Without the flag the overclock is compiled out entirely. The multiplier is compiled into the modules, so changing it means re-running `install.sh`. In a mixed 8GB+10GB system the same multiplier lands on every card.
If a value turns out to be unstable - reinstall without `--mclk-ndiv` (or run `./remove.sh`) from a working state.
### IOMMU
IOMMU  The installer adds `iommu=pt` (passthrough) to the kernel command line by default — it has negligible overhead and is required for VM passthrough. IOMMU must also be enabled in BIOS (VT-d on Intel, AMD-Vi / SVM on AMD). 
To skip IOMMU configuration:

```
sudo ./install.sh --no-iommu
```

### Surviving Kernel Updates (Anti-rollback)
Surviving Kernel Updates 
The patched modules are built against one specific kernel. Without help, the first kernel update leaves the card on the stock driver — reporting 8GB instead of 64GB — or on nouveau. The installer wires the rebuild into the kernel update path by default, so this does not happen.
A new kernel triggers a rebuild through the package manager hook for your distro, **before** you reboot:  
| Distro  | Hook  |  
| --- | --- |  
| Fedora, RHEL, openSUSE  | `/etc/kernel/install.d/95-cmpunlocker.install`  |  
| Debian, Ubuntu, HiveOS  | `/etc/kernel/postinst.d/cmpunlocker`  |  
| Arch  | `/etc/pacman.d/hooks/95-cmpunlocker.hook`  |  
`cmpunlocker-rebuild.service` is the safety net for what hooks cannot see — a hand-built kernel, a restored snapshot, or a hook that ran before the kernel headers were unpacked. It holds the boot until the patched modules exist, because a rig that silently comes up at 8GB is worse than one slow boot. It gives up after three consecutive failures rather than delaying every boot forever.
Two more things keep the stock driver from winning:
  * `/etc/depmod.d/cmpunlocker.conf` makes the patched modules outrank the stock ones. The distro driver is rebuilt on every kernel update too, into `extra/` (akmod) or `updates/dkms/` (dkms), right next to ours.
  * `nvidia-fallback.service` is masked and nouveau is blacklisted, so a driver that fails to load does not hand the card to nouveau.


**The NVIDIA packages are pinned to their installed version.** A driver upgrade past the versions in `driver/VERSION` makes every later rebuild fail, which is the rollback this is meant to prevent. This covers the GSP firmware the driver actually loads, from `/lib/firmware/nvidia/<driver-version>/`, which ships in the driver package itself.
GPU _firmware_ packages (`nvidia-gpu-firmware` and friends) are deliberately left unpinned — they belong to `linux-firmware`, and holding them back can wedge system upgrades on a dependency conflict. They are also no longer able to affect the unlock: the optional Booter payload override is read from `/var/lib/cmpunlocker/dmem.bin` rather than from `/lib/firmware/nvidia/ga100/gsp/`, a directory that firmware updates add files to and that no amount of pinning could safely protect.

```
sudo ./install.sh --no-pin       # allow driver upgrades, accept the risk
sudo ./install.sh --no-persist   # manage rebuilds yourself
```

Checking on it:

```
systemctl status cmpunlocker-rebuild
sudo /usr/lib/cmpunlocker/pin-packages.sh status
cat /var/log/cmpunlocker/rebuild-$(uname -r).log
sudo /usr/lib/cmpunlocker/rebuild.sh          # rebuild for the running kernel by hand
```

To take a pinned driver upgrade: `sudo /usr/lib/cmpunlocker/pin-packages.sh unpin`, upgrade, then re-run `install.sh` (which re-pins). If the new driver version is not in `driver/VERSION`, the build will refuse it.
Everything above is undone by `./remove.sh --yes`.
--- 
## Verify
After install and cold reboot:

```
# Memory — 8GB card should show 65536 MiB, 10GB card 40960 MiB
nvidia-smi --query-gpu=index,memory.total,pci.bus_id --format=csv

# Unlock logs
sudo dmesg | grep CMPUNLOCK

# P2P read matrix (multi-GPU, only with --p2p) — should be OK, not GNS
nvidia-smi topo -p2p r

# Link topology (multi-GPU) — PIX / PHB / SYS depending on how the GPUs are wired
nvidia-smi topo -m
```

### Benchmark
SM count, memory size and bandwidth, PCIe link speed, tensor core throughput (TF32/BF16/INT8), SM clock, and hardware features (NVENC/NVDEC):

```
./benchmark/nvidia_bench             # GPU 0, auto-sized iterations
./benchmark/nvidia_bench 1           # explicit GPU index
./benchmark/nvidia_bench 0 50        # explicit iteration count
./benchmark/nvidia_bench --csv       # one header line + one data line
./benchmark/nvidia_bench --help      # all options
```

A pre-built x86-64 binary is included. On aarch64 (or to rebuild), install the CUDA toolkit and build from source:

```
cd benchmark  nvcc -O3 -o nvidia_bench nvidia_bench.cu -lnvidia-ml -ldl \
  -gencode arch=compute_80,code=sm_80  strip nvidia_bench
```

## What Gets Unlocked  
| Feature  | Status  |  
| --- | --- |  
| Full SM compute throughput (SS0/SS1)  | Working  |  
| Memory geometry (64GB on 8GB cards, 40GB on 10GB cards)  | Working  |  
| PCIe Gen 2 speeds  | Working  |  
| GPU-to-GPU P2P (`cudaDeviceEnablePeerAccess`)  | Working (BAR1 P2P, needs kernel patches — see below)  |  
| HBM2e memory overclock/downclock  | Working  |  
| Persistence across kernel updates (auto-rebuild) (anti-rollback)  | Working  |  
| BAR1 64mb->64gb (requires Above 4G Decoding in BIOS)  | Working  |  
## GPU-to-GPU P2P
P2P works, over **BAR1 P2P** — not the mailbox. Measured **1.68 GB/s** each way on PCIe Gen2 x4, full 4-GPU mesh (12/12 directed pairs), verified with real transfers rather than `cudaDeviceCanAccessPeer()`.
That distinction matters: `--p2p` forces the _advertised_ caps to OK, so `canAccessPeer` and `nvidia-smi topo -p2p` report success whether or not a single byte moves. Verify with an actual copy, and ideally one that seeds the destination with a pattern that appears nowhere on the source — otherwise a mapping that silently aliases to local memory reads back as a pass.
### Why the mailbox path is a dead end
GA100 PCIe P2P normally goes through the write-mailbox (`P2P_CONNECTIVITY_PCIE_PROPRIETARY`). On these cards the driver configures it correctly — the connection type is chosen, both GSP RPCs return `NV_OK`, the mailbox BAR1 area is allocated — and transfers still move exactly zero bytes, with no error and no Xid. Hours went into that path before concluding it is not recoverable in software.
BAR1 P2P sidesteps it. Peer PTEs are rewritten from `GMMU_APERTURE_PEER` to `SYS_COH`/`SYS_NONCOH` pointing at the peer's BAR1 bus address, so peer memory is mapped as though it were system memory and the dead peer aperture is never used.
### What is required
All of these, together:  
| Piece  | Where  |  
| --- | --- |  
| BAR1 P2P HALs forced on for Ampere + peer-PTE rewrite  | `driver/patches/0011-p2p-bar1.patch`  |  
| Skip mailbox peer pre-registration  | `driver/patches/0013-skip-mailbox-peer-preinit.patch`  |  
| Restore the BAR1-P2P read cap  | `driver/patches/0015-bar1p2p-readcap-override.patch`  |  
| 64GB BAR1 on every GPU, from a normal boot  | `kernel-patches/`  |  
|  `RMForceStaticBar1=1` and `RMPcieP2PType=1`  |  `/etc/modprobe.d` + `update-initramfs -u`  |  
Two of the driver patches close gaps that stop BAR1 P2P from ever being selected:
  * **`0013`**—`_kbusInitP2P_GM107` pre-registers a mailbox peer id for every GPU pair at init, leaving `p2pPcie.peerNumberMask` non-zero. BAR1 P2P refuses to coexist with a mailbox peer, so that pre-registration silently forces the mailbox path forever, before any P2P is ever requested.
  * **`0015`**—`_p2pCapsGetHostSystemStatusOverPcieBar1` grants the read cap only for a common PCIe switch, Ryzen, or Xeon-SPR. On an older Xeon it hinges on `bCommonPciSwitchFound`, and `clFindCommonDownstreamBR()` fails to recognise the PLX switch these cards sit behind, returning `0xFF` even though `lspci` shows both GPUs on downstream ports of the same switch.


`0015` forces the read cap for any CMP pair, regardless of topology. That was written when all four GPUs sat behind one switch, and the README used to warn that it would be claiming something untrue across separate root ports. It has since been tested on eight GPUs split across two switches — see [Across two PCIe switches](https://github.com/bayley/cmpunlocker#across-two-pcie-switches) — and the override holds.
The static BAR1 requirement is also why `RMPcieP2PType=1` is needed: with the mailbox enabled, its 512KB area is allocated inside BAR1 and `ALIGN_UP`s the static mapping's start offset to 512MB, pushing the total to ~64.10GB — just over a 64GB BAR1, so static BAR1 fails and BAR1 P2P is refused.
Module parameters must be followed by `update-initramfs -u`; nvidia loads from the initramfs, so a new file in `/etc/modprobe.d` alone does nothing.
### Across two PCIe switches
The original bring-up used four GPUs behind a single PLX switch. The machine has since been filled to eight, four behind each of two switches on the **same** CPU root complex:

```
00:02.0 ── switch A ── 03:{04,08,0c,10,14}.0   GPU 0-3   (03:08.0 empty)
00:03.0 ── switch B ── 0a:{04,08,0c,10,14}.0   GPU 4-7   (0a:04.0 empty)

```

`nvidia-smi topo -m` reports `PIX` inside a switch and `PHB` between them, so cross-switch peer traffic goes up to the IIO and back down.
**It works.** All 56 directed pairs pass the alias-proof test — real peer reads _and_ real peer writes, no local aliasing — including all 32 cross-switch pairs. No driver change was needed beyond what was already in place.
This is worth stating plainly because the driver predicts otherwise. `_p2pCapsGetHostSystemStatusOverPcieBar1` sets the _write_ cap unconditionally but grants the _read_ cap only for `bCommonPciSwitchFound || RYZEN || XEON_SPR`. This host is a Xeon E5 v4 (`cpuType=0x8`), so NVIDIA's own logic rates cross-root-complex P2P **reads** unsupported here. Empirically they are fine. The whitelist is conservative, not descriptive — but note that it is NVIDIA's gate, not a law of the hardware, so re-run the alias test on any new host rather than assuming this generalises.
Two things do have to change for the second switch:
**1. ACS on every port, not just the first switch.** `pci=disable_acs_redir` originally listed only switch B. Switch A came up with `ReqRedir+ CmpltRedir+`, which forces peer traffic up to the root complex and back even between two GPUs on the _same_ switch. It still works — it is a performance bug, not a correctness one, which is exactly why it is easy to miss:  
| before  | after  |  
| --- | --- |  
| same-switch A  | 1.20 GB/s  | **1.68 GB/s**  |  
| same-switch B  | 1.68 GB/s  | 1.68 GB/s  |  
| cross-switch  | 1.45 GB/s  | 1.45 GB/s (unchanged)  |  
Cross-switch is unaffected because that traffic traverses the host bridge regardless. The fix is to list **all** ports of **both** switches, including downstream ports with nothing plugged in yet, so adding a card later needs no cmdline change. Verify with `lspci -vv | grep ACSCtl` — every port should read `ReqRedir- CmpltRedir-`. To test without rebooting, `setpci -s <port> ECAP_ACS+6.w=0000` applies it live.
**2. Address space.** Eight 64GB BAR1s at 128GB stride span `0x20000000000`–`0x2f000000000`, i.e. 512GB across the two root ports. This needs `pci=hpmmioprefsize=2T` plus a BIOS MMIO-high window large enough to hold it, and kernel patch `0001` — without `0001` the root port window is under-sized and some GPUs silently get no BAR1 at all.
Steady state, excluding the degraded card below:

```
same-switch  : 18 pairs, all 1.68 GB/s
cross-switch : 24 pairs, 1.07-1.45 GB/s  (~78% of same-switch)

```

Cross-switch is asymmetric — A→B runs a uniform 1.45 GB/s while B→A is 1.07–1.31 — which is a root-complex property, not something to tune out. Keep tensor-parallel groups inside one switch where it is free to do so, but cross-switch is entirely usable.
**Check link width when a pair looks slow.** One card trained at x2 instead of x4 and was exactly half speed everywhere (0.84 GB/s same-switch, ~0.58 cross). `LnkSta` said `Width x2` against a `LnkCap` of x16, with **zero** AER correctable errors — a clean negotiation at the wrong width, i.e. physical (riser, seating, contamination), not signal integrity under load. All CMP 170HX report `x4 (downgraded)` from an x16 capability; that x4 is normal, x2 is not.

```
nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.width.current --format=csv
```

### Two ways to hang the machine
Both cost a hard power cycle, TTY included:
  * `NVreg_RegistryDwords="ForceP2P=0x11"`.
  * Unloading and re-inserting the nvidia module on a live system. Apply module parameters by writing `/etc/modprobe.d`, rebuilding the initramfs, and rebooting.


## Documentation
  * [Installation](https://github.com/bayley/cmpunlocker/blob/master/docs/INSTALLATION.md) — requirements and install steps
  * [Architecture](https://github.com/bayley/cmpunlocker/blob/master/docs/ARCHITECTURE.md) — how the unlock works
  * [Debugging](https://github.com/bayley/cmpunlocker/blob/master/docs/DEBUGGING.md) — when something does not come up
  * [Contributing](https://github.com/bayley/cmpunlocker/blob/master/docs/CONTRIBUTING.md) — making changes


## Uninstall

```
sudo ./remove.sh --yes
```

Then perform a cold reboot (full power off, then boot).
This removes the patched modules from disk, undoes the kernel-update hooks, releases the package pin, and rebuilds the initramfs. The driver already running in memory is left alone — the card comes up on the stock driver at the next boot, which is the safe order.
`--reload` swaps the running driver for the stock one immediately instead of waiting for the reboot. It is off by default because loading the stock `nvidia-drm` against a CMP 170HX can wedge the machine: the card has no usable display engine, and the kernel keeps answering pings while userspace stops making progress. There is no reason to take that risk during an uninstall you are going to reboot from anyway.
## Community
Join our [Discord community](https://discord.gg/CdHSakKSFv) to discuss with other users.
## About
A cmpunlocker fork with working p2p on SYS-4028GR-TRT2
### Resources
### Contributing
### Stars
**26** stars
### Watchers
**2** watching
### Forks
[ forks](https://github.com/bayley/cmpunlocker/forks)
## 
No releases published
## 
## Languages
  * [Shell39.6%](https://github.com/bayley/cmpunlocker/search?l=shell)
  * [C31.5%](https://github.com/bayley/cmpunlocker/search?l=c)
  * [Cuda25.8%](https://github.com/bayley/cmpunlocker/search?l=cuda)
  * [Python3.1%](https://github.com/bayley/cmpunlocker/search?l=python)


You can’t perform that action at this time. 
