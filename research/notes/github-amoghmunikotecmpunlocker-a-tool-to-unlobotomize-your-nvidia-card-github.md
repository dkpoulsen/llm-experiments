---
title: 'GitHub - amoghmunikote/cmpunlocker: A tool to unlobotomize your NVIDIA card!
  · GitHub'
id: github-amoghmunikotecmpunlocker-a-tool-to-unlobotomize-your-nvidia-card-github
tags:
- cmp170hx-training-feasibility-a0d4fc
- hardware
- cmp170hx
- unlock
- pcie-link
- primary-source
created: '2026-10-27T11:09:05.233615Z'
updated: '2026-10-27T19:00:31.414545Z'
source: https://github.com/amoghmunikote/cmpunlocker
source_domain: github.com
fetched_at: '2026-10-27T11:09:04.610120Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'amoghmunikote/cmpunlocker — the upstream CMP 170HX unlock tool (808 stars,
  55 commits, latest 2d10f89 2026-10-08, release v0.5 2026-10-05), and the primary
  source for what the community unlock actually restores. README ''What Gets Unlocked''
  table lists as Working: full SM compute throughput (SS0/SS1), memory geometry (64
  GB on 8 GB cards, 40 GB on 10 GB cards), PCIe Gen2 speeds, ECC DRAM and SRAM, full
  BAR1 size (64 GB), JTAG host-to-JTAG register access, VFIO-based passthrough, GPU
  profiling, and persistence across reboot via patched modules. Two points matter
  for this run. First, ECC is listed as Working here, which contradicts club-170hx''s
  ''ECC untested / upstream work in progress'' and the Consensus-Protocol wiki''s
  frontier listing — so ECC availability is a live contradiction in the corpus, not
  a settled fact. Second, P2P is NOT in the upstream feature list at all: BAR1 P2P
  exists only in forks (bayley/cmpunlocker, admunch888/cmpunlocker), so ''no P2P''
  is correct for the upstream tool and wrong as a hardware statement. Requirements:
  Linux x86-64, root, nvidia-open 610.xx.xx+ already installed, matching kernel headers,
  Secure Boot disabled, network access on first install. Install is sudo ./install.sh
  with optional --profile=8gb or --profile=10gb to force the memory geometry, then
  a reboot. Uninstall is ./remove.sh --yes plus reboot. Python 3 is used at build
  time to select the 8 GB/10 GB geometry.'
---

*Suggested by [[club-170hxdocscmp170hx-understandingmd-at-main-pixelmlclub-170hx-github]] — Cited as the upstream unlock tool the club's driver stack is built on (88e39ce); primary source for Gen2 link, 64 GB memory and 74-SM unlocks*

[Skip to content](https://github.com/amoghmunikote/cmpunlocker#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/amoghmunikote/cmpunlocker) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/amoghmunikote/cmpunlocker) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/amoghmunikote/cmpunlocker) to refresh your session. Dismiss alert
[**49** Branches](https://github.com/amoghmunikote/cmpunlocker/branches)[](https://github.com/amoghmunikote/cmpunlocker/tags)
Go to file
Open more actions menu
## Latest commit
[Add support for nvidia-open 615.78.08](https://github.com/amoghmunikote/cmpunlocker/commit/2d10f891a58824f0591c3332c4eccf37ff36b086)
success
Oct 8, 2026
[2d10f89](https://github.com/amoghmunikote/cmpunlocker/commit/2d10f891a58824f0591c3332c4eccf37ff36b086) · Oct 8, 2026
## History
[55 Commits](https://github.com/amoghmunikote/cmpunlocker/commits/master/)
Open commit details
55 Commits
## Folders and files  
| Name  | Name  | Last commit message  | Last commit date  |  
| --- | --- | --- | --- |  
|   |   |   | Sep 13, 2026  |  
|   |   |   | Oct 5, 2026  |  
|   | Sep 13, 2026  |  
|   |   | [Add support for nvidia-open 615.78.08](https://github.com/amoghmunikote/cmpunlocker/commit/2d10f891a58824f0591c3332c4eccf37ff36b086 "Add support for nvidia-open 615.78.08")  | Oct 8, 2026  |  
|   |   |   | Jul 29, 2026  |  
|   |   |   | Sep 13, 2026  |  
|   |   |   | Sep 13, 2026  |  
|   |   |   | Jul 18, 2026  |  
|   |   |   | Sep 13, 2026  |  
|   |   |   | Sep 7, 2026  |  
|   |   |   | Jul 15, 2026  |  
|   |   | Oct 4, 2026  |  
|   |   |   | Sep 13, 2026  |  
|   |   |   | Sep 13, 2026  |  
|   |   |   | Sep 13, 2026  |  
|   |   |   | Sep 10, 2026  |  
| View all files |  
## Repository files navigation
## What is cmpunlocker?
cmpunlocker restores numerous features that are restricted in firmware/OTP configuration of the NVIDIA CMP 170HX. cmpunlocker has been featured by multiple outlets like wccftech, Tom's Hardware and LinusTechTips. 
**[Join our Discord community](https://discord.gg/CdHSakKSFv)** for support and discussions.
## Proof of Concept
Below are memory and performance results after applying the unlock:  
| **Memory Unlock Results**  |  
| --- |  
| **Performance Benchmarks ([OpenCL-Benchmark](https://github.com/ProjectPhysX/OpenCL-Benchmark))**  |  
| --- |  
## Requirements
  * Linux (x86-64)
  * Root access
  * NVIDIA CMP 170HX
  * **nvidia-open 610.xx.xx+ already installed** (libs + firmware)
  * Kernel headers matching the running kernel (`linux-headers-$(uname -r)` / `kernel-devel`)
  * Secure Boot disabled (patched modules are unsigned)
  * Network access on first install (downloads matching stock `open-gpu-kernel-modules` sources)
  * Python 3 (used at build time to select 8GB/10GB geometry)


## Install
To install cmpunlocker, run the following command:

```
sudo ./install.sh
```

To force a certain memory profile, use the `--profile` option:

```
sudo ./install.sh --profile=8gb    # 8GB card → 64GB unlock
sudo ./install.sh --profile=10gb   # 10GB card → 40GB unlock
```

Then perform a reboot.
## What Gets Unlocked  
| Feature  | Status  |  
| --- | --- |  
| Full SM compute throughput (SS0/SS1)  | Working ✓  |  
| Memory geometry (64GB on 8GB cards, 40GB on 10GB cards)  | Working ✓  |  
| PCIe Gen 2 speeds  | Working ✓  |  
| Error-Correcting Code (ECC) DRAM and SRAM   | Working ✓  |  
| Full BAR1 Size (64GB)  | Working ✓  |  
| JTAG (Host2Jtag register access)  | Working ✓  |  
| VFIO-based passthrough  | Working ✓  |  
| GPU profiling  | Working ✓  |  
| Persistence across reboot (patched modules)  | Working ✓  |  
## Uninstall
To uninstall cmpunlocker, run the following command:

```
sudo ./remove.sh --yes
```

Then perform a reboot.
## Contributions
Please read [docs/CONTRIBUTING.md](https://github.com/amoghmunikote/cmpunlocker/blob/master/docs/CONTRIBUTING.md) before opening a PR.
## Support & Community
Having issues? Need help? Join our [Discord community](https://discord.gg/CdHSakKSFv) to discuss with other users and get support.
## About
A tool to unlobotomize your NVIDIA card!
[github.com/amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
### Resources
### Contributing
### Stars
**808** stars
### Watchers
**43** watching
### Forks
[ forks](https://github.com/amoghmunikote/cmpunlocker/forks)
## 
[ Release v0.5Latest Oct 5, 2026 ](https://github.com/amoghmunikote/cmpunlocker/releases/tag/v0.5)
## 
## Languages
  * [Shell80.2%](https://github.com/amoghmunikote/cmpunlocker/search?l=shell)
  * [Python17.4%](https://github.com/amoghmunikote/cmpunlocker/search?l=python)
  * [C2.2%](https://github.com/amoghmunikote/cmpunlocker/search?l=c)
  * [Makefile0.2%](https://github.com/amoghmunikote/cmpunlocker/search?l=makefile)


You can’t perform that action at this time. 
