---
title: 'Inconsistent P2P bandwidth and NCCL allreduce bus bandwidth · Issue #814 ·
  NVIDIA/nccl · GitHub'
id: inconsistent-p2p-bandwidth-and-nccl-allreduce-bus-bandwidth-issue-814-nvidianccl
tags:
- cmp170hx-training-feasibility-a0d4fc
- gap-cc-2
- nccl
- all-reduce
- host-staged
- 6-rank
- measured
- community-reported
created: '2026-10-27T14:36:29.346067Z'
updated: '2026-10-27T19:00:31.777993Z'
source: https://github.com/NVIDIA/nccl/issues/814
source_domain: github.com
fetched_at: '2026-10-27T14:36:29.344734Z'
fetch_provider: builtin
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'NVIDIA/nccl issue #814 (comaniac, Mar-Apr 2023) — a 6x RTX 4090 host-staged
  all-reduce receipt, the closest committed artifact to a 5-rank no-P2P measurement.
  [T2 community measurement, 6 ranks.] Command: NCCL_P2P_DISABLE=1 mpirun -np 1 all_reduce_perf
  -d half -b 1M -e 64M -f 2 -g 6 on a 6x4090 box whose topology mixes PHB and SYS
  links; the thread never states the link generation, but its p2pBandwidthLatencyTest
  P2P-disabled matrix shows 20-22 GB/s unidirectional per pair, i.e. PCIe 4.0 x16-class
  links (inference, not a quoted spec). Results (out-of-place algbw/busbw): 1 MiB
  2.04/3.39; 2 MiB 1.70/2.83; 4 MiB 1.64/2.74; 8 MiB 1.54/2.56; 16 MiB 1.54/2.57;
  32 MiB 1.48/2.47; 64 MiB 1.45/2.42; ''Avg bus bandwidth: 2.70867''. So a 6-rank
  host-staged all-reduce on a fast PCIe fabric lands at ~2.4-2.7 GB/s busbw / ~1.45-1.5
  GB/s algbw — 14-27x above the cc-2 falsifier of 0.1 GB/s algbw. The reporter also
  posts an all-pairs all_reduce_perf matrix (busbw 2.61-4.50 GB/s between pairs) and
  p2pBandwidthLatencyTest matrices with and without --sm_copy (P2P writes 9.6-10.2
  GB/s, P2P disabled 20-22 GB/s, device-to-self 911-922 GB/s). NCCL maintainer sjeaugey''s
  diagnosis: ''That''s unfortunately a common problem, in particular with AMD CPUs...
  NCCL needs to communicate in both directions; it''s not just a copy. This tends
  to decrease performance, even more than 2x... With 6 GPUs and an allreduce you have
  6 flows interleaving on the CPU'' — i.e. the CPU, not the link, is the binding resource
  for host-staged collectives at N>=6. Caveat: PCIe 4.0 x16-class with P2P disabled,
  not Gen2 x4; the CPU-bound regime does not transfer directly to a 2 GB/s link, where
  the link binds instead.'
utility_score: 16.0
---

Inconsistent P2P bandwidth and NCCL allreduce bus bandwidth · Issue #814 · NVIDIA/nccl · GitHub
Skip to content
You signed in with another tab or window.
Reload
to refresh your session.
You signed out in another tab or window.
Reload
to refresh your session.
You switched accounts on another tab or window.
Reload
to refresh your session.
Dismiss alert
Inconsistent P2P bandwidth and NCCL allreduce bus bandwidth
#814
New issue
Copy link
New issue
Copy link
Closed
Closed
Inconsistent P2P bandwidth and NCCL allreduce bus bandwidth
#814
Copy link
Description
comaniac
opened
on Mar 31, 2023
Issue body actions
Hi there,
I'm now working on a machine with 6 RTX 4090 GPUs. The topology is shown as follows:
GPU0
GPU1
GPU2
GPU3
GPU4
GPU5
CPU Affinity
GPU0
X
PHB
SYS
SYS
SYS
SYS
0-63
GPU1
PHB
X
SYS
SYS
SYS
SYS
0-63
GPU2
SYS
SYS
X
SYS
SYS
SYS
0-63
GPU3
SYS
SYS
SYS
X
PHB
SYS
0-63
GPU4
SYS
SYS
SYS
PHB
X
SYS
0-63
GPU5
SYS
SYS
SYS
SYS
SYS
X
0-63
A weird part in this platform is GPU0-GPU1 and GPU3-GPU4 are connected via PHB while the rests are connected via SYS. While I cannot control this topology, I'm trying to achieve a reasonable bandwidth when performing AllReduce. Here is the command and the log (I've added
`NCCL_DISABLE_P2P=1`
based on
NVIDIA/nccl-tests#117
):
```
NCCL_P2P_DISABLE=1 mpirun --allow-run-as-root -np 1 ./nccl-tests/build/all_reduce_perf -d half -b 1M -e 64M -f 2 -g 6
# nThread 1 nGpus 6 minBytes 1048576 maxBytes 67108864 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
# Using devices
#  Rank  0 Group  0 Pid   2249 on 28dc06d2a640 device  0 [0x01] NVIDIA GeForce RTX 4090
#  Rank  1 Group  0 Pid   2249 on 28dc06d2a640 device  1 [0x02] NVIDIA GeForce RTX 4090
#  Rank  2 Group  0 Pid   2249 on 28dc06d2a640 device  2 [0x2d] NVIDIA GeForce RTX 4090
#  Rank  3 Group  0 Pid   2249 on 28dc06d2a640 device  3 [0x41] NVIDIA GeForce RTX 4090
#  Rank  4 Group  0 Pid   2249 on 28dc06d2a640 device  4 [0x42] NVIDIA GeForce RTX 4090
#  Rank  5 Group  0 Pid   2249 on 28dc06d2a640 device  5 [0x61] NVIDIA GeForce RTX 4090
#       size         count     type  redop      root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)        (elements)                                     (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
     1048576      262144      float     sum      -1     515.2    2.04    3.39      0    523.1    2.00    3.34      0
     2097152      524288      float     sum      -1   1233.1    1.70    2.83      0   1230.7    1.70    2.84      0
     4194304      1048576    float     sum      -1   2554.7    1.64    2.74      0   2540.1    1.65    2.75      0
     8388608      2097152    float     sum      -1   5462.3    1.54    2.56      0   5464.2    1.54    2.56      0
    16777216     4194304    float     sum      -1    10892    1.54    2.57      0    10929    1.54    2.56      0
    33554432     8388608    float     sum      -1    22618    1.48    2.47      0    22649    1.48    2.47      0
    67108864    16777216   float     sum      -1    46206    1.45    2.42      0    46237    1.45    2.42      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 2.70867
```
On the other hand, here is the results of running
`p2pBandwidthLatencyTest`
without and with
`--sm_copy`
:
```
# Common info
[P2P (Peer-to-Peer) GPU Bandwidth Latency Test]
Device: 0, NVIDIA GeForce RTX 4090, pciBusID: 1, pciDeviceID: 0, pciDomainID:0
Device: 1, NVIDIA GeForce RTX 4090, pciBusID: 2, pciDeviceID: 0, pciDomainID:0
Device: 2, NVIDIA GeForce RTX 4090, pciBusID: 2d, pciDeviceID: 0, pciDomainID:0
Device: 3, NVIDIA GeForce RTX 4090, pciBusID: 41, pciDeviceID: 0, pciDomainID:0
Device: 4, NVIDIA GeForce RTX 4090, pciBusID: 42, pciDeviceID: 0, pciDomainID:0
Device: 5, NVIDIA GeForce RTX 4090, pciBusID: 61, pciDeviceID: 0, pciDomainID:0
Device=0 CAN Access Peer Device=1
Device=0 CAN Access Peer Device=2
Device=0 CAN Access Peer Device=3
Device=0 CAN Access Peer Device=4
Device=0 CAN Access Peer Device=5
Device=1 CAN Access Peer Device=0
Device=1 CAN Access Peer Device=2
Device=1 CAN Access Peer Device=3
Device=1 CAN Access Peer Device=4
Device=1 CAN Access Peer Device=5
Device=2 CAN Access Peer Device=0
Device=2 CAN Access Peer Device=1
Device=2 CAN Access Peer Device=3
Device=2 CAN Access Peer Device=4
Device=2 CAN Access Peer Device=5
Device=3 CAN Access Peer Device=0
Device=3 CAN Access Peer Device=1
Device=3 CAN Access Peer Device=2
Device=3 CAN Access Peer Device=4
Device=3 CAN Access Peer Device=5
Device=4 CAN Access Peer Device=0
Device=4 CAN Access Peer Device=1
Device=4 CAN Access Peer Device=2
Device=4 CAN Access Peer Device=3
Device=4 CAN Access Peer Device=5
Device=5 CAN Access Peer Device=0
Device=5 CAN Access Peer Device=1
Device=5 CAN Access Peer Device=2
Device=5 CAN Access Peer Device=3
Device=5 CAN Access Peer Device=4

***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.
So you can see lesser Bandwidth (GB/s) and unstable Latency (us) in those cases.

P2P Connectivity Matrix
     D\D     0     1     2     3     4     5
     0       1     1     1     1     1     1
     1       1     1     1     1     1     1
     2       1     1     1     1     1     1
     3       1     1     1     1     1     1
     4       1     1     1     1     1     1
     5       1     1     1     1     1     1
```
```
# Without --sm_copy
Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 911.08  20.10  22.16  22.14  22.33  22.37
     1  20.39 920.20  22.26  22.24  22.34  22.22
     2  22.18  22.37 920.74  22.09  22.40  22.24
     3  22.24  22.42  22.25 922.37  20.46  22.24
     4  22.19  22.39  22.25  20.43 920.62  22.23
     5  22.09  22.30  22.22  22.32  22.38 921.96
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 912.68   9.90   9.90   9.90   9.90   9.90
     1   9.90 922.37   9.90   9.90   9.90   9.90
     2  10.01  10.01 922.24  10.01  10.01  10.01
     3  10.23  10.23  10.23 921.83  10.23  10.23
     4   9.65   9.64   9.65   9.65 921.29   9.65
     5  10.22  10.22  10.22  10.22  10.22 922.37
Unidirectional P2P=Enabled Bandwidth (P2P Reads) Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 911.08   9.90  10.01  10.22   9.64  10.22
     1   9.90 922.37  10.01  10.22   9.64  10.22
     2   9.90   9.90 921.17  10.22   9.64  10.21
     3   9.95   9.95  10.45 920.20   9.95  10.21
     4   9.95   9.95  10.45   9.95 920.74  10.21
     5   9.95   9.95  10.46   9.95   9.95 921.29
Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 918.31  21.12  30.53  29.85  30.07  30.86
     1  21.27 922.37  30.56  29.79  29.92  30.91
     2  30.64  30.73 924.01  30.52  30.81  30.76
     3  29.63  29.93  30.58 924.53  21.06  30.95
     4  29.81  29.87  30.58  21.36 923.69  30.79
     5  30.95  31.08  30.79  30.80  30.89 923.17
Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 918.80  19.91  19.91  19.91  19.91  19.91
     1  19.91 923.19  19.90  19.91  19.91  19.91
     2  19.91  19.90 923.19  19.91  19.91  20.42
     3  19.91  19.91  19.91 923.19  19.91  19.91
     4  19.91  19.91  19.91  19.91 923.46  19.91
     5  19.91  19.91  20.42  19.91  19.91 923.19
P2P=Disabled Latency Matrix (us)
   GPU     0      1      2      3      4      5
     0   1.46  11.62  11.68  11.60  13.65  12.72
     1  11.53   1.38  11.71  11.65  11.64  11.52
     2  11.53  11.53   1.34  11.43  11.56  11.85
     3  11.76  12.99  13.79   1.43  13.32  11.86
     4  11.37  20.18  11.51  20.08   1.32  20.12
     5  11.59  18.63  12.03  11.67  11.65   1.38

   CPU     0      1      2      3      4      5
     0   2.68   8.49   8.51   8.10   8.07   8.14
     1   8.36   2.63   8.51   8.13   8.17   8.04
     2   8.43   8.41   2.62   8.19   8.16   8.11
     3   8.21   8.16   8.28   2.54   7.96   7.85
     4   8.14   8.14   8.30   7.93   2.53   7.86
     5   8.24   8.18   8.32   7.89   7.82   2.56
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3      4      5
     0   1.46   1.23   1.20   1.20   1.20   1.20
     1   1.19   1.33   1.19   1.16   1.15   1.19
     2   1.17   1.22   1.35   1.19   1.19   1.18
     3   1.20   1.15   1.17   1.36   1.17   1.22
     4   1.21   1.21   1.23   1.17   1.33   1.16
     5   1.17   1.17   1.19   1.19   1.18   1.41

   CPU     0      1      2      3      4      5
     0   2.66   2.17   2.16   2.16   2.14   2.18
     1   2.21   2.62   2.15   2.17   2.14   2.19
     2   2.25   2.24   2.68   2.19   2.26   2.24
     3   2.17   2.08   2.12   2.63   2.13   2.15
     4   2.14   2.12   2.11   2.16   2.64   2.10
     5   2.15   2.09   2.47   2.07   2.11   2.67
P2P=Enabled Latency (P2P Reads) Matrix (us)
   GPU     0      1      2      3      4      5
     0   1.47  11.72  10.26  11.54  11.82  11.82
     1  10.35   1.35  11.45  12.10  11.36  11.72
     2  10.80  11.55   1.35  11.74  11.17  11.26
     3  11.83  11.27  10.71   1.37  12.10  11.45
     4  10.81  10.62  11.17  11.36   1.34  11.46
     5  11.90  11.26  10.61  11.53  11.81   1.42

   CPU     0      1      2      3      4      5
     0   2.61   5.52   5.55   5.18   5.20   5.14
     1   5.50   2.63   5.54   5.20   5.22   5.26
     2   5.50   5.54   2.68   5.19   5.21   5.49
     3   5.53   5.46   5.54   2.60   5.23   5.27
     4   5.56   5.49   5.62   5.25   2.62   5.24
     5   5.51   5.59   5.55   5.19   5.17   2.64
```
```
# With --sm_copy
Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 912.14  19.98  22.21  22.21  22.37  22.20
     1  20.43 921.23  22.14  22.08  22.37  22.16
     2  22.16  22.44 920.74  22.07  22.39  22.13
     3  22.18  22.31  22.28 920.74  20.41  22.18
     4  22.26  22.36  22.23  20.43 921.83  22.25
     5  22.16  22.39  22.21  22.35  22.33 922.92
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 912.14  10.14  10.14  10.14  10.14  10.14
     1  10.14 938.44  10.14  10.14  10.14  10.14
     2  10.19  11.04 937.88  11.04  11.04  11.04
     3  10.49  10.48  10.50 938.44  10.49  10.50
     4  10.49  10.50  10.49  10.49 938.53  10.49
     5  10.97  10.96  10.97  10.96  10.97 939.00
Unidirectional P2P=Enabled Bandwidth (P2P Reads) Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 912.68  12.43  12.43  12.43  12.43  12.43
     1  12.43 916.42  12.43  12.43  12.43  12.43
     2  12.36  12.36 914.28  12.35  12.34  12.34
     3  12.46  12.46  12.46 915.89  12.46  12.46
     4  12.46  12.46  12.46  12.46 913.21  12.46
     5  12.40  12.40  12.40  12.40  12.40 915.38
Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 918.52  21.15  30.61  29.81  30.00  31.01
     1  21.04 923.69  30.56  29.64  29.94  30.86
     2  30.68  30.69 923.19  30.60  30.80  30.86
     3  29.59  29.89  30.55 923.92  21.59  30.91
     4  29.78  29.89  30.54  21.37 923.46  30.88
     5  30.90  30.93  30.97  30.93  31.12 923.46
Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 919.12  14.92  20.43  20.45  20.46  20.45
     1  14.84 932.28  20.43  20.43  20.45  20.45
     2  20.44  20.43 929.78  20.98  20.99  21.91
     3  20.45  20.44  20.99 931.17  14.91  20.99
     4  20.45  20.43  20.98  14.90 931.98  20.98
     5  20.45  20.45  21.91  20.98  20.99 932.28
P2P=Disabled Latency Matrix (us)
   GPU     0      1      2      3      4      5
     0   1.46  11.59  11.68  11.91  14.13  13.42
     1  19.78   1.38  11.68  14.16  11.90  13.48
     2  13.78  11.76   1.32  11.80  11.96  11.50
     3  11.37  11.47  11.45   1.43  12.61  11.85
     4  11.93  11.92  12.70  11.87   1.32  11.85
     5  11.53  11.51  11.36  11.37  11.79   1.37

   CPU     0      1      2      3      4      5
     0   2.69   8.47   8.50   8.05   8.03   8.14
     1   8.40   2.65   8.51   8.05   8.05   8.18
     2   8.38   8.39   2.63   8.15   8.15   8.14
     3   8.15   8.16   8.28   2.54   7.97   7.88
     4   8.16   8.23   8.32   7.92   2.54   7.89
     5   8.23   8.14   8.27   7.81   7.80   2.53
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3      4      5
     0   1.47   2.59   2.61   2.60   2.61   2.60
     1   2.40   1.34   2.40   2.39   2.40   2.40
     2   2.42   2.43   1.35   2.43   2.42   2.42
     3   2.47   2.46   2.45   1.35   2.45   2.44
     4   2.43   2.42   2.43   2.43   1.34   2.42
     5   2.43   2.43   2.43   2.42   2.43   1.41

   CPU     0      1      2      3      4      5
     0   2.68   2.21   2.17   2.18   2.17   2.17
     1   2.17   2.61   2.15   2.13   2.14   2.15
     2   2.20   2.20   2.69   2.23   2.21   2.23
     3   2.09   2.12   2.09   2.64   2.09   2.11
     4   2.12   2.09   2.07   2.08   2.62   2.07
     5   2.11   2.05   2.06   2.06   2.08   2.58
P2P=Enabled Latency (P2P Reads) Matrix (us)
   GPU     0      1      2      3      4      5
     0   1.47   3.37   3.36   3.37   3.37   3.36
     1   3.16   1.33   3.17   3.16   3.15   3.16
     2   3.21   3.19   1.34   3.21   3.19   3.19
     3   3.26   3.23   3.24   1.35   3.24   3.25
     4   3.18   3.17   3.15   3.17   1.34   3.16
     5   3.25   3.24   3.24   3.25   3.24   1.41

   CPU     0      1      2      3      4      5
     0   2.73   2.15   2.16   2.16   2.17   2.15
     1   2.15   2.62   2.16   2.16   2.16   2.17
     2   2.20   2.19   2.68   2.18   2.17   2.19
     3   2.09   2.08   2.09   2.59   2.09   2.35
     4   2.09   2.12   2.12   2.11   2.64   2.11
     5   2.11   2.11   2.12   2.12   2.24   2.63
```
We can see that even with
`--sm_copy`
,
`p2pBandwidthLatencyTest`
could still the similar results. My understanding is that this means there still something that can be improved in NCCL, so I attempted to optimize the NCCL topology file. Here is the one dumped from NCCL:
```
<system version="1">
  <cpu numaid="-1" arch="x86_64" vendor="AuthenticAMD" familyid="143" modelid="49">
    <pci busid="0000:01:00.0" class="0x030000" vendor="0x10de" device="0x2684" subsystem_vendor="0x1458" subsystem_device="0x40bf" link_speed="16.0 GT/s PCIe" link_width="16">
      <gpu dev="0" sm="89" rank="0" gdr="0"/>
    </pci>
    <pci busid="0000:02:00.0" class="0x030000" vendor="0x10de" device="0x2684" subsystem_vendor="0x1458" subsystem_device="0x40bf" link_speed="16.0 GT/s PCIe" link_width="16">
      <gpu dev="1" sm="89" rank="1" gdr="0"/>
    </pci>
    <pci busid="0000:2d:00.0" class="0x030000" vendor="0x10de" device="0x2684" subsystem_vendor="0x1458" subsystem_device="0x40bf" link_speed="16.0 GT/s PCIe" link_width="16">
      <gpu dev="2" sm="89" rank="2" gdr="0"/>
    </pci>
    <pci busid="0000:41:00.0" class="0x030000" vendor="0x10de" device="0x2684" subsystem_vendor="0x1458" subsystem_device="0x40bf" link_speed="16.0 GT/s PCIe" link_width="16">
      <gpu dev="3" sm="89" rank="3" gdr="0"/>
    </pci>
    <pci busid="0000:42:00.0" class="0x030000" vendor="0x10de" device="0x2684" subsystem_vendor="0x1458" subsystem_device="0x40bf" link_speed="16.0 GT/s PCIe" link_width="16">
      <gpu dev="4" sm="89" rank="4" gdr="0"/>
    </pci>
    <pci busid="0000:61:00.0" class="0x030000" vendor="0x10de" device="0x2684" subsystem_vendor="0x1458" subsystem_device="0x40bf" link_speed="16.0 GT/s PCIe" link_width="16">
      <gpu dev="5" sm="89" rank="5" gdr="0"/>
    </pci>
    <nic>
      <net name="eth0" dev="0" speed="10000" port="0" latency="0.000000" guid="0x0" maxconn="65536" gdr="0"/>
    </nic>
  </cpu>
</system>
```
I measured the AllReduce bandwidth between every GPU pairs (all numbers are the bus bandwidth in GB/s from
`all_reduce_perf`
):
GPU0
GPU1
GPU2
GPU3
GPU4
GPU5
GPU0
X
2.61
3.60
3.58
3.57
3.59
GPU1
X
3.60
3.57
3.57
3.60
GPU2
X
3.62
3.61
4.50
GPU3
X
2.61
3.61
GPU4
X
3.61
GPU5
X
According to this result, I changed the topology file (by adjusting the order of
`<pci>`
blocks) to make a ring with the order of
`0 → 2 → 5 → 3 → 1 → 4`
. However, it doesn't work.
In summary,
Why NCCL AllReduce can only achieve ~2.6 GB/s bus bandwidth while P2P test shows a much higher value even with
`--sm_copy`
?
Is there anything I could do with the NCCL topo file or other configurations?
Thanks!
Reactions are currently unavailable
Activity
sjeaugey
commented
on
Apr 3, 2023
sjeaugey
on Apr 3, 2023
Member
More actions
That's unfortunately a common problem, in particular with AMD CPUs. Even though the p2pBandwidth benchmark shows 10GB/s (which is already not great), NCCL needs to communicate in both directions; it's not just a copy. This tends to decrease performance, even more than 2x. It seems that AMD CPUs don't like having lots of flows at the same time, and as the flows interleave, they break the pipelining of the traffic, causing even lower performance. With 6 GPUs and an allreduce you have 6 flows interleaving on the CPU, which I'm not surprised leads to that performance.
comaniac
commented
on
Apr 4, 2023
comaniac
on Apr 4, 2023
Author
More actions
Thanks for your reply and it's sad to know that this is a common problem for the RTX 40-series and AMD CPU combination. If possible, I'll perform benchmarking on a machine with Intel CPUs to verify this, but anyways like you pointed out, I'm not sure if this would help if the p2p bandwidth is already that worse. I'm closing this issue for now.
Thanks again!
comaniac
closed this
as
completed
on Apr 4, 2023
Bear-Beer
commented
on
Aug 9, 2024
Bear-Beer
on Aug 9, 2024
More actions
Any update for the combination of AMD CPU + NCCL?
I was wondering if the latest version of NCCL has fixed this issue?
sjeaugey
commented
on
Aug 12, 2024
sjeaugey
on Aug 12, 2024
Member
More actions
NCCL 2.23 should disable P2P in more cases, but nothing more than setting
`NCCL_P2P_LEVEL=PHB`
already does.
Nero10578
commented
on
Apr 23, 2025
Nero10578
on Apr 23, 2025
More actions
NCCL 2.23 should disable P2P in more cases, but nothing more than setting
`NCCL_P2P_LEVEL=PHB`
already does.
Sorry this is an old thread, but is this thread implying P2P is supposed to work on RTX 4090s? I thought Geforce cards since 20 series had PCIe P2P disabled?
kiskra-nvidia
commented
on
Apr 23, 2025
kiskra-nvidia
on Apr 23, 2025
Member
More actions
No, it is not implying that... As far as I know you are correct that P2P is not supported on any recent GeForce cards. Sylvain's comment was about recent NCCL releases more actively avoiding the use of P2P in cases where we expect it to perform poorly,  even if it is nominally supported.
Sign up for free
to join this conversation on GitHub.
Already have an account?
Sign in to comment
Metadata
Metadata
Assignees
No one assigned
Labels
No labels
No labels
Type
No type
Projects
No projects
Milestone
No milestone
Relationships
None yet
Development
No branches or pull requests
Issue actions
Open in GitHub Copilot app
You can’t perform that action at this time.