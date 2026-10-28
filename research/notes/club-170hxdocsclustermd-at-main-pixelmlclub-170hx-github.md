---
title: club-170hx/docs/CLUSTER.md at main · PixelML/club-170hx · GitHub
id: club-170hxdocsclustermd-at-main-pixelmlclub-170hx-github
tags:
- cmp170hx-training-feasibility-a0d4fc
- cmp-170hx
- cluster-design
- parallelism-strategy
- pipeline-parallel
- expert-parallel
- community-guidance
created: '2026-10-27T10:59:40.091041Z'
updated: '2026-10-27T19:00:31.390474Z'
source: https://github.com/PixelML/club-170hx/blob/main/docs/CLUSTER.md
source_domain: github.com
fetched_at: '2026-10-27T10:59:39.469601Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'Short (524-word) community design doc for building a low-cost SM80 cluster
  out of CMP 170HX cards. Its parallelism table is a direct counterpoint to the repo''s
  claim that expert and tensor parallelism are ruled out by the fabric: pipeline parallel
  is the ''preferred starting point'' because it communicates stage boundaries rather
  than frequent full-tensor collectives; tensor parallel is ''use only after measurement''
  since all-reduce traffic can erase compute gains on slow PCIe; data parallel is
  ''good for independent jobs/throughput'' but each replica needs its own weights
  or a suitable sharding design; expert parallel is ''workload-dependent'' with routing
  traffic and runtime support that ''must be measured''. It also notes there is no
  NVLink in the tested configuration and that PCIe links may negotiate conservatively
  under virtualization, which changes which parallelism strategy wins. Rest of the
  doc is operational: storage layout, preflight inventory, scheduling rules and a
  rule that multi-node claims must publish link speed, transport, topology, batch
  size and failure behaviour.'
utility_score: 14.0
---

[Skip to content](https://github.com/PixelML/club-170hx/blob/main/docs/CLUSTER.md#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/PixelML/club-170hx/blob/main/docs/CLUSTER.md) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/PixelML/club-170hx/blob/main/docs/CLUSTER.md) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/PixelML/club-170hx/blob/main/docs/CLUSTER.md) to refresh your session. Dismiss alert
## Collapse file tree
## Files
Search this repository(forward slash)` forward slash/`
/
# CLUSTER.md
Copy path
More file actions
More file actions
## Latest commit
[docs: establish CMP 170HX community knowledge base (](https://github.com/PixelML/club-170hx/commit/7336363dbf37ab7f9404aa8e9b82ce91286344e5)[#1](https://github.com/PixelML/club-170hx/pull/1)[)](https://github.com/PixelML/club-170hx/commit/7336363dbf37ab7f9404aa8e9b82ce91286344e5)
Open commit details
Aug 30, 2026
[7336363](https://github.com/PixelML/club-170hx/commit/7336363dbf37ab7f9404aa8e9b82ce91286344e5) · Aug 30, 2026
## History
[History](https://github.com/PixelML/club-170hx/commits/main/docs/CLUSTER.md)
Open commit details
History
54 lines (38 loc) · 2.47 KB
/
# CLUSTER.md
Copy path
## File metadata and controls
  * Preview
  * 

54 lines (38 loc) · 2.47 KB
Copy raw file
Download raw file
Outline
Edit and raw actions
# Building a low-cost SM80 cluster
The goal is a useful, reproducible compute pool—not a claim that CMP 170HX cards become A100s. The design should lean into HBM capacity and work around weak interconnects.
## Topology first

```
shared model storage
        │
        ▼
  compute node / VM
   ├── CMP card 0 ── model stages 0…N
   ├── CMP card 1 ── model stages N…M
   └── CMP card 2 ── model stages M…end

```

There is no NVLink in the tested configuration. PCIe links may also negotiate conservatively under virtualization. This changes which parallelism strategy wins.
## Parallelism choices  
| Strategy  | Fit on this hardware  | Why  |  
| --- | --- | --- |  
| Pipeline parallel  | Preferred starting point  | Communicates stage boundaries instead of frequent full-tensor collectives  |  
| Tensor parallel  | Use only after measurement  | All-reduce traffic can erase compute gains on slow PCIe  |  
| Data parallel  | Good for independent jobs/throughput  | Each replica needs its own weights or a suitable sharding design  |  
| Expert parallel  | Workload-dependent  | Routing traffic and runtime support must be measured  |  
The three-card DeepSeek-V4-Flash result validates pipeline parallelism as a practical path. It does not prove pipeline parallel is optimal for every model.
## Storage layout
Use shared high-capacity storage for model weights and local disks for code, environments, caches that need high IOPS, and transient outputs.
Before a launch, record:

```
findmnt -T /path/to/model/library
df -hT /path/to/model/library
du -sh /path/to/model/checkpoint
```

Do not silently fall back to a small root disk when shared storage disappears. Treat a missing mount or unsafe free-space level as a stop condition.
## Scheduling rules
  1. Give every card a stable anonymous label and record its physical slot/riser/power mapping privately.
  2. Run a preflight inventory and temperature check before each workload.
  3. Prevent benchmark and serving jobs from sharing a card unless the experiment explicitly measures contention.
  4. Preserve raw results with runtime/model commits and power/thermal telemetry.
  5. Cold-cycle only after orderly guest/host shutdown and only when PCIe recovery requires it.


## Scaling beyond one node
Measure the network before distributed serving or training. Model capacity can scale across nodes while latency and throughput regress from communication. Publish link speed, transport, topology, batch size, and failure behavior alongside any multi-node claim.
You can’t perform that action at this time. 
