---
title: 'GitHub - mcrl/tccl: Thunder Research Group''s Collective Communication Library
  · GitHub'
id: github-mcrltccl-thunder-research-groups-collective-communication-library-github
tags:
- cmp170hx-training-feasibility-a0d4fc
- nccl
- pcie-fabric
- collective-communication
- numa
- bounce-buffer-placement
- code-artifact
created: '2026-10-27T10:32:34.534217Z'
updated: '2026-10-27T19:00:31.220644Z'
source: https://github.com/mcrl/tccl
source_domain: github.com
fetched_at: '2026-10-27T10:32:33.911835Z'
fetch_provider: crawl4ai
status: review
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: 'TCCL (Thunder Research Group, SNU; ASPLOS 2024; github.com/mcrl/tccl): a
  fork of NCCL for ''PCIe-dependent systems'' — clusters where GPUs must communicate
  through the CPU''s PCIe host bridge and possibly NUMA interconnects, with no NVLink
  and no external PCIe switches. Its premise is the key claim for this run: existing
  collective libraries do NOT achieve the performance their analytical models predict
  on PCIe-dependent systems, for two reasons that only appear when traffic goes through
  the CPU — transfers slow down when multiple transfers traverse the CPU in a particular
  pattern because of NUMA architecture, and bounce buffers are poorly placed when
  GPUs communicate via CPU memory. Neither congestion pattern shows up on links that
  bypass the CPU, so NCCL''s topology logic never learns to avoid them. TCCL''s pathfinder
  profiles each candidate communication path and merges single-node paths into multi-node
  paths (brute force is exponential in GPU count), and the runtime is a drop-in libnccl.so
  replacement selected via NCCL_ALGO=TCCL plus TCCL_XML_FILE. Thin source (README
  only, ~1.1k words) but ground-truth tier: it is the code artifact behind the ASPLOS
  paper, last committed Jul 2025.'
utility_score: 14.0
---

*Suggested by [[efficient-expert-parallel-communication]] — official TCCL repository for collective communication on PCIe-dependent GPU systems*

[Skip to content](https://github.com/mcrl/tccl#start-of-content)
You signed in with another tab or window. [Reload](https://github.com/mcrl/tccl) to refresh your session. You signed out in another tab or window. [Reload](https://github.com/mcrl/tccl) to refresh your session. You switched accounts on another tab or window. [Reload](https://github.com/mcrl/tccl) to refresh your session. Dismiss alert
[**1** Branch](https://github.com/mcrl/tccl/branches)
Go to file
Open more actions menu
## Latest commit
[Merge pull request](https://github.com/mcrl/tccl/commit/351d064856e322ec6e4546808e7d1e433d24f941) [#4](https://github.com/mcrl/tccl/pull/4) [from gajagajago/master](https://github.com/mcrl/tccl/commit/351d064856e322ec6e4546808e7d1e433d24f941)
Open commit details
Jul 8, 2025
[351d064](https://github.com/mcrl/tccl/commit/351d064856e322ec6e4546808e7d1e433d24f941) · Jul 8, 2025
## History
[9 Commits](https://github.com/mcrl/tccl/commits/master/)
Open commit details
9 Commits
## Folders and files  
| Name  | Name  | Last commit message  | Last commit date  |  
| --- | --- | --- | --- |  
|   |   |   | Apr 25, 2024  |  
|   |   |   | Mar 7, 2024  |  
|   |   |   | Mar 7, 2024  |  
|   | Mar 7, 2024  |  
| [Remove hardcoded values from runtime](https://github.com/mcrl/tccl/commit/e615c91ce2ea92fbb33a32924824ad43b6414bc0 "Remove hardcoded values from runtime")  | Apr 25, 2024  |  
|   |   | [Remove hardcoded values from runtime](https://github.com/mcrl/tccl/commit/e615c91ce2ea92fbb33a32924824ad43b6414bc0 "Remove hardcoded values from runtime")  | Apr 25, 2024  |  
|   |   |   | Apr 25, 2024  |  
|   |   |   | Mar 7, 2024  |  
|   |   |   | Mar 9, 2024  |  
|   |   |   | Mar 9, 2024  |  
|   |   |   | Mar 7, 2024  |  
|   |   |   | Jul 8, 2025  |  
| View all files |  
## Repository files navigation
Collective communication library from Thunder Research Group.
## Introduction
TCCL is a fork of NCCL that aims to accelerate collective communication primitives on systems that lack proprietary high-speed interconnects between GPUs, such as NVLink, and external PCIe switches. We refer to these systems as _PCIe-dependent systems_ because GPUs must communicate through the PCIe host bridge of the CPU and possibly the interconnect between NUMA nodes or CPU sockets.
We found that existing collective communication libraries do not achieve the target performance that their performance model estimates in PCIe-dependent systems. One of the reasons is that transfers slow down when multiple transfers go through the CPU in a specific pattern due to NUMA architectures. Another is the poor placement of the bounce buffers when GPUs communicate via CPU memory. Such congestion patterns are not identified or avoided by the existing collective communication libraries because the phenomenon does not appear when communication utilizes links outside the CPU.
TCCL consists of TCCL pathfinder and TCCL runtime. The pathfinder is a module that searches for communication paths. The pathfinder always profiles each communication path of interest to check the performance it achieves to avoid congestion by taking it. Since the number of possible communication paths increases exponentially with the number of GPUs in a cluster, it is infeasible to brute-force search all the communication paths. Thus, the pathfinder measures the performance of intra-node paths and merges the single-node paths to construct multi-node paths.
The runtime is a modification of NCCL such that the paths collective communication operations use are replaced with the paths found by the pathfinder while allowing a transparent interface to use TCCL without any code changes.
## Build
### TCCL runtime
As TCCL runtime is a fork of NCCL, the build process is similar to NCCL. To build the library:

```
$ cd tccl
$ make -j src.build
$ ls build/lib
... libnccl.so ...
```

For other options, check the original NCCL repository.
### TCCL pathfinder
TCCL pathfinder is compiled with cmake:

```
$ cd tccl/tools
$ mkdir build
$ cd build
$ cmake ..
$ make -j `nproc`
$ ls
... pathfinder ...
```

## Searching communication paths
Once you have built TCCL, you should execute the pathfinder to generate an XML file that contains communication path information. As the pathfinder is an MPI program that runs on multiple nodes, the launch command depends on the cluster environment. In the cluster managed by SLURM, you can launch the pathfinder as follows:

```
$ mkdir workspace
$ salloc -N 3 --exclusive \
    mpirun -mca btl ^openib -mca pml ucx --bind-to none \
    -npernode 11 \
    tccl/tools/build/pathfinder -o workspace
$ python tccl/tools/scripts/preprocess_xml.py --dir workspace --output path_info.xml
```

The pathfinder generates intermediate XML files in the workspace directory. `preprocess_xml.py` merges the intermediate XML files and generates a single XML file that be used by the TCCL runtime.
The pathfinder should be launched in 3 nodes: one node is a main node, and the other two nodes are sub-nodes used to profile inter-node transfers. The main node require `3 + (number of GPUs) * 2` proceeses: 1 main process, 2 for inter-node transfers, and 2 for each GPU. The sub-nodes requires just 2 processes: 1 for inter-node transfer, and another for a GPU that sends/receives data. The above example simply launches the pathfinder on 3 nodes with 11 processes per node, which is suitable for a cluster with 4 GPUs per node.
## Usage
To use TCCL in applications, replace the shared library of NCCL with that of TCCL using mechanisms such as `LD_PRELOAD`. Note that the shared library of TCCL is named `libnccl.so` to maximize compatibility. You should also set two environment variables: `TCCL_XML_FILE` and `NCCL_ALGO`. `TCCL_XML_FILE` should be the path to the XML file generated by the pathfinder. `NCCL_ALGO` should be set to TCCL to use the communication paths found by TCCL. Otherwise, it will fall back to NCCL's algorithms.
For example, the following command launches 2-node 8-GPU AllReduce `nccl-tests` with TCCL:

```
$ mpirun -mca btl ^openib -mca pml ucx --bind-to none \
    -npernode 4 \
    -x LD_PRELOAD=tccl/build/lib/libnccl.so \
    -x NCCL_DEBUG=INFO \
    -x NCCL_DEBUG_SUBSYS=TCCL \
    -x NCCL_ALGO=TCCL,RING \
    -x TCCL_XML_FILE=$TCCL_XML_FILE \
    nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2
... NCCL INFO TCCL channel setup done ...
... Out of bounds values : 0 OK ...
```

If the installation was successful, the log will state that the TCCL channels have been created properly and there are no wrong values. You may compare the performance with NCCL by removing `NCCL_ALGO=TCCL` from the scripts.
## Artifact Evaluation
See [Artifact Evaluation](https://github.com/mcrl/tccl/blob/master/AE.md) for the details to reproduce the results in the [paper](https://dl.acm.org/doi/10.1145/3620666.3651362).
## References
If you find TCCL relevant to your research, please consider citing:

```
@inproceedings{tccl-asplos24,
    title = {{TCCL}: Discovering Better Communication Paths for {PCIe} {GPU} Clusters},
    author = {Kim, Heehoon and Ryu, Junyeol and Lee, Jaejin},
    booktitle = {Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3},
    pages = {999–1015},
    year = {2024},
    url = {https://doi.org/10.1145/3620666.3651362},
}

```

## About
Thunder Research Group's Collective Communication Library
### Resources
### Stars
**53** stars
### Watchers
**2** watching
### Forks
[ forks](https://github.com/mcrl/tccl/forks)
## 
[ tccl-aecLatest Mar 9, 2024 ](https://github.com/mcrl/tccl/releases/tag/tccl-aec)
## 
  * [**csehydrogen** Heehoon Kim](https://github.com/csehydrogen)
  * [**gajagajago** Junyeol Ryu](https://github.com/gajagajago)


## Languages
  * [C++68.6%](https://github.com/mcrl/tccl/search?l=c%2B%2B)
  * [HTML19.4%](https://github.com/mcrl/tccl/search?l=html)
  * [C7.2%](https://github.com/mcrl/tccl/search?l=c)
  * [Python1.4%](https://github.com/mcrl/tccl/search?l=python)
  * [JavaScript0.8%](https://github.com/mcrl/tccl/search?l=javascript)
  * [Cuda0.7%](https://github.com/mcrl/tccl/search?l=cuda)
  * [Other1.9%](https://github.com/mcrl/tccl/search?l=Other)


You can’t perform that action at this time. 
