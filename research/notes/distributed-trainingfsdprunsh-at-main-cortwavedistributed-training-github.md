---
title: distributed-training/fsdp/run.sh at main · cortwave/distributed-training ·
  GitHub
id: distributed-trainingfsdprunsh-at-main-cortwavedistributed-training-github
tags:
- cmp170hx-training-feasibility-a0d4fc
created: '2026-10-27T14:37:29.082996Z'
updated: '2026-10-27T19:00:30.550157Z'
source: https://github.com/cortwave/distributed-training/blob/main/fsdp/run.sh
source_domain: github.com
fetched_at: '2026-10-27T14:37:29.081844Z'
fetch_provider: builtin
status: deprecated
type: note
tier: ground_truth
content_type: code
deprecated: false
summary: distributed-training/fsdp/run.sh at main · cortwave/distributed-training
  · GitHub
---

distributed-training/fsdp/run.sh at main · cortwave/distributed-training · GitHub
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
Files
Expand file tree
main
/
run.sh
Copy path
Blame
More file actions
Blame
More file actions
Latest commit
History
History
History
executable file
·
274 lines (222 loc) · 7.53 KB
main
/
run.sh
Copy path
Top
File metadata and controls
Code
Blame
executable file
·
274 lines (222 loc) · 7.53 KB
Raw
Copy raw file
Download raw file
Open symbols panel
Edit and raw actions
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274
#!
/bin/bash
#
benchmark.sh - FSDP benchmarking: GPU count × Batch size × Attention × Strategy
set
-e
#
Colors for output
RED=
'
\033[0;31m
'
GREEN=
'
\033[0;32m
'
YELLOW=
'
\033[1;33m
'
BLUE=
'
\033[0;34m
'
NC=
'
\033[0m
'
#
No Color
#
Configuration
MAX_STEPS=100
LOG_DIR=
"
runs/benchmarks_fsdp
"
TIMESTAMP=
$(
date +%Y%m%d_%H%M%S
)
#
Function to print colored output
print_info
() {
echo
-e
"
${GREEN}
[INFO]
${NC}
$1
"
}
print_warning
() {
echo
-e
"
${YELLOW}
[WARN]
${NC}
$1
"
}
print_error
() {
echo
-e
"
${RED}
[ERROR]
${NC}
$1
"
}
print_section
() {
echo
-e
"
\n
${BLUE}
═══════════════════════════════════════
${NC}
"
echo
-e
"
${BLUE}
$1
${NC}
"
echo
-e
"
${BLUE}
═══════════════════════════════════════
${NC}
\n
"
}
#
Function to check GPU availability
check_gpus
() {
if
!
command
-v nvidia-smi
&
>
/dev/null
;
then
print_error
"
nvidia-smi not found. CUDA setup required.
"
>&2
exit
1
fi
local
gpu_count=
$(
nvidia-smi --query-gpu=name --format=csv,noheader
|
wc -l
)
if
[
$gpu_count
-lt
1 ]
;
then
print_error
"
No GPUs detected
"
>&2
exit
1
fi
print_info
"
Detected
$gpu_count
GPU(s) available for testing
"
>&2
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
|
head -n 4
>&2
echo
"
"
>&2
echo
$gpu_count
}
#
Function to check Flash Attention availability
check_flash_attention
() {
python3 -c
"
import flash_attn; print('available')
"
2>
/dev/null
||
echo
"
not_available
"
}
#
Function to run a benchmark
run_benchmark
() {
local
num_gpus=
$1
local
batch_size=
$2
local
attn_impl=
$3
local
strategy=
$4
local
run_name=
"
fsdp_
${TIMESTAMP}
_
${strategy}
_
${num_gpus}
gpu_bs
${batch_size}
_
${attn_impl}
"
print_info
"
▶
${num_gpus}
GPU(s) | Batch
${batch_size}
|
${attn_impl}
|
${strategy}
"
#
Clear GPU memory
sleep 2
#
Run training with CSV logging
if
torchrun \
--nproc_per_node=
$num_gpus
\
--nnodes=1 \
fsdp/main.py \
--max_steps=
$MAX_STEPS
\
--batch_size=
$batch_size
\
--attn_implementation=
$attn_impl
\
--sharding_strategy=
$strategy
\
--run_name=
$run_name
\
--csv_path=
"
${LOG_DIR}
/benchmark_results.csv
"
\
2>&1
|
tee -a
"
benchmark_fsdp_
${TIMESTAMP}
.log
"
;
then
print_info
"
✓ Completed\n
"
return
0
else
print_error
"
✗ Failed\n
"
return
1
fi
}
#
Function to display CSV results
show_csv_results
() {
local
csv_file=
"
${LOG_DIR}
/benchmark_results.csv
"
if
[
-f
"
$csv_file
"
]
;
then
print_section
"
Benchmark Results
"
#
Use Python to format the CSV nicely
python3
<<
EOF
import csv
import sys
try:
with open('
$csv_file
', 'r') as f:
reader = csv.DictReader(f)
rows = list(reader)
if not rows:
print("No results in CSV")
sys.exit(0)
# Print header
print(f"{'GPUs':<5} {'Batch':<6} {'Attn':<8} {'Strategy':<10} {'Throughput':<12} {'Speedup':<9} {'Efficiency':<11} {'Step Time':<11} {'Memory':<9}")
print("-" * 105)
# Print rows
for row in rows:
attn_short = row['attn_implementation'][:6]  # Shorten for display
strategy_short = row.get('sharding_strategy', 'FULL')[:10]
print(f"{row['num_gpus']:<5} "
f"{row['batch_size_per_gpu']:<6} "
f"{attn_short:<8} "
f"{strategy_short:<10} "
f"{float(row['samples_per_sec']):<12.1f} "
f"{float(row['speedup']):<9.2f}x "
f"{float(row['efficiency_percent']):<11.1f}% "
f"{float(row['avg_step_time']):<11.4f}s "
f"{float(row['peak_memory_gb']):<9.2f}GB")
print(f"\nCSV file:
$csv_file
")
except Exception as e:
print(f"Error reading CSV: {e}")
EOF
else
print_warning
"
CSV file not found:
$csv_file
"
fi
}
#
Main benchmark suite
run_benchmark_suite
() {
local
available_gpus=
$1
print_section
"
FSDP Benchmark: GPU × Batch × Attention × Strategy
"
print_info
"
Timestamp:
$TIMESTAMP
"
print_info
"
Available GPUs:
$available_gpus
"
print_info
"
Max steps per test:
$MAX_STEPS
"
#
Check Flash Attention availability
local
has_flash=
$(
check_flash_attention
)
if
[
"
$has_flash
"
=
"
available
"
]
;
then
print_info
"
Flash Attention 2: ✓ Available
"
else
print_warning
"
Flash Attention 2: ✗ Not installed (will skip flash_attention_2 tests)
"
fi
echo
"
"
#
Test configurations
local
gpu_counts=(2 4)
local
batch_sizes=(2 4 8)
#
FSDP often allows larger batches due to sharding
local
attention_impls=(
"
sdpa
"
)
local
strategies=(
"
FULL_SHARD
"
"
SHARD_GRAD_OP
"
"
NO_SHARD
"
"
HYBRID_SHARD
"
)
#
Add flash attention if available
if
[
"
$has_flash
"
=
"
available
"
]
;
then
attention_impls+=(
"
flash_attention_2
"
)
fi
#
Calculate total tests
local
total_tests=0
for
num_gpus
in
"
${gpu_counts[@]}
"
;
do
if
[
$num_gpus
-le
$available_gpus
]
;
then
total_tests=
$((
total_tests
+
${
#
batch_sizes[@]}
*
${
#
attention_impls[@]}
*
${
#
strategies[@]}
))
fi
done
print_info
"
Total tests to run:
$total_tests
\n
"
local
current_test=0
#
Run all combinations
for
num_gpus
in
"
${gpu_counts[@]}
"
;
do
if
[
$num_gpus
-gt
$available_gpus
]
;
then
continue
fi
for
batch_size
in
"
${batch_sizes[@]}
"
;
do
for
attn_impl
in
"
${attention_impls[@]}
"
;
do
for
strategy
in
"
${strategies[@]}
"
;
do
current_test=
$((
current_test
+
1
))
print_info
"
Test
$current_test
/
$total_tests
"
if
!
run_benchmark
$num_gpus
$batch_size
$attn_impl
$strategy
;
then
print_warning
"
Test failed, continuing with next test...
"
fi
done
done
done
done
#
Show results
show_csv_results
#
Launch TensorBoard
print_section
"
Launching TensorBoard
"
tensorboard --logdir=
$LOG_DIR
--port=6007 --bind_all
>
/dev/null
2>&1
&
TB_PID=
$!
sleep 3
if
ps -p
$TB_PID
>
/dev/null
;
then
print_info
"
✓ TensorBoard started (PID:
$TB_PID
)
"
print_info
"
Access at: http://localhost:6007
"
#
Use different port than DDP
print_info
"
To stop: pkill -f tensorboard
"
echo
$TB_PID
>
tensorboard_fsdp.pid
else
print_error
"
Failed to start TensorBoard
"
fi
print_section
"
Benchmark Complete
"
print_info
"
Results saved to:
${LOG_DIR}
/benchmark_results.csv
"
}
#
Print usage
usage
() {
cat
<<
EOF
${BLUE}
FSDP Benchmarking: GPU × Batch × Attention × Strategy
${NC}
Usage:
$0
[OPTIONS]
Tests all combinations of:
- GPU counts: 1, 2, 4 (up to available)
- Batch sizes: 1, 2, 4, 8
- Attention: sdpa, eager, flash_attention_2
- Strategy: FULL_SHARD
Options:
--max-steps N   Steps per test (default:
$MAX_STEPS
)
--help          Show this help message
EOF
}
#
Parse command line arguments
while
[[
$#
-gt
0 ]]
;
do
case
$1
in
--max-steps)
MAX_STEPS=
"
$2
"
shift
2
;;
--help)
usage
exit
0
;;
*
)
print_error
"
Unknown option:
$1
"
usage
exit
1
;;
esac
done
#
Check GPU availability
AVAILABLE_GPUS=
$(
check_gpus
)
#
Run benchmark suite
run_benchmark_suite
$AVAILABLE_GPUS
You can’t perform that action at this time.