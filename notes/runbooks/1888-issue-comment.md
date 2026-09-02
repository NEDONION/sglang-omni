## Summary

First-pass Layer 1–5 runtime profile for `whisper_asr` on **one RTX 4090 (24 GB)**. Every Whisper
number currently in `docs/cookbook/whisper_asr.md` was measured on an H200, so this fills the
consumer-hardware end of the range rather than reproducing existing work.

Four headline results:

1. **Stock defaults run unchanged on 24 GB.** No `mem_fraction_static` or `max_running_requests`
   tuning was needed. Steady GPU memory grew 332 MiB across a 64× concurrency range
   (22154 → 22486 MiB), and 3264/3264 requests completed at every level with zero shed.
2. **Short-form throughput saturates at concurrency 32.** Going to 64 adds no throughput
   (47.68 → 47.15 req/s) and doubles mean latency (0.668 → 1.339 s).
3. **The saturation point is workload-shape dependent.** Long-form 60 s audio saturates at **8**
   and *regresses* at 32. A concurrency recommendation derived from short-form alone would
   overload it.
4. **Host-side orchestration is not the bottleneck** — a negative result. It accounts for 2.9 % of
   request latency at c=64, and its share *falls* as concurrency rises. The declining GPU busy
   ratio is not the host-bound signature #1798 §2 Layer 3 warns about.

The mechanism, stated as one claim:

> On a 24 GB consumer card the Whisper decode batch is already full at concurrency 32 — decode
> time is 581.00 ms at c=32 and 580.12 ms at c=64. Admission beyond that point converts directly
> into queue wait, which grows from 6.3 % to 47.8 % of request latency. Prefill is constant
> throughout because the encoder processes a fixed 30 s window.

## Frozen baseline

```text
run_id:
whisper-large-v3-rtx4090-20260902-0716

sglang-omni:
024d099bba15cc2023a55dca42936f118a512252   (main, clean)

dependency freeze sha256:
223870b864d199d2d3413eb77107d94f5109f93ade5cde40406744a0fd4d63c7

model:
openai/whisper-large-v3

datasets:
seedtts          27f4c1adee83b5b29b7c4b375f6b976324bda308
longlibriheavy   09bc067255eeb0d0bca62357ac985c2ebdc5169c
meanwhile        5a6b431a268523a6603f199d859fc25a24c22900

GPU:
NVIDIA GeForce RTX 4090, 24564 MiB, SM 8.9, driver 595.84, 450 W cap

host:
128 cores, 503 GiB RAM, dedicated single-tenant instance

container:
hongccc/sglang-omni:dev

stack:
torch 2.13.0+cu130 (CUDA 13.0) / sglang 0.5.18 / sglang-omni 0.1.4 / transformers 5.12.1
runtime /usr/bin/python3.12 (3.12.3)
```

Environment was built with this repo's own CI recipe (`.github/scripts/prepare_omni_venv.sh`),
which reuses the image's pinned wheels via `uv venv --system-site-packages` plus
`uv pip install --no-deps -e .`.

## Completeness gates

- 19 / 19 exact dependency pins verified by `prepare_omni_venv.sh`; import probe passed
- repo working tree clean at the profiled commit (`dirty: 0 files`)
- GPU verified idle before every server start (`0 %`, `1 MiB`, no compute apps)
- **dedicated instance, single tenant** — no co-tenant contention, so no CPU pinning was required
- Layer 3: 7 / 7 concurrency levels complete, **3264 / 3264 requests at every level**, zero shed,
  zero HTTP errors
- Layer 2: 4 / 4 levels profiled, 200 / 200 requests each, `stage_breakdown` present for all
- Layer 5: longlibriheavy-60 **300 / 300** at 3 / 3 levels; meanwhile **180 / 180** at 3 / 3 levels
- every measured level ran one discarded warmup pass plus the reported repeats
- 62 remote artifact files reconciled against 62 retrieved locally before the host was destroyed
- rented host destroyed after retrieval; billing stopped

## Layer 1 — GPU busy ratio

`nvidia-smi` sampled at 100 ms across a warmed pass (#1798 §2 Layer 1, method 2).
Note that `--sample-util` reports GPU memory, power and host CPU but **not** `utilization.gpu`,
so this needed a separate sampler.

| conc | samples | util mean | busy ratio (>0 %) | share >50 % | power mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 2245 | 48.4 % | 96.8 % | 41.4 % | 141.5 W |
| 8 | 780 | 41.6 % | 89.6 % | 25.6 % | 149.7 W |
| 32 | 521 | 41.8 % | 86.4 % | 39.3 % | 163.1 W |
| 64 | 585 | 36.1 % | 78.6 % | 23.1 % | 151.5 W |

The profiled passes compute `gpu_util_percent` independently and agree: 48.1 / 47.5 / 36.7 / 35.3 %.
Sample counts are well above the ~20 floor #1798 warns about, and each window brackets the load
test itself.

Utilization is far from saturation at every level and *falls* as concurrency rises. #1798 flags
that exact shape as possibly host-bound rather than as headroom, so Layer 2 was run to separate
the two. **Evidence: strong** — two independent methods agree.

## Layer 2 — stage breakdown

Used this repo's own request-level profiler (`--profile-events`), not py-spy — see the tooling
note at the end. Average ms per stage interval, 200 samples per level:

| stage interval | c=1 | c=8 | c=32 | c=64 |
|---|---:|---:|---:|---:|
| `stage_input_received→stage_complete` (total) | 95.84 | 251.22 | 722.03 | 1357.41 |
| `scheduler_prefill_end→stage_complete` (decode) | 58.82 | 186.46 | **581.00** | **580.12** |
| `scheduler_prefill_start→…_end` (prefill) | 30.72 | 32.08 | 32.85 | 33.55 |
| `scheduler_queue_enter→scheduler_prefill_start` (queue wait) | 2.21 | 8.18 | 45.53 | **649.08** |
| `scheduler_request_build_start→…_end` | 2.89 | 3.56 | 6.76 | 7.17 |
| `scheduler_request_build_end→scheduler_queue_enter` | 0.34 | 5.54 | 22.77 | 32.17 |

As a share of total: queue wait goes **2.3 % → 47.8 %**, while host-side stages
(`request_build` + handoff) stay at **3.4 % → 2.9 %**.

**Finding 1 — decode is already saturated at concurrency 32.** 581.00 ms at c=32 versus
580.12 ms at c=64 is statistically identical across a doubling of admitted concurrency. Adding
concurrency past 32 does not add work to the decode batch. **Evidence: strong.**

**Finding 2 — at c=64, 47.8 % of request latency is queue wait** (2.21 → 649.08 ms). This is
precisely where the Layer 3 latency doubling comes from: it is queueing, not slower compute.
**Evidence: strong.**

**Finding 3 — prefill is constant.** 30.72 → 33.55 ms across a 64× concurrency range, because the
encoder processes a fixed 30 s window. Its share collapses from 32.1 % to 2.5 %. **Evidence: strong.**

**Finding 4 — host-side orchestration is not the bottleneck.** `request_build` plus the
build→queue handoff account for 2.9 % at c=64, and their share *decreases* with concurrency
(3.4 % → 2.9 %). This rules out the host-bound explanation for the declining busy ratio.
**Evidence: strong. Stated explicitly as a negative result.**

## Layer 3 — short-form concurrency sweep

SeedTTS EN, 1088 clips, 3 measured repeats plus one discarded warmup per level.

| conc | req/s mean | req/s best | lat mean | lat p50 | lat p95 | RTFx | corpus WER | completed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.74 | 9.86 | 0.102 s | 0.102 s | 0.124 s | 46.1 | 0.0137 | 3264/3264 |
| 2 | 15.76 | 15.82 | 0.127 s | 0.130 s | 0.161 s | 74.6 | 0.0137 | 3264/3264 |
| 4 | 23.00 | 23.31 | 0.174 s | 0.171 s | 0.233 s | 108.9 | 0.0138 | 3264/3264 |
| 8 | 31.32 | 31.61 | 0.255 s | 0.251 s | 0.346 s | 148.3 | 0.0138 | 3264/3264 |
| 16 | 40.37 | 40.67 | 0.395 s | 0.393 s | 0.532 s | 191.2 | 0.0138 | 3264/3264 |
| **32** | **47.68** | **48.02** | 0.668 s | 0.658 s | 0.923 s | **225.8** | 0.0138 | 3264/3264 |
| 64 | 47.15 | 47.73 | 1.339 s | 1.338 s | 1.626 s | 223.3 | 0.0138 | 3264/3264 |

Resource envelope:

| conc | GPU mem steady | power peak | system CPU peak |
|---:|---:|---:|---:|
| 1 | 22154 MiB | 153.4 W | 47.9 % |
| 8 | 22156 MiB | 172.9 W | 42.2 % |
| 32 | 22350 MiB | 192.3 W | 34.4 % |
| 64 | 22486 MiB | 192.5 W | 47.0 % |

**Finding 5 — throughput saturates at 32 and does not scale to 64**, while mean latency doubles.
**Evidence: strong.**

**Finding 6 — stock defaults fit a 24 GB card with room to spare.** `mem_fraction_static` defaults
to 0.85 for Whisper and `max_running_requests` to 64; neither needed lowering, and peak power was
192.5 W against a 450 W cap. **Evidence: strong.**

**Finding 7 — no accuracy cost from concurrency.** Corpus WER is 0.0137–0.0138 at every level.
**Evidence: strong.**

For contrast, the H100 Qwen3-ASR profile in `docs/developer_reference/qwen3_asr_concurrency_profile.md`
sheds 1–4 % of requests at concurrency 64 when `request_build_max_pending` overflows. Whisper on a
4090 showed no shedding at any level.

## Layer 5 — functional regression on long-form audio

Also covers the long-form half of this issue's scope.

**longlibriheavy-60** — 100 clips of 60 s audio, 3 repeats plus warmup:

| conc | req/s mean | RTFx | lat mean | lat p95 | corpus WER | completed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.324 | 145.8 | 0.430 s | 0.530 s | 0.1063 | 300/300 |
| **8** | **7.566** | **474.6** | 1.036 s | 1.370 s | 0.1062 | 300/300 |
| 32 | 7.029 | 440.8 | 4.173 s | 5.669 s | 0.1062 | 300/300 |

**meanwhile** — 60 clips, structurally different monologue audio:

| conc | req/s mean | RTFx | lat mean | lat p95 | corpus WER | completed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.240 | 70.9 | 0.806 s | 0.979 s | 0.0987 | 180/180 |
| 8 | 4.542 | 259.8 | 1.692 s | 2.135 s | 0.0984 | 180/180 |
| **32** | **4.913** | **281.0** | 5.592 s | 7.768 s | 0.0985 | 180/180 |

**Finding 8 — no accuracy regression under concurrency on either long-form set.** WER is flat
within each dataset (0.1062–0.1063 and 0.0984–0.0987). The higher absolute WER relative to
short-form is dataset difficulty, not a regression. **Evidence: strong.**

**Finding 9 — three workload shapes, three behaviours.** **Evidence: strong.**

| workload | saturates at | behaviour at conc 32 |
|---|---|---|
| SeedTTS short-form | **32** | flat (47.68 → 47.15 req/s) |
| longlibriheavy-60 | **8** | **regresses** (7.57 → 7.03 req/s), latency 4× |
| meanwhile | still climbing | slight gain (4.54 → 4.91 req/s) |

#1798 §2 Layer 1 warns that the trend can flip entirely between workload shapes. This run observes
that flip directly, and it is the single most transferable result here: any published concurrency
guidance for Whisper has to state the audio-length regime it applies to.

**Finding 10 — RTFx is higher on long-form** (474.6 versus 225.8) because longer audio amortizes
the fixed per-request cost, consistent with the constant ~33 ms prefill in Layer 2.
**Evidence: strong.**

## Layer 4 — not run, and why

#1798 §3 item 6 conditions Layer 4 on "each suspected overhead source found in Layer 2". Layer 2
found none: host-side stages sit at 2.9 % and their share falls with concurrency. There is no
single-variable knob to A/B without first establishing *why* decode leaves the GPU at 35–48 %,
which needs a kernel timeline this environment could not capture (see tooling note).

**This is a deliberate skip under the methodology's own precondition, not an omission.**

The natural follow-up, if someone wants it, is an A/B of `max_running_requests` at 32 versus the
default 64 on 24 GB hardware — Finding 2 makes that a concrete hypothesis rather than a guess.
I did not run it, so I am not claiming an outcome.

## Findings, evidence, recommendation

| # | Finding | Evidence | Recommendation | Strength |
|---|---|---|---|---|
| 1 | Stock defaults run unchanged on 24 GB; steady memory 22154→22486 MiB across 64× concurrency | Layer 3, 3264/3264 at every level | Document 24 GB as a supported Whisper configuration; no consumer-GPU profile file appears necessary | **Strong** |
| 2 | Short-form saturates at concurrency 32; latency doubles to 64 with no throughput gain | Layer 3 | Treat 32 as the short-form operating point on this class of card | **Strong** |
| 3 | Decode batch already full at 32 — 581.00 vs 580.12 ms | Layer 2 | Admission above 32 buys nothing here for short-form | **Strong** |
| 4 | 47.8 % of c=64 latency is queue wait | Layer 2 | Worth A/B-ing `max_running_requests` nearer the decode batch limit on 24 GB | **Strong** |
| 5 | Host-side orchestration is not the bottleneck (2.9 %, share falling) | Layer 2 | None — negative result, rules out the host-bound reading of Layer 1 | **Strong** |
| 6 | Saturation point is workload-shape dependent; long-form saturates at 8 and regresses at 32 | Layer 5, two datasets | Any published concurrency guidance must state the audio-length regime | **Strong** |
| 7 | No accuracy cost from concurrency on any of three datasets | Layers 3 and 5 | None needed | **Strong** |
| 8 | GPU utilization stays 35–48 % with decode dominant; cause unresolved | Layer 1 (two methods), Layer 2 rules out host-bound | Needs a kernel timeline before any kernel-level work is justified | **Weak** |

On Finding 8, one code-level narrowing: with `enable_pre_lm_encoder` on (the default),
`_resolve_encoder_graph_buckets` sets `capture_limit = pre_lm_max_batch_size` (8), and
`pre_lm_max_batch_size` also caps the encoder batch itself — so the encoder path is graph-covered
across its whole range and is **not** where the gap lives. That points at decode.
*Code-level evidence only, not re-verified at runtime.*

## Reproduction

```bash
# server, stock defaults
sgl-omni serve --model-path openai/whisper-large-v3 --port 8000

# Layer 3
python -u -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-large-v3 \
  --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup \
  --sample-util --util-gpu-ids 0 --util-interval 0.5 --fingerprint \
  --save-raw-dir raw/seedtts_en --output seedtts_en_sweep.json

# Layer 1 — separate sampler, one level at a time
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader,nounits -lms 100 > util_c$C.csv &
python -u -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-large-v3 \
  --concurrencies $C --repeats 1 --warmup --output bench_c$C.json

# Layer 2
python -u -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-large-v3 \
  --concurrencies 1,8,32,64 --repeats 1 --warmup --max-samples 200 \
  --profile-events --profile-urls http://127.0.0.1:8000 \
  --profile-event-dir layer2/events --output layer2/profiled.json

# Layer 5 + long-form coverage
python -u -m benchmarks.eval.benchmark_asr_longform \
  --dataset longlibriheavy-60 --port 8000 --model-path openai/whisper-large-v3 \
  --concurrencies 1,8,32 --repeats 3 --warmup --max-samples 100 \
  --sample-util --util-gpu-ids 0 --fingerprint --output longform60.json

python -u -m benchmarks.eval.benchmark_asr_longform \
  --dataset meanwhile --port 8000 --model-path openai/whisper-large-v3 \
  --concurrencies 1,8,32 --repeats 3 --warmup --max-samples 60 \
  --sample-util --util-gpu-ids 0 --fingerprint --output meanwhile.json
```

## Raw artifacts

Every number in the tables above is reproduced inline, so nothing here depends on an artifact a
reader cannot see. The underlying files are retained locally under
`whisper-large-v3-rtx4090-20260902-0716/`:

| artifact | size | contents |
|---|---:|---|
| `seedtts_en_sweep.json` | 102 KB | Layer 3, all 7 levels, with `provenance` and `environment_fingerprint` blocks |
| `layer2/profiled.json` | 65 KB | Layer 2 `stage_breakdown` + per-pass `gpu_util_percent` |
| `longform60.json` / `meanwhile.json` | 62 KB each | Layer 5 |
| `layer1/util_c{1,8,32,64}.csv` | 7–33 KB | raw `nvidia-smi` 100 ms samples |
| `raw/**/*.jsonl` | 21 files | per-request records |
| `serve.log` | 78 MB | server log, mostly the 118 Inductor autotune rounds |

Happy to post any of these inline or in a follow-up if a reviewer wants to re-derive a specific
number. The rented host has been destroyed, so the local copy is the only one.

## Operational notes

Whole run took **2.2 GPU-hours** on a rented RTX 4090. Timings, in case they help whoever picks up
the next model in this batch:

| step | wall time |
|---|---|
| container image pull | 5 min |
| environment via `prepare_omni_venv.sh` | 30 s |
| model + 3 datasets (38 GB HF cache) | 3.5 min |
| **server cold start** | **~10 min** |
| Layer 3 full sweep | 35 min |
| Layers 1 + 2 + 5 | ~25 min |

Cold start is dominated by **118 rounds of Inductor autotune**. Worth knowing before planning a
segmented rental, since stopping and restarting an instance pays it again.

## Two tooling notes for #1798 §1/§2 (per §3 item 9)

Both are general rather than Whisper-specific, so flagging them here rather than editing the
methodology directly — happy to open a PR against `METHODOLOGY.md` once #1850 lands.

1. **py-spy needs `CAP_SYS_PTRACE`, which stock containers do not grant.** §4's tools index gives
   the py-spy command with no stated precondition. On a standard Docker or rented-cloud container
   it fails with `Permission denied (os error 13)` even as root, because `yama.ptrace_scope = 1`
   and the sysctl is read-only from inside. Worth noting the `--cap-add SYS_PTRACE` requirement at
   instance-creation time, and that `--profile-events` is the privilege-free alternative for
   stage-level attribution. The same restriction blocks `nsys`, which is why Finding 8 stays weak.
2. **`--sample-util` does not report GPU utilization.** Its `resources` block carries
   `gpu_memory_used_*`, `power_peak_w`, `system_cpu_peak_percent` and
   `gpu_process_cpu_peak_percent` — but no `utilization.gpu`. Layer 1's busy ratio needs a separate
   `nvidia-smi` sampler. Easy to assume otherwise from the flag name, and the profiled pass *does*
   carry `gpu_util_percent`, which makes the gap easier to miss.
