# Whisper ASR runtime profile — first pass on a single RTX 4090

First-pass Layer 1–5 profile for `whisper_asr` following the methodology in [#1798](https://github.com/sgl-project/sglang-omni/issues/1798).
Run on consumer hardware, which is the gap this fills: every Whisper number currently
in `docs/cookbook/whisper_asr.md` was measured on an H200.

**Headline:** stock defaults run unchanged on a 24 GB card. Short-form throughput
saturates at concurrency 32, but the saturation point is **workload-shape dependent** —
long-form 60 s audio saturates at 8 and regresses at 32. The bottleneck is decode batch
capacity, not host-side orchestration.

---

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, SM 8.9, driver 595.84, TDP cap 450 W |
| Host | 128 cores, 503 GiB RAM |
| Isolation | **Dedicated single-tenant instance.** No co-tenant on the GPU, no CPU pinning needed — `nvidia-smi` showed 0 % / 1 MiB before every run |
| Container | `hongccc/sglang-omni:dev` |
| torch | 2.13.0+cu130 (CUDA build 13.0, cuDNN 92000) |
| sglang | 0.5.18 |
| sglang-omni | 0.1.4 |
| transformers | 5.12.1 |
| Repo commit | `024d099bba15cc2023a55dca42936f118a512252` (main, clean) |
| Dependency freeze | `223870b864d199d2d3413eb77107d94f5109f93ade5cde40406744a0fd4d63c7` |
| Model | `openai/whisper-large-v3` |
| Datasets | seedtts `27f4c1adee83b5b29b7c4b375f6b976324bda308` · longlibriheavy `09bc067255eeb0d0bca62357ac985c2ebdc5169c` · meanwhile `5a6b431a268523a6603f199d859fc25a24c22900` |

Environment was installed with the repo's own CI recipe
(`.github/scripts/prepare_omni_venv.sh`), which reuses the image's pinned wheels via
`uv venv --system-site-packages` plus `uv pip install --no-deps -e .`. It verified 19
exact dependency pins.

---

## Reproduce

```bash
# server, stock defaults
sgl-omni serve --model-path openai/whisper-large-v3 --port 8000

# Layer 3 — short-form concurrency sweep
python -u -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-large-v3 \
  --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup \
  --sample-util --util-gpu-ids 0 --util-interval 0.5 --fingerprint \
  --save-raw-dir raw/seedtts_en --output seedtts_en_sweep.json

# Layer 1 — GPU busy ratio (nvidia-smi sampling, method 2 in [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §2)
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader,nounits -lms 100 > util_c$C.csv &
python -u -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-large-v3 \
  --concurrencies $C --repeats 1 --warmup --output bench_c$C.json

# Layer 2 — stage breakdown via the built-in request profiler
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

---

## Layer 1 — GPU busy ratio

**Method:** `nvidia-smi` sampling at 100 ms across a warmed 1-repeat pass
([#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §2 Layer 1, method 2). `--sample-util` reports memory, power and host CPU but
**not** GPU utilization, so this had to be measured separately.

| conc | samples | util mean | util median | busy ratio (>0 %) | share >50 % | power mean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2245 | 48.4 % | 50.0 % | 96.8 % | 41.4 % | 141.5 W |
| 8 | 780 | 41.6 % | 44.0 % | 89.6 % | 25.6 % | 149.7 W |
| 32 | 521 | 41.8 % | 42.0 % | 86.4 % | 39.3 % | 163.1 W |
| 64 | 585 | 36.1 % | 41.0 % | 78.6 % | 23.1 % | 151.5 W |

The profiled passes report `gpu_util_percent` independently and agree: 48.1 / 47.5 /
36.7 / 35.3 %. Peak power was 192.5 W against a 450 W cap.

**Finding:** utilization is far from saturation at every level, and *falls* as concurrency
rises. [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) flags this exact shape as possibly host-bound rather than as headroom.
Layer 2 was run to separate the two — and rules host-bound out (below). **Evidence: strong**
for the measurement, and two independent methods agree.

---

## Layer 2 — stage breakdown

**Method:** the repo's own request-level profiler (`--profile-events`), not py-spy.
py-spy cannot run in a stock container: `yama.ptrace_scope = 1`, the sysctl is read-only,
and the default Docker capability set (`CapEff 00000000a80405fb`) has no `CAP_SYS_PTRACE`,
so it fails with `Permission denied (os error 13)` even as root. The built-in profiler goes
through `/start_request_profile` and needs no privileges.

Average per-stage duration, 200 samples per level:

| stage interval | c=1 | c=8 | c=32 | c=64 |
|---|---:|---:|---:|---:|
| `stage_input_received→stage_complete` (total) | 95.84 | 251.22 | 722.03 | 1357.41 |
| `scheduler_prefill_end→stage_complete` (decode) | 58.82 | 186.46 | **581.00** | **580.12** |
| `scheduler_prefill_start→scheduler_prefill_end` (prefill) | 30.72 | 32.08 | 32.85 | 33.55 |
| `scheduler_request_build_start→…_end` | 2.89 | 3.56 | 6.76 | 7.17 |
| `scheduler_queue_enter→scheduler_prefill_start` (queue wait) | 2.21 | 8.18 | 45.53 | **649.08** |
| `scheduler_request_build_end→scheduler_queue_enter` | 0.34 | 5.54 | 22.77 | 32.17 |

As a share of total:

| stage | c=1 | c=8 | c=32 | c=64 |
|---|---:|---:|---:|---:|
| decode | 61.4 % | 74.2 % | 80.5 % | 42.7 % |
| prefill | 32.1 % | 12.8 % | 4.5 % | 2.5 % |
| **queue wait** | 2.3 % | 3.3 % | 6.3 % | **47.8 %** |
| host-side (build + handoff) | 3.4 % | 3.6 % | 4.1 % | **2.9 %** |

**Findings:**

1. **Decode is already saturated at concurrency 32.** Decode time is 581.00 ms at c=32 and
   580.12 ms at c=64 — statistically identical. Adding concurrency past 32 does not add
   work to the batch. **Evidence: strong.**
2. **At c=64, 47.8 % of request latency is pure queue wait** (2.21 → 649.08 ms). This is
   exactly where the latency doubling in Layer 3 comes from. **Evidence: strong.**
3. **Prefill is constant** — 30.72 → 33.55 ms across a 64× concurrency range, because the
   encoder processes a fixed 30 s window. Its share collapses from 32.1 % to 2.5 %.
   **Evidence: strong.**
4. **Host-side orchestration is not the bottleneck — negative result, stated explicitly.**
   `request_build` plus the build→queue handoff account for 2.9 % at c=64, and their share
   *falls* as concurrency rises (3.4 % → 2.9 %). The dropping busy ratio in Layer 1 is
   therefore **not** the host-bound signature [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) warns about. **Evidence: strong.**

**What is left unexplained:** why utilization sits at 35–48 % when decode is the dominant
stage. The stage breakdown shows *where* time goes, not whether the decode kernels
themselves are launch-bound or bandwidth-bound. Settling that needs a kernel timeline
(nsys), which the same container restrictions blocked. **Evidence: weak — flagged, not
concluded.**

---

## Layer 3 — short-form concurrency sweep

SeedTTS EN, 1088 clips, 3 repeats plus one discarded warmup per level.
**3264/3264 requests completed at every level; zero shed, zero errors.**

| conc | req/s mean | req/s best | lat mean | lat p50 | lat p95 | RTFx | corpus WER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.74 | 9.86 | 0.102 s | 0.102 s | 0.124 s | 46.1 | 0.0137 |
| 2 | 15.76 | 15.82 | 0.127 s | 0.130 s | 0.161 s | 74.6 | 0.0137 |
| 4 | 23.00 | 23.31 | 0.174 s | 0.171 s | 0.233 s | 108.9 | 0.0138 |
| 8 | 31.32 | 31.61 | 0.255 s | 0.251 s | 0.346 s | 148.3 | 0.0138 |
| 16 | 40.37 | 40.67 | 0.395 s | 0.393 s | 0.532 s | 191.2 | 0.0138 |
| **32** | **47.68** | **48.02** | 0.668 s | 0.658 s | 0.923 s | **225.8** | 0.0138 |
| 64 | 47.15 | 47.73 | 1.339 s | 1.338 s | 1.626 s | 223.3 | 0.0138 |

Resource envelope:

| conc | GPU mem steady | power peak | system CPU peak |
|---:|---:|---:|---:|
| 1 | 22154 MiB | 153.4 W | 47.9 % |
| 8 | 22156 MiB | 172.9 W | 42.2 % |
| 32 | 22350 MiB | 192.3 W | 34.4 % |
| 64 | 22486 MiB | 192.5 W | 47.0 % |

**Findings:**

1. **Throughput saturates at concurrency 32** and does not scale to 64 (47.68 → 47.15)
   while mean latency doubles (0.668 → 1.339 s). **Evidence: strong.**
2. **Stock defaults fit a 24 GB card with room to spare.** `mem_fraction_static` defaults
   to 0.85 for Whisper and `max_running_requests` to 64; neither needed lowering. Steady
   memory grew only 332 MiB across a 64× concurrency range (22154 → 22486 MiB).
   **Evidence: strong.**
3. **No accuracy cost from concurrency** — corpus WER is 0.0137–0.0138 at every level.
   **Evidence: strong.**
4. For contrast, the H100 Qwen3-ASR profile sheds 1–4 % of requests at concurrency 64 when
   `request_build_max_pending` overflows. Whisper on a 4090 shows no shedding at any level.

---

## Layer 5 — functional regression on long-form audio

Also satisfies the long-form coverage in this issue's scope. Both datasets completed 100 %.

**longlibriheavy-60** (100 clips of 60 s audio):

| conc | req/s mean | RTFx | lat mean | lat p95 | corpus WER | completed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.324 | 145.8 | 0.430 s | 0.530 s | 0.1063 | 300/300 |
| **8** | **7.566** | **474.6** | 1.036 s | 1.370 s | 0.1062 | 300/300 |
| 32 | 7.029 | 440.8 | 4.173 s | 5.669 s | 0.1062 | 300/300 |

**meanwhile** (60 clips, structurally different monologue):

| conc | req/s mean | RTFx | lat mean | lat p95 | corpus WER | completed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.240 | 70.9 | 0.806 s | 0.979 s | 0.0987 | 180/180 |
| 8 | 4.542 | 259.8 | 1.692 s | 2.135 s | 0.0984 | 180/180 |
| **32** | **4.913** | **281.0** | 5.592 s | 7.768 s | 0.0985 | 180/180 |

**Findings:**

1. **No accuracy regression under concurrency.** WER is flat on both sets
   (0.1062–0.1063 and 0.0984–0.0987). The higher absolute WER versus short-form is dataset
   difficulty, not a regression. **Evidence: strong.**
2. **The saturation point moves with workload shape — three shapes, three behaviours.**
   **Evidence: strong.**

   | dataset | saturates at | behaviour at conc 32 |
   |---|---|---|
   | SeedTTS short-form | **32** | flat (47.68 → 47.15) |
   | longlibriheavy-60 | **8** | **regresses** (7.57 → 7.03), latency 4× |
   | meanwhile | still climbing | slight gain (4.54 → 4.91) |

   [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) warns that the trend can flip entirely between workload shapes. This run confirms
   it: a "concurrency 32 is optimal" conclusion drawn from short-form alone would
   noticeably overload longlibriheavy-60.
3. **RTFx is higher on long-form** (474.6 vs 225.8) because longer audio amortizes the
   fixed per-request cost — consistent with the constant ~33 ms prefill in Layer 2.

---

## Layer 4 — not run, and why

[#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §3 item 6 conditions Layer 4 on "each suspected overhead source found in Layer 2".
Layer 2 found none: host-side stages sit at 2.9 % and their share falls with concurrency.
There is no single-variable knob to A/B without first establishing *why* decode leaves the
GPU at 35–48 %, which needs the kernel timeline this environment could not capture.

**This is a deliberate skip under the methodology's own condition, not an omission.**

---

## Findings, evidence, recommendation

| # | Finding | Evidence | Recommendation | Strength |
|---|---|---|---|---|
| 1 | Stock defaults run unchanged on 24 GB; steady memory 22154→22486 MiB across a 64× concurrency range | Layer 3 sweep, 3264/3264 at every level | Document 24 GB as a supported Whisper configuration; no consumer-GPU profile file appears necessary | **Strong** |
| 2 | Short-form throughput saturates at concurrency 32; latency doubles to 64 with no throughput gain | Layer 3 | Treat 32 as the short-form operating point | **Strong** |
| 3 | Decode batch is already full at 32 — decode time identical at 32 and 64 (581.00 / 580.12 ms) | Layer 2 stage breakdown | Admission above 32 buys nothing on this hardware for short-form | **Strong** |
| 4 | At concurrency 64, 47.8 % of latency is queue wait | Layer 2 | Consider capping `max_running_requests` near the decode batch limit on 24 GB cards rather than the default 64 | **Strong** |
| 5 | Host-side orchestration is *not* the bottleneck (2.9 % at c=64, share falling) | Layer 2 | No CPU-side work indicated; negative result | **Strong** |
| 6 | Saturation point is workload-shape dependent — long-form 60 s saturates at 8 and regresses at 32 | Layer 5, two datasets | Any published concurrency guidance for Whisper must state the audio-length regime it applies to | **Strong** |
| 7 | No accuracy cost from concurrency on any of the three datasets | Layers 3 and 5 | None needed | **Strong** |
| 8 | GPU utilization stays at 35–48 % with decode dominant; cause unresolved | Layer 1 two methods; Layer 2 rules out host-bound | Needs a kernel timeline (nsys) on an environment with the required capabilities before any kernel-level work is justified | **Weak** |

---

## Cost and operational notes

The whole run took **2.2 GPU-hours on a rented RTX 4090 (~$0.83)**. Timings, in case they
help whoever picks up the next model:

| step | wall time |
|---|---|
| container image pull | 5 min |
| environment via `prepare_omni_venv.sh` | 30 s |
| model + 3 datasets (38 GB HF cache) | 3.5 min |
| **server cold start** | **~10 min** |
| Layer 3 full sweep | 35 min |
| Layer 1 + Layer 2 + Layer 5 | ~25 min |

Cold start is dominated by **118 rounds of Inductor autotune** — worth knowing before
planning a segmented rental, since a stop/start pays it again.

---

## Two items for [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §1/§2 (per §3 item 9)

Both are general, not Whisper-specific:

1. **py-spy needs `CAP_SYS_PTRACE`, which stock containers do not grant.** §4's tools index
   gives the py-spy command with no precondition. On a standard Docker or rented-cloud
   container it fails with `Permission denied (os error 13)` even as root, because
   `yama.ptrace_scope = 1` and the sysctl is read-only. Worth noting the
   `--cap-add SYS_PTRACE` requirement, and that `--profile-events` is the privilege-free
   alternative for stage-level attribution.
2. **`--sample-util` does not report GPU utilization.** Its `resources` block carries
   memory, power and host CPU only, so Layer 1's busy ratio needs a separate `nvidia-smi`
   sampler. Easy to assume otherwise from the flag name.

Happy to open a PR for either if useful.
