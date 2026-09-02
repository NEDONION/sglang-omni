First-pass Layer 1–5 profile, run on a **single RTX 4090 (24 GB)** — the gap this fills is
that every Whisper number currently in `docs/cookbook/whisper_asr.md` was measured on an H200.

**Headline**

- **Stock defaults run unchanged on 24 GB.** No `mem_fraction_static` or `max_running_requests`
  tuning needed; steady memory grew 332 MiB across a 64× concurrency range. 3264/3264 requests
  completed at every level, zero shed.
- **Short-form saturates at concurrency 32**; going to 64 adds no throughput (47.68 → 47.15 req/s)
  and doubles latency (0.668 → 1.339 s).
- **The saturation point is workload-shape dependent** — long-form 60 s audio saturates at **8**
  and *regresses* at 32. A conclusion drawn from short-form alone would overload it.
- **Host-side orchestration is not the bottleneck** (2.9 % of latency at c=64, share *falling*
  with concurrency). The dropping busy ratio is not the host-bound signature #1798 warns about.

---

### Layer 3 — short-form (SeedTTS EN, 1088 clips, 3 repeats + warmup)

| conc | req/s | lat mean | lat p95 | RTFx | WER | completed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.74 | 0.102 s | 0.124 s | 46.1 | 0.0137 | 3264/3264 |
| 8 | 31.32 | 0.255 s | 0.346 s | 148.3 | 0.0138 | 3264/3264 |
| 16 | 40.37 | 0.395 s | 0.532 s | 191.2 | 0.0138 | 3264/3264 |
| **32** | **47.68** | 0.668 s | 0.923 s | **225.8** | 0.0138 | 3264/3264 |
| 64 | 47.15 | 1.339 s | 1.626 s | 223.3 | 0.0138 | 3264/3264 |

<details><summary>all 7 levels + resource envelope</summary>

| conc | req/s mean | req/s best | lat mean | lat p50 | lat p95 | RTFx | WER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.74 | 9.86 | 0.102 s | 0.102 s | 0.124 s | 46.1 | 0.0137 |
| 2 | 15.76 | 15.82 | 0.127 s | 0.130 s | 0.161 s | 74.6 | 0.0137 |
| 4 | 23.00 | 23.31 | 0.174 s | 0.171 s | 0.233 s | 108.9 | 0.0138 |
| 8 | 31.32 | 31.61 | 0.255 s | 0.251 s | 0.346 s | 148.3 | 0.0138 |
| 16 | 40.37 | 40.67 | 0.395 s | 0.393 s | 0.532 s | 191.2 | 0.0138 |
| 32 | 47.68 | 48.02 | 0.668 s | 0.658 s | 0.923 s | 225.8 | 0.0138 |
| 64 | 47.15 | 47.73 | 1.339 s | 1.338 s | 1.626 s | 223.3 | 0.0138 |

| conc | GPU mem steady | power peak | system CPU peak |
|---:|---:|---:|---:|
| 1 | 22154 MiB | 153.4 W | 47.9 % |
| 8 | 22156 MiB | 172.9 W | 42.2 % |
| 32 | 22350 MiB | 192.3 W | 34.4 % |
| 64 | 22486 MiB | 192.5 W | 47.0 % |

Peak power 192.5 W against a 450 W cap.

</details>

### Layer 1 — GPU busy ratio

`nvidia-smi` sampled at 100 ms over a warmed pass (§2 method 2). Note `--sample-util` reports
memory, power and host CPU but **not** GPU utilization, so this needed a separate sampler.

| conc | samples | util mean | busy ratio (>0 %) | share >50 % |
|---:|---:|---:|---:|---:|
| 1 | 2245 | 48.4 % | 96.8 % | 41.4 % |
| 8 | 780 | 41.6 % | 89.6 % | 25.6 % |
| 32 | 521 | 41.8 % | 86.4 % | 39.3 % |
| 64 | 585 | 36.1 % | 78.6 % | 23.1 % |

The profiled passes report `gpu_util_percent` independently and agree (48.1 / 47.5 / 36.7 / 35.3 %).
Utilization is far from saturation everywhere and *falls* with concurrency. Layer 2 was run to
separate "headroom" from "host-bound".

### Layer 2 — stage breakdown

Used the repo's own request profiler (`--profile-events`), not py-spy — see the note at the bottom.
Average ms per stage, 200 samples per level:

| stage | c=1 | c=8 | c=32 | c=64 |
|---|---:|---:|---:|---:|
| total (`stage_input_received→stage_complete`) | 95.84 | 251.22 | 722.03 | 1357.41 |
| decode (`prefill_end→stage_complete`) | 58.82 | 186.46 | **581.00** | **580.12** |
| prefill (`prefill_start→prefill_end`) | 30.72 | 32.08 | 32.85 | 33.55 |
| queue wait (`queue_enter→prefill_start`) | 2.21 | 8.18 | 45.53 | **649.08** |
| request_build | 2.89 | 3.56 | 6.76 | 7.17 |
| build→queue handoff | 0.34 | 5.54 | 22.77 | 32.17 |

As a share of total, queue wait goes **2.3 % → 47.8 %**, while host-side stages
(build + handoff) stay at **3.4 % → 2.9 %**.

1. **Decode is already saturated at 32** — 581.00 ms at c=32 vs 580.12 ms at c=64, statistically
   identical. Concurrency past 32 adds no work to the batch.
2. **At c=64, 47.8 % of latency is queue wait.** That is where the Layer 3 latency doubling comes from.
3. **Prefill is constant** (30.72 → 33.55 ms) because the encoder processes a fixed 30 s window;
   its share collapses 32.1 % → 2.5 %.
4. **Negative result, stated explicitly:** host-side orchestration is *not* the bottleneck.

### Layer 5 — functional regression on long-form

| dataset | conc | req/s | RTFx | lat mean | WER | completed |
|---|---:|---:|---:|---:|---:|---:|
| longlibriheavy-60 | 1 | 2.324 | 145.8 | 0.430 s | 0.1063 | 300/300 |
| longlibriheavy-60 | **8** | **7.566** | **474.6** | 1.036 s | 0.1062 | 300/300 |
| longlibriheavy-60 | 32 | 7.029 | 440.8 | 4.173 s | 0.1062 | 300/300 |
| meanwhile | 1 | 1.240 | 70.9 | 0.806 s | 0.0987 | 180/180 |
| meanwhile | 8 | 4.542 | 259.8 | 1.692 s | 0.0984 | 180/180 |
| meanwhile | **32** | **4.913** | **281.0** | 5.592 s | 0.0985 | 180/180 |

WER is flat within each dataset — no accuracy cost from concurrency. Three workload shapes,
three behaviours: short-form saturates at 32, longlibriheavy-60 at **8** (and regresses at 32
with 4× latency), meanwhile is still climbing at 32. This is the flip #1798 warns about,
observed directly.

### Layer 4 — not run

§3 item 6 conditions Layer 4 on "each suspected overhead source found in Layer 2". Layer 2 found
none — host-side stages sit at 2.9 % and their share falls with concurrency. There is no
single-variable knob to A/B without first establishing *why* decode leaves the GPU at 35–48 %.
**A deliberate skip under the methodology's own precondition, not an omission.**

---

### Findings

| # | Finding | Recommendation | Strength |
|---|---|---|---|
| 1 | Stock defaults run unchanged on 24 GB; steady memory 22154→22486 MiB across 64× concurrency | Document 24 GB as a supported Whisper configuration | **Strong** |
| 2 | Short-form throughput saturates at 32; latency doubles to 64 with no gain | Treat 32 as the short-form operating point | **Strong** |
| 3 | Decode batch already full at 32 (581.00 / 580.12 ms) | Admission above 32 buys nothing here for short-form | **Strong** |
| 4 | 47.8 % of c=64 latency is queue wait | Worth A/B-ing `max_running_requests` nearer the decode batch limit on 24 GB | **Strong** |
| 5 | Host-side orchestration is not the bottleneck (2.9 %, share falling) | None — negative result | **Strong** |
| 6 | Saturation point is workload-shape dependent (long-form saturates at 8) | Any published concurrency guidance must state the audio-length regime | **Strong** |
| 7 | No accuracy cost from concurrency on any of three datasets | None needed | **Strong** |
| 8 | GPU utilization stays 35–48 % with decode dominant; cause unresolved | Needs a kernel timeline before kernel-level work is justified | **Weak** |

On #8, one code-level narrowing: with `enable_pre_lm_encoder` on (the default),
`_resolve_encoder_graph_buckets` sets `capture_limit = pre_lm_max_batch_size` (8), and
`pre_lm_max_batch_size` also caps the encoder batch itself — so the encoder path is
graph-covered across its whole range and is **not** where the gap lives. That points at decode.
*Code-level evidence only, not re-verified at runtime.*

---

<details><summary>Environment and reproduction</summary>

| | |
|---|---|
| GPU | RTX 4090, 24564 MiB, SM 8.9, driver 595.84, 450 W cap |
| Host | 128 cores, 503 GiB RAM |
| Isolation | **Dedicated single-tenant instance**; `nvidia-smi` showed 0 % / 1 MiB before every run, so no CPU pinning was needed |
| Container | `hongccc/sglang-omni:dev` |
| torch / sglang / sglang-omni / transformers | 2.13.0+cu130 (CUDA 13.0) / 0.5.18 / 0.1.4 / 5.12.1 |
| Repo commit | `024d099bba15cc2023a55dca42936f118a512252` (main, clean) |
| Dependency freeze | `223870b864d199d2d3413eb77107d94f5109f93ade5cde40406744a0fd4d63c7` |
| Datasets | seedtts `27f4c1ad…` · longlibriheavy `09bc0672…` · meanwhile `5a6b431a…` |

Environment built with the repo's own CI recipe (`.github/scripts/prepare_omni_venv.sh`),
which reuses the image's pinned wheels via `uv venv --system-site-packages` plus
`uv pip install --no-deps -e .`; it verified 19 exact dependency pins.

```bash
# server, stock defaults
sgl-omni serve --model-path openai/whisper-large-v3 --port 8000

# Layer 3
python -u -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-large-v3 \
  --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup \
  --sample-util --util-gpu-ids 0 --util-interval 0.5 --fingerprint \
  --save-raw-dir raw/seedtts_en --output seedtts_en_sweep.json

# Layer 1 (separate sampler, per level)
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

Whole run: **2.2 GPU-hours**. Cold start alone is ~10 min, dominated by **118 rounds of Inductor
autotune** — worth knowing before planning a segmented rental, since a stop/start pays it again.

</details>

### Raw artifacts

Attached to this comment: the four summary JSONs (`seedtts_en_sweep.json`, `layer2/profiled.json`,
`longform60.json`, `meanwhile.json`) and the Layer 1 `nvidia-smi` CSVs.

Per #1798's Result-tracking rule, bulky artifacts stay local and are named rather than attached:
the 21 per-request JSONL files under `raw/seedtts_en/` and `raw/longform60/`, and the 78 MB
`serve.log` (mostly the 118 Inductor autotune rounds). They lived under `/workspace/runs/20260902-0716/`
on the rented host, which has since been destroyed; a copy is retained locally. **No conclusion above
rests on an artifact that is not attached here** — every number in the tables comes from the attached
summary JSONs.

This comment is the durable record for this run: #1888 is already a sub-issue of #1798, so no
separate tracking sub-issue was filed.

---

<details><summary>Two notes for #1798 §1/§2 (per §3 item 9) — both general, not Whisper-specific</summary>

1. **py-spy needs `CAP_SYS_PTRACE`, which stock containers do not grant.** §4's tools index gives
   the py-spy command with no precondition. On a standard Docker or rented-cloud container it
   fails with `Permission denied (os error 13)` even as root, because `yama.ptrace_scope = 1` and
   the sysctl is read-only. Worth noting the `--cap-add SYS_PTRACE` requirement, and that
   `--profile-events` is the privilege-free alternative for stage-level attribution.
2. **`--sample-util` does not report GPU utilization.** Its `resources` block carries memory,
   power and host CPU only, so Layer 1's busy ratio needs a separate `nvidia-smi` sampler. Easy
   to assume otherwise from the flag name.

Happy to open a PR for either if useful.

</details>
