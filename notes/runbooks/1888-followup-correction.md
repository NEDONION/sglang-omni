Two corrections to the report above, plus one piece of evidence it was missing. Both came from
re-reading the retained server log rather than from a new run.

### Correction 1 — cold start is dominated by CUDA graph capture, not autotune

I wrote that the ~10 minute cold start is "dominated by 118 rounds of Inductor autotune". That
mis-attributes it. The log gives an exact breakdown:

```text
07:21:42.430  Init torch distributed begin
07:21:42.507  Init torch distributed end            elapsed=0.08 s
07:21:42.510  Load weight begin
07:21:44.977  Load weight end                       elapsed=2.39 s
07:21:45.259  Capture target decode CUDA graph begin
07:31:07.958  Capture target decode CUDA graph end  elapsed=562.70 s
07:31:11.460  Process asr ready
```

**569.03 s total, of which decode CUDA graph capture is 562.70 s — 98.9 %.** Weight loading is
2.39 s; the model is only ~3 GB, so disk is not the constraint. Autotune is real but it runs
*inside* the capture, once per bucket, rather than being a separate phase alongside it.

Prefill CUDA graph was disabled in this run (`Disable prefill CUDA graph because ...`), so the
whole figure is decode-side.

The operational conclusion is unchanged and if anything stronger: a stop/start pays this again.

### Correction 2 — "decode batch already full at 32" overstated what was established

Finding 3 said the decode batch is "already full at concurrency 32", citing 581.00 ms at c=32
versus 580.12 ms at c=64. **The measurement stands.** The phrasing does not: it implies a known
capacity limit, and I had not established what sets it.

I then looked for one, and the first candidate was wrong in an instructive way. sglang defaults
the decode CUDA graph batch cap by GPU memory tier (`server_args.py`):

```python
elif gpu_mem < 35 * 1024:          # A10, 4090, 5090
    if self.tp_size < 4:
        decode_cuda_graph_config.max_bs = 24
elif gpu_mem < 90 * 1024:          # H100, A100
    if self.tp_size < 4:
        decode_cuda_graph_config.max_bs = 256
```

A 24 GB 4090 at tp=1 lands on 24, which is close enough to the observed knee to look like the
answer. **The runtime log says otherwise:**

```text
Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1,
bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64], avail mem=2.96 GB
```

Buckets go to 64. The tiered default never applied, because the omni side passes
`max_running_requests` (64) down as an explicit `cuda_graph_max_bs` override —
`server_args_builder.py` maps `cuda_graph_max_bs` to `cuda_graph_max_bs_decode`, and
`bootstrap.py` only reads the resolved value.

So: **decode CUDA graph bucket capacity is ruled out as the cause of the knee** — both 32 and 64
are inside the captured set. That is a real narrowing, and it is new information relative to the
report above.

What remains open is what does set it. Available memory after capture was 2.61 GB, so KV pool
capacity is a candidate, as is scheduler admission. Neither is tested.

Finding 3 should therefore read as three separate claims:

| claim | evidence |
|---|---|
| Throughput knee is at concurrency 32 | **Strong** — Layer 3 sweep |
| Decode CUDA graph bucket capacity is not the cause | **Strong** — captured buckets reach 64 |
| What actually caps it | **None** — untested |

### A third note for #1798 §1/§2

The two tooling notes in the report have a companion, and this one cost me a wrong conclusion:

**When settling a configuration question by code reading, read the whole resolution chain, not
just the defaulting layer.** §2 Layer 4 says a deterministic, boot-time-computed configuration
outcome can be settled by reading the code instead of booting a server. That is true, but the
value a server actually resolves to is the *last* writer in the chain, and an upper layer passing
an explicit override makes the lower layer's tiered default irrelevant. Reading only the default
gave a plausible-looking number (24) that matched the observation closely enough to be believable
and was still wrong.

The cheap guard: whatever the code says, grep the server log for the resolved value before
building an argument on it. sglang logs the captured bucket list at startup, which settles it in
one line.
