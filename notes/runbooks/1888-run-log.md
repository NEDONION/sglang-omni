# [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 执行日志 · 第一次运行

> 配套方案见 [1888-whisper-profiling.md](./1888-whisper-profiling.md)。
> 这份是**实际跑的时候发生了什么**——含每一步的真实耗时、踩的坑、和当场量到的数字。

| | |
| --- | --- |
| **日期** | 2026-09-02 |
| **实例** | Vast.ai 合约 `49621079`，offer `48176421`，Arizona US |
| **硬件** | RTX 4090 24564 MiB · 驱动 595.84 · 128 核 · 503 GiB RAM · **独占，无 co-tenant** |
| **镜像** | `hongccc/sglang-omni:dev` |
| **仓库** | `024d099b`（当天最新 main），工作区干净 |
| **软件栈** | torch 2.13.0+cu130 · CUDA 13.0 · transformers 5.12.1 · sglang 0.5.18 · sglang-omni 0.1.4 |
| **单价** | $0.375/hr + 存储 $0.20/GB月 × 120G |

---

## 时间线与真实耗时

| 阶段 | 耗时 | 备注 |
| --- | --- | --- |
| 创建实例 → 镜像拉完 | **5 min** | `loading` 到 `running` |
| SSH 密钥排障 | **12 min** | ⚠️ 见坑 1、坑 2 |
| 装环境（CI 脚本） | **0.5 min** | 镜像里依赖齐全，只装本体 |
| 下模型 + 3 个数据集 | **3.5 min** | HF 缓存共 **38 G** |
| 服务冷启动 | **~10 min** | ⚠️ 见坑 3 |
| 冒烟测试（20 条） | **1 min** | |
| Phase 3 全量扫描 | 进行中 | |

---

## 坑

### 坑 1 · `vastai create ssh-key` 传路径不报错

官方 SKILL.md 的示例 `vastai create ssh-key ~/.ssh/id_ed25519.pub` 会把**文件路径字符串**当成密钥存进账户，API 不校验，还返回 `success: True`。必须写：

```bash
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
```

验证 `public_key` 字段以 `ssh-ed25519 AAAA` 开头。存错了的后果是实例 `authorized_keys` 里是一串垃圾，SSH 连不上。

### 坑 2 · 私钥有 passphrase 时 agent 连不上

本机的 `id_ed25519` 是 2022 年生成的、带 passphrase。非交互环境弹不出输入框，表现为服务端 `Server accepts key` 但随即 `Permission denied`——**服务端认公钥，客户端签不了名**。

解法是给租用机器单独生成一把无密码的：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_vast -N "" -C "vast-gpu"
vastai create ssh-key "$(cat ~/.ssh/id_vast.pub)"
vastai attach ssh <INSTANCE_ID> "$(cat ~/.ssh/id_vast.pub)"
```

ssh config 里要同时写 `IdentityFile ~/.ssh/id_vast` 和 `IdentitiesOnly yes`，否则客户端还是会优先试那把带密码的。

### 坑 3 · 冷启动 10 分钟，几乎全是 Inductor autotune

服务日志里 **118 轮 `AUTOTUNE`**，逐个 benchmark matmul kernel 变体（`triton_mm_1220` 这种编号说明试了上千个），每轮约 1.3 秒。这是 torch.compile 的冷启动代价。

**影响分段策略**：停机再开又要付一次这 10 分钟，所以尽量一次做完。

### 坑 4 · 不加 `--warmup` 首轮数据完全失真

冒烟测试没加 `--warmup`，conc=2 的三轮：

```
rep=1  wall=39.821s   ← 首轮
rep=2  wall= 1.181s   ← 快 34 倍
rep=3  wall= 1.171s
```

汇总表被首轮污染（`wall mean 14.058`、`lat p95 12.789` 全是假的）。**正式扫描必须加 `--warmup`。**

---

## 与预判不符的地方

**我预判「`mem_fraction_static: 0.85` 是照 H200 调的，24G 上要往下降」——实际不用。**

默认参数直接跑通：显存稳定 **21.4 G / 24.5 G**，60/60 请求全完成，无 OOM，不需要降 `mem_fraction_static` 或 `max_running_requests`。

「**默认配置在消费级 24G 卡上开箱可用**」本身就是报告里的一条结论——Whisper 现有基准全部来自 H200，这一档此前没有公开数据。

---

## 冒烟测试结果（20 条 SeedTTS EN，未预热）

```
[conc=2  rep=1] wall=39.821s  thrpt= 0.502/s  lat_mean=3.982s  wer=0.0069  20/20
[conc=2  rep=2] wall= 1.181s  thrpt=16.929/s  lat_mean=0.118s  wer=0.0069  20/20
[conc=2  rep=3] wall= 1.171s  thrpt=17.072/s  lat_mean=0.117s  wer=0.0069  20/20
[conc=32 rep=1] wall= 0.623s  thrpt=32.123/s  lat_mean=0.563s  wer=0.0069  20/20
[conc=32 rep=2] wall= 0.672s  thrpt=29.748/s  lat_mean=0.618s  wer=0.0069  20/20
[conc=32 rep=3] wall= 0.479s  thrpt=41.723/s  lat_mean=0.425s  wer=0.0069  20/20
```

**WER 0.0069 在全部 6 轮完全一致**，输出稳定。

---

## Layer 1 早期信号（全量扫描进行中）

扫描期间 GPU 利用率采样：

```
46% / 49% / 46% / 48% / 50%   显存 21674 MiB
```

**利用率稳定在 ~50%，远低于饱和。** 按方法论 Layer 1 的判定规则，这指向 **CPU / 调度瓶颈**而非 GPU 算力瓶颈，下一步应做 Layer 2 的 py-spy 采样。

⚠️ 但这只是粗采样，正式结论要等 `--sample-util` 产出的完整数据。而且要注意方法论的提醒：利用率随并发的趋势可能在不同 workload 形状间翻转，短音频的结论不能直接套到长音频。

---

## Phase 3 · 全量并发扫描（Layer 3）

命令：

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-large-v3 \
  --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup \
  --sample-util --util-gpu-ids 0 --util-interval 0.5 --fingerprint \
  --save-raw-dir $RUN/raw/seedtts_en --output $RUN/seedtts_en_sweep.json
```

耗时约 **35 分钟**。SeedTTS EN 1088 条 × 3 重复 × 7 档，每档另有 1 轮丢弃的 warmup。
**全部 7 档 3264/3264 请求完成，零错误、零丢弃。**

| conc | req/s mean | req/s best | lat mean | lat p50 | lat p95 | RTFx | corpus WER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.74 | 9.86 | 0.102s | 0.102s | 0.124s | 46.1 | 0.0137 |
| 2 | 15.76 | 15.82 | 0.127s | 0.130s | 0.161s | 74.6 | 0.0137 |
| 4 | 23.00 | 23.31 | 0.174s | 0.171s | 0.233s | 108.9 | 0.0138 |
| 8 | 31.32 | 31.61 | 0.255s | 0.251s | 0.346s | 148.3 | 0.0138 |
| 16 | 40.37 | 40.67 | 0.395s | 0.393s | 0.532s | 191.2 | 0.0138 |
| **32** | **47.68** | **48.02** | 0.668s | 0.658s | 0.923s | **225.8** | 0.0138 |
| 64 | 47.15 | 47.73 | 1.339s | 1.338s | 1.626s | 223.3 | 0.0138 |

**读法：**

- **吞吐在并发 32 饱和**，到 64 不再增长（47.68 → 47.15，反而略降），而平均延迟**翻倍**（0.668 → 1.339 s）。典型饱和拐点。
- **WER 全程 0.0137–0.0138**，7 档之间无差异 —— 加并发不损精度。
- **各档 3264/3264 全部完成**。作为对照，H100 上的 Qwen3-ASR 在并发 64 会因 `request_build_max_pending` 溢出丢 1–4% 请求（见 `qwen3_asr_concurrency_profile.md`）；Whisper 在 4090 上没有这个现象。

资源侧（来自 `--sample-util`）：

| conc | GPU mem 稳态 MiB | 功耗峰值 W | 系统 CPU 峰值 % |
|---:|---:|---:|---:|
| 1 | 22154 | 153.4 | 47.9 |
| 8 | 22156 | 172.9 | 42.2 |
| 32 | 22350 | 192.3 | 34.4 |
| 64 | 22486 | 192.5 | 47.0 |

显存从 22154 → 22486 MiB，**并发翻 64 倍只多用 332 MiB**。功耗峰值 192.5 W，而 RTX 4090 TDP 约 450 W —— **GPU 远未吃满**。

---

## Layer 1 · GPU 忙碌比

⚠️ `--sample-util` 产出的 `resources` 块只有显存 / 功耗 / CPU 百分比，**没有 GPU 利用率**。
Layer 1 的忙碌比必须另测。方法论 §2 Layer 1 的「方法 2：nvidia-smi 采样」：

```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader,nounits -lms 100 > util_c$C.csv &
python -u -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --concurrencies $C --repeats 1 --warmup ...
```

100 ms 采样，服务已预热：

| conc | 样本数 | util 均值 | util 中位 | **忙碌比 (>0)** | >50% 占比 | 功耗均值 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2245 | 48.4% | 50.0% | **96.8%** | 41.4% | 141.5 W |
| 8 | 780 | 41.6% | 44.0% | **89.6%** | 25.6% | 149.7 W |
| 32 | 521 | 41.8% | 42.0% | **86.4%** | 39.3% | 163.1 W |
| 64 | 585 | 36.1% | 41.0% | **78.6%** | 23.1% | 151.5 W |

**结论：GPU 利用率随并发上升反而下降**（48.4% → 36.1%），忙碌比同步下滑（96.8% → 78.6%）。

按方法论 Layer 1 的判定规则，利用率**远低于饱和** → 瓶颈在 **CPU / 调度**，而非 GPU 算力。

⚠️ 但方法论 Layer 3 一节专门预警过这个形态：

> 当忙碌比随并发**下降**（而不是上升）、且波动较大时，容易被误读成「并发还不够高」，但它也可能是 host-bound 的症状 —— CPU 侧调度/同步开销随并发变重，在 GPU kernel 时间线上撕出更多、更长的空隙，把忙碌比拖下来。

区分方法是在各并发点抓 nsys trace（`--trace` 带 `osrt`），看 host 侧同步/等待系统调用（`pthread_cond_wait` 等）的频率是否随并发同步上升。**这是佐证不是证明。**

---

## 坑 5 · py-spy 在租用容器里跑不了（Layer 2 受阻）

```
yama.ptrace_scope = 1                  ← 限制模式
/proc/sys/.../ptrace_scope 只读         ← 容器无权修改
CapEff 00000000a80405fb                ← Docker 默认能力集，不含 CAP_SYS_PTRACE
py-spy dump --pid <server>
  → Error: Failed to copy Py_Version symbol
     Caused by: Permission denied (os error 13)
```

即使在容器里是 root 也不行 —— 缺的是 capability，不是 uid。

**这是通用坑，符合 [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §3 第 9 条「回填进方法论」的条件**：方法论 §4 工具索引里直接给了 py-spy 命令，但没提它在标准 Docker/云容器里默认不可用。

绕过办法：
1. 创建实例时加 `--cap-add SYS_PTRACE`（Vast 的 `create instance` 是否支持待验证）
2. **改用仓库自带的请求级 profiler** —— `--profile-events` 走 HTTP 端点
   （`/start_request_profile`、`/stop_request_profile`），配合 `sglang_omni.profiler.views`
   做阶段/跳数拆解，**不需要任何特权**

---

## 数据存放

| 位置 | 内容 |
| --- | --- |
| `~/sglang-omni-runs/20260902-0716/` | **全量 32 M**，含 21 个 `raw/*.jsonl`（每请求记录）和完整 serve.log |
| `notes/runbooks/1888-data/` | 小体积产物 272 K：扫描 JSON、环境指纹、Layer 1 汇总，随笔记一起版本管理 |
| 远程 `/workspace/runs/20260902-0716/` | 原始位置，实例销毁即消失 |

---

## Layer 2 · 阶段拆解（改用内置 profiler）

py-spy 走不通，改用仓库自带的请求级 profiler —— `--profile-events` 会在每档并发的正式轮次后
**额外跑一轮带事件记录的 pass**，产出 `stage_breakdown`。走 HTTP 端点，**不需要 ptrace 特权**。

```bash
python -u -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-large-v3 \
  --concurrencies 1,8,32,64 --repeats 1 --warmup --max-samples 200 \
  --profile-events --profile-urls http://127.0.0.1:8000 \
  --profile-event-dir $RUN/layer2/events --output $RUN/layer2/profiled.json
```

### 各阶段平均耗时 (ms)

| 阶段 | c=1 | c=8 | c=32 | c=64 |
| --- | ---: | ---: | ---: | ---: |
| **总计**（stage_input_received→stage_complete） | 95.84 | 251.22 | 722.03 | 1357.41 |
| decode（prefill_end→stage_complete） | 58.82 | 186.46 | **581.00** | **580.12** |
| prefill（prefill_start→prefill_end） | 30.72 | 32.08 | 32.85 | 33.55 |
| request_build | 2.89 | 3.56 | 6.76 | 7.17 |
| **排队等待**（queue_enter→prefill_start） | 2.21 | 8.18 | 45.53 | **649.08** |
| build→queue 交接 | 0.34 | 5.54 | 22.77 | 32.17 |

### 占「总计」比例 (%)

| 阶段 | c=1 | c=8 | c=32 | c=64 |
| --- | ---: | ---: | ---: | ---: |
| decode | 61.4 | 74.2 | 80.5 | 42.7 |
| prefill | 32.1 | 12.8 | 4.5 | 2.5 |
| request_build | 3.0 | 1.4 | 0.9 | 0.5 |
| **排队等待** | 2.3 | 3.3 | 6.3 | **47.8** |
| build→queue 交接 | 0.4 | 2.2 | 3.2 | 2.4 |

GPU 利用率（profiled pass 自测，与 nvidia-smi 采样吻合）：48.1 → 47.5 → 36.7 → 35.3 %

### 三条结论

**① decode 在并发 32 饱和。** `581.00` vs `580.12` ms —— c=32 与 c=64 的 decode 耗时几乎完全相同。
批次在 32 就满，再加并发不进入计算。**证据：强。**

**② 并发 64 时近一半延迟是纯排队。** 排队等待 45.53 → 649.08 ms，占比 6.3% → **47.8%**。
这解释了 Layer 3 的「吞吐持平、延迟翻倍」—— 多出的 32 个请求在队列里干等。**证据：强。**

**③ prefill 恒定不变。** 30.72 → 33.55 ms，并发翻 64 倍只涨 9%。
Whisper 的 encoder 处理固定 30 秒窗口，单请求成本与并发无关；其占比从 32.1% 塌到 2.5%。**证据：强。**

### ⚠️ 修正 Layer 1 的初步判定

Layer 1 看到利用率随并发下降，我据方法论初判为「CPU / 调度瓶颈」。**阶段拆解否定了这个假设：**

CPU 侧的 `request_build` + `build→queue 交接` 在 c=64 时合计只占 **2.9%**，而且**占比随并发是下降的**
（3.4% → 2.9%）。不存在「CPU 开销随并发变重」的现象。

真实机制是：请求在等 decode 槽位。GPU 利用率仅 35–48%，说明 Whisper large-v3 的 decode 步
在 batch ≤32 时喂不饱 4090 —— 更像 launch / 带宽受限而非算力受限。

**「为什么利用率低」这一条证据强度：弱。** 要 nsys kernel 时间线才能坐实，而 nsys 在本容器
同样受权限限制。报告里应如实标注，不要写成已证实的结论。

---

## Layer 5 + Scope ② · 长音频

两个长音频数据集，各自 3 档并发 × 3 重复 + warmup，全部 100% 完成。

### longlibriheavy-60（100 条 60 秒音频）

| conc | req/s mean | RTFx | lat mean | lat p95 | corpus WER | 完成 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.324 | 145.8 | 0.430 s | 0.530 s | 0.1063 | 300/300 |
| **8** | **7.566** | **474.6** | 1.036 s | 1.370 s | 0.1062 | 300/300 |
| 32 | 7.029 | 440.8 | 4.173 s | 5.669 s | 0.1062 | 300/300 |

### meanwhile（60 条，结构不同的独白）

| conc | req/s mean | RTFx | lat mean | lat p95 | corpus WER | 完成 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.240 | 70.9 | 0.806 s | 0.979 s | 0.0987 | 180/180 |
| 8 | 4.542 | 259.8 | 1.692 s | 2.135 s | 0.0984 | 180/180 |
| **32** | **4.913** | **281.0** | 5.592 s | 7.768 s | 0.0985 | 180/180 |

### 结论

**① Layer 5 通过。** 两个数据集的 corpus WER 在各并发档之间完全稳定
（longlibriheavy-60: 0.1062–0.1063；meanwhile: 0.0984–0.0987）。**加并发不损精度。**
注意长音频 WER（约 10%）远高于短音频（1.38%），这是数据集难度差异，不是回归。

**② 饱和点随 workload 形状移动 —— 三种形状三种行为。证据：强。**

| 数据集 | 饱和点 | 并发 32 时 |
| --- | --- | --- |
| SeedTTS 短音频 | conc **32** | 持平（47.68 → 47.15） |
| longlibriheavy-60 | conc **8** | **倒退**（7.57 → 7.03），延迟 4× |
| meanwhile | 仍在缓慢上升 | 微增（4.54 → 4.91） |

方法论 Layer 3 一节写着「利用率随并发的趋势可能在不同 workload 形状之间完全翻转 ——
必须按形状分别测，不要用一种形状的结论覆盖另一种」。**本次实测坐实了这条警告**：
如果只测短音频就下「并发 32 是最优」的结论，在 longlibriheavy-60 上会明显过载。

**③ RTFx 长音频反而更高**（474.6 vs 短音频 225.8）。长音频摊薄了每请求的固定开销
（prefill 恒定 ~33 ms、request_build、排队），符合 Layer 2 的拆解。

---

## 坑 6 · 长音频 benchmark 的 `--model-path` 默认值是别的模型

`benchmark_asr_longform` 不传 `--model-path` 时，日志打印的是
`against 127.0.0.1:8000 (Qwen/Qwen3-ASR-1.7B)`。请求大概率仍打到实际服务的 Whisper，
但**结果 JSON 里记录的模型名是错的**，数据溯源作废。第一次跑漏了这个参数，发现后重跑。

**每次都要显式传 `--model-path`，并核对日志首行打印的模型名。**

## 坑 7 · 不放 tmux 的命令会随 SSH 断连一起丢

`meanwhile` 那轮我直接在 SSH 上跑，连接被远端断开。这次侥幸：进程作为孤儿存活，
且 `tee` 一直在写日志文件，数据没丢。但这是运气，不是设计。

**手册自己写了「长任务一律进 tmux」，执行时我违反了自己的规矩。**

---

## 收尾

**实例已销毁**（`vastai destroy instance 49621079`），计费停止。
销毁前核对：远程 62 个文件 = 本地 62 个文件，数据完整。

**总账单：2.2 GPU-小时 ≈ $0.83。**

### 五层完成情况

| Layer | 状态 | 说明 |
| --- | --- | --- |
| 1 GPU 忙碌比 | ✅ | nvidia-smi 100 ms 采样，两种方法互相印证 |
| 2 阶段拆解 | ✅ | 内置 profiler 替代 py-spy |
| 3 并发扫描 | ✅ | 短音频 7 档 × 3264 请求全完成 |
| 4 A/B 验证 | ⏭️ **按方法论条件跳过** | Layer 2 未发现可疑开销源，无可 A/B 的候选 |
| 5 功能回归 | ✅ | 两个长音频数据集，WER 全程稳定 |

Layer 4 的跳过依据是 [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §3 第 6 条——它的前提是「对 Layer 2 找到的每个可疑开销源」。
**报告里必须写明这是有条件的主动跳过，不是漏做。**

### 报告草稿

[1888-report-draft.md](./1888-report-draft.md)，英文，293 行。结构按 §3 第 8 条：
按层组织、每层标注证据强度、负面结论明确写出、末尾附发现—证据—建议表。

**尚未发布。** 而且还没在 issue 下认领。

### 回填方法论的两条（§3 第 9 条）

1. py-spy 需要 `CAP_SYS_PTRACE`，标准容器不给 —— §4 工具索引给了命令但没写前提
2. `--sample-util` 不产出 GPU 利用率 —— 从参数名容易误以为有

<!-- 后续进展往下追加 -->
