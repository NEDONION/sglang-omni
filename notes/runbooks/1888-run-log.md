# #1888 执行日志 · 第一次运行

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

**这是通用坑，符合 #1798 §3 第 9 条「回填进方法论」的条件**：方法论 §4 工具索引里直接给了 py-spy 命令，但没提它在标准 Docker/云容器里默认不可用。

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

<!-- 后续进展往下追加 -->
