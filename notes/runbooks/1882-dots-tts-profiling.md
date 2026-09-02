# dots.tts 运行时画像 · 执行手册

| | |
| --- | --- |
| **目标 issue** | [#1882](https://github.com/sgl-project/sglang-omni/issues/1882) `[Runtime Profiling] Dots-TTS` |
| **方法论** | [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) Layer 1–5（canonical 版在 [#1850](https://github.com/sgl-project/sglang-omni/pull/1850) 的 `METHODOLOGY.md`，未合并） |
| **模型** | `dots-studio/dots.tts-mf`（Qwen2.5-1.5B backbone + 18 层 DiT + 24 层 PatchEncoder + 48 kHz AudioVAE） |
| **硬件** | **80G 级别单卡**（A100 80G / H100 / H200）——⚠️ **不能沿用 [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 的 4090 24G**，见第三节 |
| **交付物** | **一条 issue 评论**（不是 PR） |
| **预算** | A100 80G $1.15/hr × 5–7 h ≈ **$6–8**，含试错上限 $15 |
| **状态** | 🟡 准备中 |

> 依据：命令、参数、默认值、配置均取自 main 分支 `332f4fc6` 的实际文件，以及
> `dots-studio/dots.tts-mf` 在 HuggingFace 上的 `config.json`。仓库演进后以代码为准。

---

## 一、先明确「做完」是什么样

[#1882](https://github.com/sgl-project/sglang-omni/issues/1882) 的 Done-when 两条，展开成 [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §3 的九步，和 [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 完全同构：

| # | 做什么 | 必做 |
| --- | --- | --- |
| 1 | 记录基线环境：框架版本、GPU 型号、checkpoint、数据集 revision、**仓库 commit SHA** | ✅ |
| 2 | 选卡时同时看利用率和显存，全程盯着别被抢占 | ✅ |
| 3 | **Layer 1** — 判定瓶颈在 GPU 算力还是 CPU/调度 | ✅ |
| 4 | **Layer 2** — 阶段级归因，定位最高占比的调用点，重复 3 次 | ⚠️ 条件 |
| 5 | **Layer 3** — 并发扫描，产出利用率 vs 并发的表 | ✅ |
| 6 | **Layer 4** — 对 Layer 2 的发现做单变量 A/B | ⚠️ 条件 |
| 7 | **Layer 5** — 功能/质量回归 | ✅ |
| 8 | 写报告：按层组织，标注证据强度，附发现—证据—建议表 | ✅ |
| 9 | 把通用的坑回填进 [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §1/§2 | 加分 |

**但 [#1882](https://github.com/sgl-project/sglang-omni/issues/1882) 的 Scope 比 [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 多要了三样东西**，这是两个 issue 的实质差别：

| Scope 要求 | Whisper 那次 | dots.tts 这次 |
| --- | --- | --- |
| 短 / 长两种 workload 分开测 | 短音频 / 长音频（数据集切换） | **短文本 / 长文本**（输入侧，要自己造） |
| time to first audio | 未要求 | ✅ 必须，**要开流式** |
| RTF | 未要求 | ✅ benchmark 自带 |
| per-stage timing | Layer 2 顺带 | ✅ **显式列为交付项** |
| `benchmark_tts_serving.py` | 不适用 | ✅ 显式点名，用来看 serving 行为 |

### 决策树

```
Layer 1 ─┬─ GPU 忙碌比接近饱和 → 跳过 Layer 2、4，直接 Layer 5
         └─ 忙碌比远低于 100%  → Layer 2 ─┬─ 有发现 → Layer 4
                                          └─ 没发现 → 跳过 Layer 4
```

**负面结论同样有效。** 「当前默认值已经够好、无需改动」是合法产出，只要推理写清楚。

---

## 二、⚠️ 开机前必读：dots.tts 不是一块处女地

这是和 [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 最大的不同。Whisper 那次只有 H200 的 cookbook 数字；**dots.tts 已经被专门做过一轮 perf 攻坚**，四个 PR 已合并、四个还开着。不先读完就开机，大概率会花钱重新发现别人三个月前就写下来的结论。

| 文档 / PR | 内容 | 状态 |
| --- | --- | --- |
| `docs/cookbook/dots_tts.md` §Performance | **1×H100 上 SeedTTS EN 全量 1088 条、并发 1–32 的完整表**（吞吐 / 延迟 / RTF / audio_s/s / WER），commit `2b45073c` | 已合并，**基线** |
| [#1442](https://github.com/sgl-project/sglang-omni/pull/1442) `research_log.md` | **low-fruit 优化审计**：假设、隔离的 CUDA 微基准、留/弃决策、以及**主动放弃的候选** | 🟡 开着 |
| [#1434](https://github.com/sgl-project/sglang-omni/pull/1434) / [#1459](https://github.com/sgl-project/sglang-omni/pull/1459) | **共享 codec lock 的争用**：按调用点的 profiling + GPU span 计时 | 🟡 开着 |
| [#1445](https://github.com/sgl-project/sglang-omni/pull/1445) | 拆分 reference / vocoder 的 codec lock | 🟡 开着 |
| [#1448](https://github.com/sgl-project/sglang-omni/pull/1448) | 给流式 AudioVAE decode 上 CUDA graph | 🟡 开着 |
| [#1438](https://github.com/sgl-project/sglang-omni/pull/1438) [#1440](https://github.com/sgl-project/sglang-omni/pull/1440) [#1444](https://github.com/sgl-project/sglang-omni/pull/1444) [#1446](https://github.com/sgl-project/sglang-omni/pull/1446) | 批量 denormalize / 跳过 vocoder staging / 批量流式 VAE / slot-pool | ✅ 已合并 |

**读完之后，把我们的定位写清楚**，否则报告会和别人撞车：

- **codec lock 争用**已经有人在量了（[#1434](https://github.com/sgl-project/sglang-omni/pull/1434)/[#1459](https://github.com/sgl-project/sglang-omni/pull/1459)）。我们不重新发现它，我们**用端到端的 per-stage 数据给它定一个量级上界**——「在并发 N 下，`vocoder` 阶段占端到端时间的 X%」。这对他们的 PR 是有用的旁证，不是重复劳动。
- cookbook 的 H100 表只到**并发 32**，且**只有非流式、只有短文本、只有 EN**。我们要补的是：**并发 32 以上的拐点**、**流式 TTFA**、**长文本形状**、**per-stage 拆解**。这四样目前都没有公开数据。
- ⚠️ **[#1850](https://github.com/sgl-project/sglang-omni/pull/1850) 还没合并**，`.claude/skills/model-profiling/` 在 main 上**不存在**。issue 里说的「use the `model-profiling` skill」现在做不到。方法论以 [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) 正文为准，或者从 PR 分支上把 `METHODOLOGY.md` 拉下来看。在报告里写明用的是哪一版。

### 先占坑

**动手前去 [#1882](https://github.com/sgl-project/sglang-omni/issues/1882) 留言认领。** 该 issue 目前 0 评论、无 assignee，但 dots.tts 这条线上活跃的人很多（Hayden727 / luojiaxuan / buffett0323 都有开着的 perf PR），不留言直接开跑有撞车风险。

---

## 三、⚠️ 硬件：为什么 4090 24G 这次一定不行

**这是本手册最重要的一节。** [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 那次「默认配置在消费级 24G 卡上开箱可用」是个惊喜；dots.tts 这次**从代码就能算出来不可能**，别浪费一次开机去撞。

### 声学 tail 的显存是预先按满长度整块吃掉的

`examples/configs/dots_tts.yaml` 的默认布局是 `max_running_requests: 16` × `max_generate_length: 500`。cookbook 明确写了：

> Continuous batching eagerly allocates acoustic-tail state for every slot at the full generate length.

而且 **`mem_fraction_static: 0.20` 只管 SGLang backbone 的 KV cache，不覆盖声学 tail 的池子**（cookbook 原文）。也就是说这块显存是**额外的**。

### 按 `tail.py:estimate_acoustic_pool_bytes` 实算

代入 `dots.tts-mf` 的 `config.json`（DiT 18 层 / 16 头 / hidden 1024 → head_dim 64；PatchEncoder 24 层 / 16 头；`patch_size=4` → `unit_len = hidden_patch_size(1) + latent_patch_size(4) = 5`）和 `engine_builder.py` 的 `nfe = num_steps = 4`、`num_slots = max_running_requests`、`patch_capacity = max_generate_length + 1`：

| slots | max_generate_length | 池子总计 | **+15% headroom（准入实际要求的空闲显存）** |
| ---: | ---: | ---: | ---: |
| **16** | **500**（默认） | **19.67 GiB** | **22.62 GiB** |
| 16 | 300 | 11.83 GiB | 13.61 GiB |
| 16 | 200 | 7.92 GiB | 9.10 GiB |
| 8 | 500 | 9.84 GiB | 11.31 GiB |
| 8 | 300 | 5.92 GiB | 6.80 GiB |
| 4 | 500 | 4.92 GiB | 5.66 GiB |
| 2 | 500 | 2.46 GiB | 2.83 GiB |

默认档 19.67 GiB 里，**DiT KV 独占 11.03 GiB**（`2 × nfe × 18层 × 16槽 × 16头 × 2510token × 64 × 2B`），scratch 5.70 GiB，encoder KV 2.94 GiB。

**在 24G 卡上，光准入检查就要 22.62 GiB 空闲**，而此时权重（1.5B backbone bf16 ≈ 3.1 GiB + DiT/encoder/vocoder ≈ 1.5–2 GiB）和 `mem_fraction_static=0.20`（24 GiB × 0.20 ≈ 4.9 GiB 的 backbone KV）已经吃掉了。**必然报错**：

```
dots.tts acoustic-tail admission failed at startup: estimated pools need ...
```

> 📐 **推算的置信度**：DiT 那两项（11.03 + 2.76 = 13.79 GiB，占总量 70%）只依赖 `config.json` 和代码里的常量，**是硬的**。encoder 两项（≈5.9 GiB）依赖 `PatchEncoder.out_ds_rate`——这个值在上游 `dots_tts` 包里，本地没装、没能核实，按 `patch_size=4` 取的。即使这项估偏，**24G 放不下默认配置的结论不变**。开机后第一件事就是看服务启动日志里那行 `dots.tts acoustic pools (estimate): ... total=X GiB`，**用实测值替换本表**再写进报告。

### 选卡结论

| 方案 | 卡 | 能否跑默认配置 | Vast 报价（2026-09-03 快照） | 评价 |
| --- | --- | --- | --- | --- |
| **✅ 首选** | **A100 80G SXM4** | ✅ 16×500 | **$1.15/hr** | 唯一便宜的 80G。和 cookbook 的 H100 表**不同代**，比较时要说明 |
| 备选 | H100 PCIE 80G | ✅ | $3.07/hr（UAE） | **可直接对齐 cookbook 的 H100 表**，但贵 2.7 倍 |
| 备选 | H200 141G | ✅ | $3.98/hr（SA） | 更贵，显存用不完 |
| ⚠️ 降配 | RTX 5090 32G | ⚠️ 只能 8×500 或 16×300 | $0.32/hr | **不是 canonical 配置**，issue 明确要求用 `dots_tts.yaml` |
| ❌ | RTX 4090 24G | ❌ | $0.28/hr | 见上，准入必失败 |

**建议：A100 80G，$1.15/hr。** 5–7 小时 ≈ $6–8。

如果预算卡死非要用便宜卡，**5090 32G 跑 `max_running_requests: 8` 是可接受的降级**，但报告里必须把「这不是 canonical 配置」写在最显眼的位置，且不能和 cookbook 的 H100 数字并排比较——它测的是另一个东西。

> 💡 **顺手可交付的一条**：无论最后用哪张卡，「**dots.tts 默认配置的显存下限是 ~20 GiB 声学池 + ~5 GiB 权重 + backbone KV**」这条本身就值得写进报告，cookbook 现在只有定性描述（"On a full GPU the default 16 × 500 layout is intended"），没有数字。

---

## 四、准备工作与开机

前置条件、SSH 密钥的两个坑（`create ssh-key` 要传**内容**不是路径；带 passphrase 的私钥在非交互环境签不了名，要单独生成一把无密码的）、vastai skill 装进项目目录要加 `.git/info/exclude`——**这三条完全沿用
[1888-whisper-profiling.md](./1888-whisper-profiling.md) 第二节，不重复。**

### 筛选与创建

```bash
vastai search offers 'gpu_name=A100_SXM4 num_gpus=1 cuda_vers>=13.0 \
  disk_space>=250 reliability>0.985 rentable=true' -o 'dph'
```

- **CUDA ≥ 13 是硬指标**（项目钉了 `flashinfer[cu13]` 等）
- **显存必须 ≥ 80G**，筛完用 `--raw` + jq/python 核一遍 `gpu_ram`——CLI 的表格输出会折行，容易看错
- **选 on-demand 不要 bid**：被抢占那一轮数据就废了
- **磁盘 ≥250G**（模型 + SeedTTS + 转写用的 Qwen3-ASR-1.7B），实际申请 **150G** 就够

```bash
vastai create instance <OFFER_ID> \
  --image hongccc/sglang-omni:dev \
  --disk 150 --ssh --direct
```

### ⚠️ 坑：py-spy 需要的 `CAP_SYS_PTRACE` 拿不到

[#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 的实测教训：租用容器里 `yama.ptrace_scope = 1`、`CapEff` 不含 `CAP_SYS_PTRACE`、且 sysctl 在容器内只读，**py-spy 和 nsys 都用不了**。

`vastai create instance --env` 官方描述只保证 env 变量和端口映射，`--cap-add=SYS_PTRACE` 能不能透传**不确定**。可以试：

```bash
vastai create instance <OFFER_ID> --image hongccc/sglang-omni:dev \
  --disk 150 --ssh --direct --env '--cap-add=SYS_PTRACE'
```

**但别把方案压在这上面。** 见第十节——dots.tts 这次的 Layer 2 有一条**不需要任何特权**的路，而且比 py-spy 更贴合 issue 要的「per-stage timing」。

---

## 五、Agent 怎么驱动远程机器

和 [1888-whisper-profiling.md](./1888-whisper-profiling.md) 第四节一致，不重复。要点：

- **控制面走 vastai CLI，数据面走 SSH**。`vastai execute` 是一次性受限命令，服务常驻 + 旁路扫描 + 中途看进度它都做不到
- **长任务一律 tmux**，`tmux capture-pane -p -t <s> | tail -30` 看进度
- **`create instance`（花钱）和 `destroy instance`（不可逆）每次人工确认**

```
本地 Mac (Claude Code)
   │  ssh   ──────────────▶  Vast 实例 A100 80G
   │                          ├── tmux serve : sgl-omni serve  (:8000 常驻)
   │                          ├── tmux bench : benchmark 扫描
   │                          └── tmux util  : nvidia-smi 采样器
   ◀── rsync ────────────      runs/<ts>/{*.json, events/, raw/, logs/}
```

---

## 六、Phase 0 — 前置检查

```bash
cd /workspace/sglang-omni
mkdir -p runs/$(date +%Y%m%d-%H%M)/{raw,logs,events,util}
RUN=runs/$(ls -t runs | head -1)

# ① 硬件 + 驱动 + 当前占用（同时看利用率和显存）
nvidia-smi --query-gpu=index,name,driver_version,utilization.gpu,memory.used,memory.total \
  --format=csv | tee $RUN/env_gpu.txt
nvidia-smi | tee -a $RUN/env_gpu.txt

# ② 软件栈指纹（TTS benchmark 没有 --fingerprint，必须手工留档）
{
  python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda);print('transformers',transformers.__version__)"
  pip show sglang sglang-omni 2>/dev/null | grep -E "^Name|^Version"
  echo "omni commit: $(git rev-parse HEAD)"
  echo "dirty: $(git status --porcelain | wc -l) files"
} | tee $RUN/env_sw.txt
```

⚠️ **`benchmark_tts_seedtts.py` 没有 `--fingerprint`**（ASR 那个有）。[#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §1 第 1 条和 [#1850](https://github.com/sgl-project/sglang-omni/pull/1850) 的「baseline fingerprint 必须含 commit SHA」这次**只能靠上面这段手工留**。别漏。

**优势照旧**：租的是独占实例，不存在共享主机的 CPU 争抢——在报告里写明。

---

## 七、Phase 1 — 拉模型和数据集

```bash
hf download dots-studio/dots.tts-mf
hf download Qwen/Qwen3-ASR-1.7B          # 转写阶段算 WER 用，别等到最后才下
python -m benchmarks.dataset.prepare --dataset seedtts   # zhaochenyang20/seed-tts-eval-arrow
```

| 名字 | 仓库 | 用途 |
| --- | --- | --- |
| `seedtts` | `zhaochenyang20/seed-tts-eval-arrow` | **短文本**，EN 1088 条，同时算 WER |
| （文本语料） | `zhaochenyang20/seed-tts-eval` rev `8f5e1aa2` `en/meta.lst` | `benchmark_tts_serving` 的 `seedtts-en` 语料 |
| 参考音频 | 仓库内 `docs/_static/audio/male-voice.wav` | dots.tts **必须**带 reference，没有 zero-shot voice |

> ⚠️ **dots.tts 没有 zero-shot voice preset。** 默认连续批处理部署下，**不带 `references` 的请求直接被拒**。所有 benchmark 命令都要带 `--ref-format references`（cookbook 里 dots 的复现命令就是这么写的），漏了会全量 4xx。

💡 Phase 0 + Phase 1 可以写成 `--onstart` 脚本，实例一起来就自动跑完。

---

## 八、Phase 2 — 冒烟：先确认默认配置起得来

**用配置文件，不要只给 `--model-path`。** cookbook 明确说了：只给 `--model-path` 会保留编译过的 tail 和批处理，但 **backbone decode 退回 eager**，慢。

```bash
tmux new -d -s serve "sgl-omni serve \
  --model-path dots-studio/dots.tts-mf \
  --config examples/configs/dots_tts.yaml \
  --allowed-local-media-path docs/_static/audio \
  --port 8000 2>&1 | tee $RUN/logs/serve.log"

until curl -sf localhost:8000/health >/dev/null; do sleep 5; done; echo UP
```

### 🔑 启动日志里有一行必须抄下来

```bash
grep "acoustic pools (estimate)" $RUN/logs/serve.log
# dots.tts acoustic pools (estimate): slots=16 nfe=4 patch_capacity=501 dtype=torch.bfloat16
#   total=X.XX GiB (dit_kv=... encoder_kv=... scratch=... aux=...) cuda_free=... cuda_total=...
grep "latent engine backend" $RUN/logs/serve.log
```

**这行就是第三节那张推算表的实测答案。** 抄进报告，替换掉估算值。

### 冒烟

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model dots-studio/dots.tts-mf --ref-format references \
  --base-url http://127.0.0.1:8000 --port 8000 \
  --lang en --max-samples 20 --max-concurrency 4 --warmup 4 --seed 42 \
  --generate-only --use-existing-server \
  --output-dir $RUN/smoke
```

| 结果 | 怎么办 |
| --- | --- |
| ✅ 跑通 | 进 Phase 3 |
| ❌ `acoustic-tail admission failed at startup` | 显存不够。**先看那行 estimate 的实测值**，再按第三节的表降 `max_running_requests` / `max_generate_length`。**记下来，这是报告里的一条发现** |
| ❌ `acoustic tail admission failed: ran out of slots` | 客户端并发超过了 `max_running_requests=16`。**这条要留到 Phase 3 专门测**——它是 serving 行为，不是 bug |
| ❌ 请求全 4xx | 多半是漏了 `--ref-format references` |

⚠️ **`--warmup` 别省。** [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 实测首轮比稳态慢 **34 倍**，会把整张汇总表污染成假数据。dots.tts 还多一层 `optimize: true` 的编译 tail 和 backbone CUDA graph，冷启动代价只会更大。

---

## 九、Phase 3 — Layer 1 + Layer 3（并发扫描）

### ⚠️ 工具缺口：TTS 的 benchmark 比 ASR 的少四个开关

这是照搬 [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 命令**会直接失败**的地方：

| 开关 | `benchmark_asr_seedtts` | `benchmark_tts_seedtts` | 替代方案 |
| --- | :---: | :---: | --- |
| `--concurrencies` | ✅ | ✅（**但要求 `--generate-only`**） | — |
| `--warmup` | ✅ flag | ✅ **int**（默认=并发数） | 注意类型不同 |
| `--repeats` | ✅ | ❌ | **shell for 循环包一层** |
| `--sample-util` / `--util-gpu-ids` | ✅ | ❌ | **外挂 nvidia-smi 采样器** |
| `--fingerprint` | ✅ | ❌ | Phase 0 手工留档 |
| `--save-raw-dir` | ✅ | ❌ | `--output-dir` 本身就写 per-request 记录 |

而且——**[#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 已经验证过 `--sample-util` 即使有也不报 GPU 利用率**（只有显存/功耗/CPU）。所以 Layer 1 的忙碌比**本来就得外挂 nvidia-smi**，这次只是从「意外」变成「计划内」。

### 采样器 + 扫描

```bash
# 后台采样器，100 ms 一次
tmux new -d -s util "nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv,noheader,nounits -lms 100 > $RUN/util/util_all.csv"
```

```bash
# Layer 3 主扫描：3 次重复靠 shell 循环
tmux new -d -s bench 'for rep in 1 2 3; do
  for C in 1 2 4 8 16 32 64; do
    echo "=== rep=$rep conc=$C $(date -Is) ===" | tee -a '"$RUN"'/util/marks.txt
    python -m benchmarks.eval.benchmark_tts_seedtts \
      --meta zhaochenyang20/seed-tts-eval-arrow \
      --model dots-studio/dots.tts-mf --ref-format references \
      --base-url http://127.0.0.1:8000 --port 8000 \
      --lang en --max-samples 1088 --max-concurrency $C --warmup 10 --seed 42 \
      --generate-only --use-existing-server \
      --output-dir '"$RUN"'/seedtts_en/rep${rep}_c${C}
  done
done 2>&1 | tee '"$RUN"'/logs/sweep.log'
```

`marks.txt` 里的时间戳用来把 `util_all.csv` 切成每个档位的区间——**这是把 Layer 1 和 Layer 3 对上的唯一办法**，别忘了打。

### 三个必须覆盖到的点

1. **并发扫到 64**（cookbook 只到 32）。默认 `max_running_requests=16`，超过之后要么排队要么 `ran out of slots`——**哪种、在哪个并发点开始，是这次要答的问题**。
2. **判定规则**：忙碌比接近饱和 → 瓶颈在 GPU kernel，跳过 Layer 2/4；远低于 100% → 进 Phase 4。
3. ⚠️ **别把「利用率随并发下降」直接读成「并发还不够」**。方法论 Layer 3 专门预警：这也可能是 host-bound 的症状（CPU 侧同步开销随并发变重，在 kernel 时间线上撕出更多空隙）。[#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 在 Whisper 上正好撞上了这个形态。

### WER 要单独一轮，且不能和服务抢卡

`--concurrencies` / `--generate-only` 只生成音频；WER 要第二趟 `--transcribe-only`，**它会自己起一个 Qwen3-ASR-1.7B 服务**。

```bash
tmux kill-session -t serve          # ⚠️ 先把 dots 服务停掉，否则两个模型抢显存
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model dots-studio/dots.tts-mf --ref-format references \
  --lang en --seed 42 --transcribe-only --port 8000 \
  --output-dir $RUN/seedtts_en/rep1_c16
```

⚠️ **停服务 = 下次开机要重付一次冷启动**（编译 tail + CUDA graph 捕获）。所以**把所有生成轮次跑完，最后统一转写**，别来回切。

---

## 十、Phase 4 — Layer 2：per-stage timing（不需要特权）

**这是 dots.tts 这次比 Whisper 那次好做的地方，也是 issue 显式点名的交付项。**

`sglang_omni` 自带一个 request-level 事件记录器，通过 HTTP 开关，**不需要 `CAP_SYS_PTRACE`**：

```bash
# 开（enable_torch=false → 只记事件，不起 torch profiler，开销小得多）
curl -s -X POST localhost:8000/start_profile -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"dots-c8\",\"enable_torch\":false,\"event_dir\":\"$PWD/$RUN/events/c8\"}"

# 跑一轮定量负载（并发 8，200 条足够）
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model dots-studio/dots.tts-mf --ref-format references \
  --base-url http://127.0.0.1:8000 --port 8000 \
  --lang en --max-samples 200 --max-concurrency 8 --warmup 10 --seed 42 \
  --generate-only --use-existing-server --output-dir $RUN/layer2_c8

# 关
curl -s -X POST localhost:8000/stop_profile -H 'Content-Type: application/json' -d '{}'

# 渲染阶段拆解
python -m sglang_omni.profiler $RUN/events/c8 --format table | tee $RUN/stage_c8.txt
python -m sglang_omni.profiler $RUN/events/c8 --out $RUN/stage_c8.json
```

产出两张表：**stage breakdown**（`stage / interval / count / total_ms / avg_ms / p95_ms`）和 **hop breakdown**（阶段间传递）。dots.tts 的三个阶段 `reference_encode` / `latent_engine` / `vocoder` 都会出现。

**至少在并发 1、8、32 各做一次**，看阶段占比**随并发怎么变**——这正好回答第二节说的「给 codec lock 争用定一个量级上界」：`vocoder` 阶段的 `avg_ms` 和 `p95_ms` 之差随并发拉大，就是锁争用的端到端证据。

⚠️ 三点：

- `enable_torch: true` 会起 torch profiler，开销大且产物巨大。**Layer 2 用 `false`**，需要 kernel 级细节时再单独开一小轮。
- **重复 3 次**（方法论要求）。三次的最高占比阶段不一致就是噪声，不能当结论。
- py-spy 仍然是备选（如果 `--cap-add` 侥幸生效）。但事件法**更贴合 issue 要的「per-stage timing」**，优先用它。

**Layer 4（A/B）是条件性的**：只有 Layer 2 真找到可疑开销源才做——单变量、两臂 warmup 对齐、2–3 次重复，且**要重跑事件采集确认那个阶段的占比真的降了**。光看吞吐不算数。

---

## 十一、Phase 5 — 长文本、流式 TTFA、质量回归

issue Scope 的第三条要求「短文本和长文本分开」，Done-when 要求 TTFA 和 RTF。三件事分三轮。

### ① 流式 TTFA（短文本）

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model dots-studio/dots.tts-mf --ref-format references \
  --base-url http://127.0.0.1:8000 --port 8000 \
  --lang en --max-samples 300 --concurrencies 1,8,32 --warmup 10 --seed 42 \
  --stream --response-format pcm \
  --generate-only --use-existing-server \
  --output-dir $RUN/stream_ttfa
```

benchmark 自己会算 `audio_ttfp_s` / `ttfa_p95`——**不用另写工具**。cookbook 里完全没有流式的数字，这一整块是新的。

### ② 长文本

⚠️ **两个 harness 都没有现成的长文本语料**：`benchmark_tts_seedtts` 的 SeedTTS 是单句，`benchmark_tts_serving` 的 `VALID_TEXT_CORPORA` 只有 `{"seedtts-en"}`。

长文本走 **`benchmark_tts_serving.py` 的 `long_prefill_decode` workload**——它会把语料拼到 `MAX_SPEECH_INPUT_CHARS = 4096` 字符，正是我们要的形状：

```bash
cat > $RUN/spec_long.json <<'EOF'
{
  "base_url": "http://127.0.0.1:8000",
  "model_name": "dots-studio/dots.tts-mf",
  "test_type": "engine",
  "seed": 42,
  "params": {
    "profile": "stress",
    "enabled_endpoints": ["speech", "speech_stream"],
    "file_ref_audio": "file:docs/_static/audio/male-voice.wav",
    "file_ref_text": "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time.",
    "load_stages": [
      {"id": "long-text-c8",  "mode": "closed_loop", "request_count": 64,
       "max_concurrency": 8,  "enabled_endpoints": ["speech"]},
      {"id": "long-text-c32", "mode": "closed_loop", "request_count": 128,
       "max_concurrency": 32, "enabled_endpoints": ["speech"]}
    ]
  }
}
EOF

python -m benchmarks.eval.benchmark_tts_serving \
  --spec $RUN/spec_long.json --out $RUN/serving_long
```

⚠️ **这个 spec 是照 `benchmarks/tts_serving/examples/stress.json` 的字段推出来的骨架，没在真机验过。** `scheduled` 模式的 `workload_schedules`（`long_prefill_decode` 就在里面）字段更多也更挑，**第一次务必先用 `request_count: 2` 跑一遍看 spec 解析器认不认**，再放量。spec 解析失败是 harness 退非零，不是 `overall.passed=false`——两者要分清。

> 📌 **dots.tts 的长度天花板**：`max_generate_length=500` patch × 每 patch ≈160 ms = **约 80 秒音频**。4096 字符的输入大概率会撞到这个上限被截断。**撞上就是一条发现**（「长文本工作负载受 `max_generate_length` 而非算力限制」），照实写，别去调大它——调大了显存表要整个重算。

### ③ Layer 5 质量回归

用 Phase 3 存下来的音频跑 `--transcribe-only`，**看各并发档之间 WER 有没有漂**。cookbook 的 H100 基线是 **1.24%–1.35%（并发 1→32）**，我们的数字应该落在同一量级；差太多就先怀疑环境而不是急着下结论。

---

## 十二、时间与预算

### 分段

```bash
vastai stop instance <id>      # 保留磁盘，只收存储费
vastai start instance <id>
vastai destroy instance <id>   # 彻底销毁
```

⚠️ **停机 ≠ 占住机器**，Vast 是市场，stop 后 GPU 释放回池子，再 start 时可能已被别人租走。同天分段可以，跨天有风险。

| 段 | 做什么 | 估时 |
| --- | --- | --- |
| 1 | Phase 0 + 1 + 2（含冷启动） | 1–1.5 h |
| 2 | Phase 3 主扫描（7 档 × 3 重复 × 1088 条） | **2–3 h** |
| 3 | Phase 4 事件采集 + Phase 5 三轮 | 1.5–2 h |
| 4 | 统一转写算 WER + 拉数据 + **destroy** | 0.5 h |

合计 **5–7 小时，A100 80G ≈ $6–8**。

### 扫描耗时怎么估

cookbook 的 H100 数据：并发 1 是 **0.935 req/s**，并发 32 是 **4.988 req/s**。1088 条：

| 并发 | H100 单轮 | A100 估算（约慢 1.5–2×） |
| ---: | ---: | ---: |
| 1 | 19 min | **30–40 min** |
| 8 | 4.7 min | 7–9 min |
| 32 | 3.6 min | 5–7 min |

**并发 1 那一档单独就要半小时以上，×3 重复 = 1.5 小时**，占掉整个扫描的一半。

> **对策一**：并发 1 和 2 只跑 **1 次重复 + `--max-samples 300`**，高并发档才上全量 ×3。在报告里写明各档的样本数和重复数——**方法论允许不等重复，只要写清楚**。
>
> **对策二**：第一次跑先只做 `--concurrencies 1 --max-samples 50`，实测后线性外推。花 3 分钟避免一次 3 小时的意外。

### 存储

150G × $0.30/GB/月 ≈ **每天 $1.5**。**做完立刻 destroy。**

---

## 十三、报告结构

交付物是**贴到 [#1882](https://github.com/sgl-project/sglang-omni/issues/1882) 的一条评论**：

```markdown
## Environment
  GPU / 驱动 / torch / sglang / transformers 版本
  checkpoint、数据集 revision、仓库 commit SHA
  服务启动日志里 acoustic pools 的实测 GiB
  CPU 隔离情况（独占实例，无共享主机争抢）
  用的是哪一版方法论（#1798 正文 / #1850 的 METHODOLOGY.md）

## Reproduce
  逐条可复制的命令（含 --ref-format references）

## Layer 1 — GPU busy ratio
  外挂 nvidia-smi 采样，按 marks.txt 切档；方法 + 数据 + 结论 + 证据强度

## Layer 3 — Concurrency sweep
  短文本表：吞吐 / 延迟 / p95 / RTF / audio_s/s / TTFA / WER，并发 1–64
  长文本表：同上，标明是否撞 max_generate_length
  ⚠️ 与 cookbook H100 表并列时，注明硬件不同代

## Layer 2 — Per-stage timing
  reference_encode / latent_engine / vocoder 的 total/avg/p95，并发 1/8/32 三档
  vocoder 的 p95-avg 差随并发的走势 → 对 #1434/#1459 的旁证

## Layer 5 — Quality regression
  各并发档 WER vs cookbook 的 1.24–1.35% 基线

## Findings
  | 发现 | 证据（命令 + 原始文件） | 建议 | 强度 |
```

**硬性要求**（[#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §3 第 8 条 + [#1850](https://github.com/sgl-project/sglang-omni/pull/1850)）：

- 每层标注**证据强度**（强/中/弱），弱的要说明为什么弱
- **每个数字都要能指回产生它的命令和原始文件**
- **负面结论明确写出来**（「已排除，无需改动」）
- 小幅差异必须跨过**实测的 A/A 噪声底**，不是猜的

---

## 十四、止损清单

| 项 | 做法 |
| --- | --- |
| **先占坑** | 开机前去 [#1882](https://github.com/sgl-project/sglang-omni/issues/1882) 留言认领。0 评论不等于没人在做 |
| **先读既有工作** | 第二节那六份。dots.tts 已经被攻坚过一轮，重复发现 = 白花钱 |
| **别用 24G 卡** | 第三节。默认配置准入要 22.6 GiB 空闲，必失败 |
| **别漏 `--ref-format references`** | dots.tts 没有 zero-shot voice，漏了全量 4xx |
| **别漏 `--warmup`** | 首轮可能慢 34 倍，污染整张表 |
| **别在服务活着时跑转写** | Qwen3-ASR-1.7B 会和 dots 抢显存 |
| **先小后大** | 任何命令第一次跑都 `--max-samples 20`，spec 第一次跑 `request_count: 2` |
| **环境优先** | Phase 0–2 验证环境。**装不通就换机器，别硬扛** |
| **抄下 estimate 那行日志** | 第八节。它是第三节整张推算表的实测答案，也是报告里最容易拿的一条新数据 |
