# Whisper 运行时画像 · 执行手册

| | |
| --- | --- |
| **目标 issue** | [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) `[Runtime Profiling] Whisper ASR` |
| **方法论** | [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) Layer 1–5 |
| **模型** | `openai/whisper-large-v3`（约 15 亿参数，FP16 权重约 3G） |
| **硬件** | Vast.ai 租用 RTX 4090 24G 单卡 |
| **交付物** | **一条 issue 评论**（不是 PR） |
| **预算** | 机时 $1.5–2，含试错上限 $5 |
| **状态** | 🟡 准备中 |

> 依据：命令、参数、默认值、数据集 revision 均取自 main 分支 332f4fc 的实际文件。仓库演进后以代码为准。

---

## 一、先明确「做完」是什么样

[#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 的验收标准只有两条，但 [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §3 把它展开成九步：

| # | 做什么 | 必做 |
| --- | --- | --- |
| 1 | 记录基线环境：框架版本、GPU 型号、checkpoint、数据集版本 | ✅ |
| 2 | 选一张利用率低且显存够的卡，全程盯着别被抢占 | ✅ |
| 3 | **Layer 1** — 判定瓶颈在 GPU 算力还是 CPU/调度 | ✅ |
| 4 | **Layer 2** — py-spy 采样，定位最高占比叶子帧，重复 3 次 | ⚠️ 条件 |
| 5 | **Layer 3** — 并发扫描，产出 GPU 利用率 vs 并发数的表 | ✅ |
| 6 | **Layer 4** — 对 Layer 2 的发现做单变量 A/B | ⚠️ 条件 |
| 7 | **Layer 5** — 长序列数据集上的功能回归 | ✅ |
| 8 | 写报告：按层组织，标注证据强度，附发现—证据—建议表 | ✅ |
| 9 | 把通用的坑回填进 [#1798](https://github.com/sgl-project/sglang-omni/issues/1798) 的 §1/§2 | 加分 |

两条容易读漏的：

- **负面结论同样有效。** 方法论原文：「当前默认值已经够好、无需改动」是同样有效的产出，只要推理写清楚。
- **短音频和长音频两组都要覆盖**，不是二选一。好在长音频那一轮同时满足 Scope ② 和 Layer 5。

### 决策树

```
Layer 1 ─┬─ GPU 忙碌比接近饱和 → 跳过 Layer 2、4，直接 Layer 5
         └─ 忙碌比远低于 100%  → Layer 2 ─┬─ 有发现 → Layer 4
                                          └─ 没发现 → 跳过 Layer 4
```

---

## 二、准备工作（一次性，本地做）

| # | 事情 | 状态 |
| --- | --- | --- |
| 1 | Vast.ai 注册、充值、`vastai set api-key` | ✅ |
| 2 | 本地有 SSH 密钥对 `~/.ssh/id_ed25519` | ✅ |
| 3 | 公钥注册到 Vast 账户 | ✅ |

### ⚠️ 坑 1：`create ssh-key` 要传内容，不是路径

官方 SKILL.md 的示例是错的，照抄会踩：

```bash
# ❌ 错：把文件路径当成密钥存进去了，API 不校验，还返回 success: True
vastai create ssh-key ~/.ssh/id_ed25519.pub

# ✅ 对
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
```

验证 `public_key` 字段是否以 `ssh-ed25519 AAAA` 开头：

```bash
vastai show ssh-keys
```

存错了的后果：建实例时 `authorized_keys` 里写进一串垃圾，**SSH 连不上，只能销毁重建**。

**顺序**：注册 key 必须在 `create instance` **之前**，否则要多跑一次 `vastai attach ssh`。

### ⚠️ 坑 2：vastai skill 装进了项目目录

`npx skills add vast-ai/vast-cli` 会装到**当前目录**的 `.agents/` 和 `.claude/skills/`，还生成 `skills-lock.json`。sglang-omni 的 `.gitignore` 不忽略这些，容易误提交。加进本地排除（不动上游的 `.gitignore`）：

```bash
printf '.agents/\n.claude/skills/vastai*\nskills-lock.json\n' >> .git/info/exclude
```

另外安装器给 `vastai` skill 打了 **High Risk** 标签。我读过 2082 行代码，唯一外部调用是 `utils.py` 里的 `subprocess.run`（用来调本机 `vastai`），没有 `eval`/`exec`/`os.system`，没有外发数据。判断是自动扫描器对「能执行 subprocess + 能调付费 API」的通用标记。

---

## 三、选机器

### 为什么是 4090 而不是 5090

| | RTX 4090 (SM89) | RTX 5090 (SM120) |
| --- | --- | --- |
| 价格 | $0.335/hr 起 | $0.388/hr 起 |
| 显存 | 24G（Whisper 只要 ~3G，够） | 32G（用不上） |
| 风险 | 项目里被反复走过的路径 | ⚠️ [#160](https://github.com/sgl-project/sglang-omni/issues/160) 未解决：sm_120 在 flash-attention 里崩溃 |

**目标是量 Whisper 的性能，不是替项目 debug Blackwell 兼容性。**

### 筛选条件

```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 cuda_vers>=13.0 \
  disk_space>=200 reliability>0.985 rentable=true' -o 'dph'
```

- **CUDA ≥ 13 是硬指标**：项目钉了 `flashinfer[cu13]`、`nixl-cu13`、`mooncake-transfer-engine-cuda13`，`pyproject.toml` 明确写着通用的 cu12 wheel 在 cu130 上会挂。已知能跑通的环境：driver 580.159.03 / CUDA 13.0（来自 issue [#1697](https://github.com/sgl-project/sglang-omni/issues/1697) 的报告）
- **选 on-demand，不要 bid**：竞价实例被抢占，正在跑的那轮数据就废了
- **磁盘 ≥200G**，但**实际只申请 ~120G**（见下方存储费）

### 创建（要人工确认，开始计费）

```bash
vastai create instance <OFFER_ID> \
  --image hongccc/sglang-omni:dev \
  --disk 120 --ssh --direct
```

`--image` 直接指定官方镜像，UCX、flash-attn、SGLang、CUDA 都是预编译好的，省掉手工装环境的全部痛苦。这是选 Vast 而不是国内容器平台的**唯一理由**——容器平台里没法再跑 Docker。

---

## 四、Agent 怎么驱动远程机器

```
本地 Mac (Claude Code)
   │  ssh   ──────────────▶  Vast 实例 RTX 4090
   │                          ├── tmux: sgl-omni serve  (:8000 常驻)
   │                          └── tmux: benchmark 扫描
   ◀── rsync ────────────      runs/<timestamp>/{*.json, raw/, logs/}
```

### 为什么必须 SSH，CLI 不行

Vast CLI 是**控制面**工具，管账户和实例生命周期；**数据面**（在机器里干活）走 SSH。证据是它自己的命令名：`ssh-url` 的描述是 "ssh url **helper**"，只打印地址不建立连接；整个命令列表里没有 `shell`/`login`/`connect`。

`vastai execute` 能跑命令，但官方描述是 "a **(constrained)** remote command"，且是一次性的：

| 需求 | `vastai execute` |
| --- | --- |
| 服务常驻 :8000，旁边同时跑扫描 | ❌ 进程随命令结束而死 |
| 扫描跑 40+ 分钟，中途看进度 | ❌ 拿不到中间状态 |
| py-spy **attach 到运行中进程的 PID** | ❌ |
| 断线后回来接着看 | ❌ 无会话概念 |

### 配 SSH 别名（一次性）

```bash
vastai ssh-url <INSTANCE_ID>          # 形如 ssh://root@ssh4.vast.ai:12345

cat >> ~/.ssh/config <<'EOF'
Host vast-gpu
  HostName ssh4.vast.ai
  Port 12345
  User root
  ServerAliveInterval 30
  ServerAliveCountMax 6
EOF

ssh vast-gpu "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"
```

### 长任务一律进 tmux

```bash
ssh vast-gpu "tmux new -d -s bench 'cd /workspace/sglang-omni && bash run_sweep.sh 2>&1 | tee runs/latest/sweep.log'"

ssh vast-gpu "tmux capture-pane -p -t bench | tail -30"        # 看进度
ssh vast-gpu "tmux has-session -t bench 2>/dev/null && echo RUNNING || echo DONE"
```

### 拉结果

```bash
rsync -avz vast-gpu:/workspace/sglang-omni/runs/ ./profiling-runs/
```

### Agent 的权限边界

只读操作（搜机器、看状态、拉日志、跑 benchmark）可直接执行。
**`create instance`（花钱）和 `destroy instance`（不可逆、连盘删）每次都要人工确认。**

---

## 五、Phase 0 — 前置检查

[#1798](https://github.com/sgl-project/sglang-omni/issues/1798) §1 的四条，一条别跳。这些输出直接构成报告的 Environment 段。

```bash
cd /workspace/sglang-omni
mkdir -p runs/$(date +%Y%m%d-%H%M)/{raw,logs}
RUN=runs/$(ls -t runs | head -1)

# ① 硬件 + 驱动 + 当前占用（同时看利用率和显存，不能只看显存）
nvidia-smi --query-gpu=index,name,driver_version,utilization.gpu,memory.used,memory.total \
  --format=csv | tee $RUN/env_gpu.txt

# ② 进程树是否干净 —— 有别的进程占卡就别开跑
nvidia-smi | tee -a $RUN/env_gpu.txt

# ③ 软件栈指纹
{
  python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda);print('transformers',transformers.__version__)"
  pip show sglang sglang-omni 2>/dev/null | grep -E "^Name|^Version"
  echo "omni commit: $(git rev-parse HEAD)"
  echo "dirty: $(git status --porcelain | wc -l) files"
} | tee $RUN/env_sw.txt
```

**两个真实的坑**（来自方法论 §1）：

- **editable 安装的实际 commit 可能和 `pyproject.toml` 的 pin 漂移**，报告里要写清测的是哪个版本
- **`pkill -f` 杀不掉孤儿编译子进程**——它们的 cmdline 只是解释器调用，不含你的启动命令串。方法论记录过这类进程存活 80 分钟以上。要按进程组或 PID 逐个杀

**一个优势**：§1 第 4 条要求共享主机隔离 CPU 核，因为争抢会让数据漂移。**租的是独占实例，不存在这个问题**——在报告里写明，数据可信度比 CI 主机更高。

---

## 六、Phase 1 — 钉版本、拉模型和数据集

数据集全部注册在 `benchmarks/dataset/prepare.py`，带固定 revision：

| 名字 | HuggingFace 仓库 | 用途 |
| --- | --- | --- |
| `seedtts` | `zhaochenyang20/seed-tts-eval-arrow` | 短音频，EN 1088 + ZH 2020 条，算 WER/CER |
| `longlibriheavy-30` | `inesc-id/longlibriheavy:llh_test_30` | 30 秒长音频 |
| `longlibriheavy-60` | `inesc-id/longlibriheavy:llh_test_60` | **60 秒长音频 → Layer 5 首选** |
| `meanwhile` | `distil-whisper/meanwhile:test` | 长音频，结构不同的独白 |
| `stt-benchmark` | `pipecat-ai/stt-benchmark-data` | 备选 |

```bash
MODEL_PATH=$(hf download openai/whisper-large-v3)
echo "MODEL_PATH=$MODEL_PATH" | tee -a $RUN/env_sw.txt

python -m benchmarks.dataset.prepare --dataset seedtts
python -m benchmarks.dataset.prepare --dataset longlibriheavy-60
python -m benchmarks.dataset.prepare --dataset meanwhile
```

> **正确性不是单独一步。** benchmark 脚本本身就算 corpus WER/CER，每个并发档位都输出。Layer 5「功能回归」就是在长音频数据集上再跑一遍看 WER 有没有跑飞，**不需要另找数据集或另写测试**。

💡 可以把 Phase 0 + Phase 1 写成 `--onstart` 脚本，实例一起来就自动跑完，这段时间不用人守着。

---

## 七、Phase 2 — 冒烟测试：24G 到底够不够

**先用默认参数，别提前调优。**

```bash
tmux new -d -s serve "sgl-omni serve \
  --model-path '$MODEL_PATH' \
  --model-name openai/whisper-large-v3 \
  --port 8000 2>&1 | tee $RUN/logs/serve.log"

until curl -sf localhost:8000/health >/dev/null; do sleep 5; done; echo UP
```

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-large-v3 \
  --max-samples 20 --concurrencies 2,32 --repeats 3
```

三种结果：

| 结果 | 怎么办 |
| --- | --- |
| ✅ 跑通 | 默认参数在 24G 上可用，**这本身是个发现**。进 Phase 3 |
| ❌ 启动 OOM | Whisper 的 `mem_fraction_static` **默认写死 0.85**（照 H200 调的）。`--mem-fraction-static` 降到 0.75 → 0.70 → 0.65 |
| ❌ 高并发 OOM | 默认 `max_running_requests=64`。降到 32 → 16，参照 `examples/configs/qwen3_asr_rtx4090.yaml` 的保守配方 |

**每次只改一个值**，记下能起来的最低配置。Whisper 现有基准**全部来自 H200**（见 `docs/cookbook/whisper_asr.md` 的 Benchmark Results），24G 上没有任何公开数据，你量出来的就是新的。

---

## 八、Phase 3 — Layer 1 + Layer 3（一次扫描，两层产出）

工具已经把方法论要的东西内置了：

| 参数 | 对应 |
| --- | --- |
| `--sample-util --util-gpu-ids --util-interval` | **Layer 1** 的 nvidia-smi 采样法 |
| `--concurrencies --repeats --warmup` | **Layer 3** 并发扫描 |
| `--fingerprint` | §1 第 2 条「记录基线环境」 |
| `--save-raw-dir` | Done-when ② 的「原始数据链接」 |
| `--profile-events --profile-event-dir` | 阶段/跳数拆解 |

```bash
tmux new -d -s bench "python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-large-v3 \
  --concurrencies 1,2,4,8,16,32,64 \
  --repeats 3 --warmup \
  --sample-util --util-gpu-ids 0 --util-interval 0.5 \
  --fingerprint \
  --save-raw-dir $RUN/raw/seedtts \
  --output $RUN/seedtts_sweep.json 2>&1 | tee $RUN/logs/sweep.log"
```

**判定**：

- 利用率**接近饱和** → 瓶颈在 GPU kernel，跳过 Layer 2、4，直接 Layer 5
- 利用率**远低于 100%** → 瓶颈在 CPU/调度，进 Phase 4

⚠️ **一个容易误判的模式**：如果利用率随并发升高**反而下降、且波动很大**，别急着下「并发还不够高」的结论——也可能是 host-bound，CPU 侧同步开销随并发变重，在 GPU 时间线上撕出更多空隙。两者根因和修法都不同。

---

## 九、Phase 4 — Layer 2（条件执行）

只在 Layer 1 判定为 CPU/调度瓶颈时做。`--idle` 是方法论明确要求的。

```bash
pip install py-spy
ps aux | grep -i "[s]gl-omni\|[s]tage"      # 找 ASR stage 的 PID

for i in 1 2 3; do
  py-spy record --format raw --idle --subprocesses \
    --pid <PID> --duration 60 --rate 20 -o $RUN/pyspy_r$i.txt
  sleep 10
done
```

- 不熟悉目标进程时**先用 `--rate 5`**，观察会不会拖慢服务
- 看某个线程当下的栈：`py-spy dump --pid <PID>`
- **三次采样的最高占比叶子帧要一致**，否则是噪声，不能当结论

**Layer 4 也是条件性的**：只有 Layer 2 真找到可疑开销源才做 A/B——单变量、两臂 warmup 和 GPU 环境对齐、2–3 次重复，并且**要重跑 py-spy 确认那个叶子帧占比降到接近零**。光看吞吐数字不算数，吞吐会因噪声抖动。

---

## 十、Phase 5 — Layer 5 + 长音频覆盖

```bash
python -m benchmarks.eval.benchmark_asr_longform \
  --dataset longlibriheavy-60 --port 8000 \
  --concurrencies 1,8,32 --repeats 3 --warmup \
  --sample-util --util-gpu-ids 0 --fingerprint \
  --save-raw-dir $RUN/raw/longform60 --output $RUN/longform60.json

python -m benchmarks.eval.benchmark_asr_longform \
  --dataset meanwhile --port 8000 \
  --concurrencies 1,8,32 --repeats 3 --warmup \
  --sample-util --util-gpu-ids 0 --fingerprint \
  --output $RUN/meanwhile.json
```

看两件事：**WER 相对短音频有没有跑飞**，以及**长音频下利用率曲线的形状是否不同**。方法论特别提醒：利用率随并发的趋势**可能在不同 workload 形状之间完全翻转**，必须分别测。

---

## 十一、时间与分段

### 分段租的机制

```bash
vastai stop instance <id>      # 保留磁盘，不收 GPU 费
vastai start instance <id>
vastai destroy instance <id>   # 彻底销毁，停止所有计费
```

| 状态 | 计费 |
| --- | --- |
| `running` | GPU 费 + 存储费 |
| `stopped` | **只有存储费** |
| `frozen` | **仍收 GPU 费**（别用） |

⚠️ **停机 ≠ 占住机器。** Vast 是市场，stop 后 GPU 释放回池子，**再 start 时那台可能已被别人租走**。所以：同一天分 2–3 段可以，跨天停机有风险，停一周基本等于放弃。

### 建议分段

| 段 | 做什么 | 时长 | 停机点 |
| --- | --- | --- | --- |
| 1 | Phase 0 + 1 + 2 | 1–1.5 h | 冒烟通过后 |
| 2 | Phase 3（Layer 1 + 3） | 1.5–2 h | 拉完结果后 |
| 3 | Phase 4（如需）+ Phase 5 | 1.5–2 h | 做完 **destroy** |

合计 **4–5.5 小时 ≈ $1.5–2**。

### 各任务耗时（从 [#1340](https://github.com/sgl-project/sglang-omni/issues/1340) 的 H100 实测推算，4090 约慢 2.5–3 倍）

| 任务 | H100 实测 | 4090 估算 |
| --- | --- | --- |
| 短音频 EN 全量（1088 条 × 3 重复 × 5 档） | 8.7 min | ~25 min |
| 短音频 ZH 全量（2020 条 × 3 重复 × 5 档） | 11.3 min | ~35 min |
| 加到 7 档并发 | — | **60–90 min** |
| Layer 2 py-spy | — | ~15 min |
| Layer 5 长音频 × 2 数据集 | — | **60–90 min** |

⚠️ **误差可能很大**。Whisper 的 encoder 处理固定 30 秒窗口，对短音频单请求开销比 Qwen3-ASR 重，实际可能更慢。

> **对策：第一次跑先只做 `--concurrencies 1` 一档**，看实际耗时再线性外推。花 3 分钟避免一次 90 分钟的意外。

### 存储是隐形成本

需要约 100G，按 $0.30/GB/月 算是 **每天 $1**——放着不用三天就超过全部算力费。**别申请 500G 的大盘，做完立刻 destroy。**

---

## 十二、报告结构

交付物是**贴到 [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 的一条评论**：

```markdown
## Environment
  GPU / 驱动 / torch / sglang / transformers 版本
  checkpoint snapshot、数据集 revision、仓库 commit
  CPU 隔离情况（独占实例，无共享主机争抢）

## Reproduce
  逐条可复制的命令

## Layer 1 — GPU busy ratio
  方法 + 数据 + 结论 + 证据强度（强/中/弱）

## Layer 3 — Concurrency sweep
  短音频、长音频各一张 GPU 利用率 vs 并发 的表

## Layer 5 — Functional regression
  长音频 WER，与短音频对比

## Findings
  | 发现 | 证据 | 建议 | 强度 |
```

**每一层都要标注证据强度**（§3 第 8 条硬性要求），**负面结论要明确写出来**（例如「已排除，无需改动」）。别只贴数字不给判断。

---

## 十三、止损清单

| 项 | 做法 |
| --- | --- |
| 先读已有文档 | `docs/cookbook/whisper_asr.md` 的 Benchmark Results 有 W-PR1 / W-PR2 / async-decode 三组 H200 数据。**开机前在本地读完**，§3 第 3 条明确要求先读，别重复造轮子 |
| 先小后大 | 任何命令第一次跑都加 `--max-samples 20` |
| 环境优先 | Phase 0–2 验证环境能不能通。**装不通就换机器，别硬扛** |
| 先占坑再花钱 | **动手前去 [#1888](https://github.com/sgl-project/sglang-omni/issues/1888) 留言认领。** 这个仓库很卷，而且要先看评论区——有过写完才发现别人早说要做的教训 |
