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

<!-- 后续进展往下追加 -->
