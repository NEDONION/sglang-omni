# 05 · 学习总结

---

## 一、我学到了什么

### 1. 「多阶段推理」是一个和「LLM 推理」正交的问题

在读这个项目之前，我脑子里的推理系统只有一个形状：tokenize → prefill → decode。SGLang-Omni 让我看到另一类问题：**当一次生成被拆成计算模式完全不同的若干段时，编排本身就是一个独立的系统问题**，跟单段算得快不快无关。

具体证据是这个项目的分工：SGLang 负责「一段 AR 算得快」，SGLang-Omni 负责「七段怎么连、怎么放、怎么传、怎么一起中止」。两者代码完全不重叠。

### 2. 控制面和数据面必须分开

`DataReadyMessage` 里只有 `data_ref`，几百 MB 的 hidden states 走 relay。这个分离带来三个后果，每一个都是我之前不会主动想到的：

- ZMQ 通道永远不会被大张量堵死，控制消息的延迟和数据量解耦
- 传输后端可以**按边**替换（同进程走引用、同卡走 CUDA IPC、跨机走 Mooncake），而 Stage 代码一行不用改
- 生命周期管理变成显式协议（`DataAckMessage`），而不是隐式的 GC

### 3. 声明式拓扑 + 运行时收窄

`wait_for` / `wait_for_fn` 这一对是全项目最优雅的设计。静态字段服务编译期（算进程拓扑、建通道），运行时函数服务每个请求（纯文本请求不等图像编码器）。同一个 pipeline 因此能同时服务纯文本和多模态请求，而不需要两套部署。

这个模式我可以直接搬到任何「拓扑固定、路径可变」的系统里。

### 4. 组合优于继承的一个极端案例

`OmniScheduler.__getattr__` 把上游类的方法动态绑到自己实例上。这不是教科书写法，但在「要复用一个大类的行为、又不想继承它的构造过程」这个约束下，它是有效的。**关键是这个项目诚实地承担了代价**：SGLang 版本 pin 死 0.5.18，静态分析失效，overlap scheduling 明确不支持并在代码里给出理由。

### 5. 好的注释写的是「为什么」不是「是什么」

这个项目的注释密度不高，但每一条都在解释决策。几个例子：

- `# note (Jiaxin Deng): daemons must predate the first CUDA init` —— 顺序约束的理由
- `# Note (Chenyang): bumped 8192 → 32768 because the V1 talker prefill replays the full thinker prompt as projected embeddings, and a 30-frame video prompt is ~22K positions` —— 一个魔数的完整来历
- `# note (yexiaodong): SGLang's PyPI wheel has unconditional CUDA dependencies, so Apple Silicon installs this exact tag from source` —— 依赖 marker 分叉的原因

**注释署名 + 给出触发原因**，这个习惯值得抄。

---

## 二、系统的设计思路

我把它归纳成五条，按重要性排：

| # | 原则 | 在代码里的体现 |
| --- | --- | --- |
| 1 | **框架层不认识模型** | `Stage` 不根据 scheduler 类型分支；`Relay` 只搬扁平字节；`Coordinator` 对调度器实现不可知 |
| 2 | **控制面传引用，数据面传张量** | `DataReadyMessage.data_ref` + 五种 relay 后端 |
| 3 | **拓扑只在代码里定义** | YAML/CLI 只能覆盖已有 stage 的设置，永远不能增删 stage |
| 4 | **契约写在类型和序列化边界上** | `DataReadyMessage.to_dict()` 里的三条互斥断言；`from_dict()` 再查一遍 |
| 5 | **默认宽容 + strict 给测试** | `registry.py` 里 import 失败只 warn；`strict=True` 时才 raise |

第 1 条是可扩展性的地基。**接一个新模型不需要碰框架层任何一行代码**——只要在 `models/<name>/` 下放一个带 `EntryClass` 的 `config.py`，registry 自动发现。唯一的例外是要在 `model_runner/sglang_model_runner.py::_register_omni_model` 加一行让 SGLang 认识新架构。

---

## 三、值得借鉴的地方

### 直接可搬的

1. **静态上界 + 运行时收窄**（`wait_for` / `wait_for_fn`）—— 任何拓扑固定但路径可变的系统
2. **字符串路径的延迟解析**（`factory_path="pkg.module.fn"` + `import_string()`）—— 配置对象要跨 spawn 传递时的标准解法，顺带让每个进程只 import 自己要的模块
3. **控制消息先发再等完成** —— 所有基于 credit / 流控的传输层
4. **有界集合 + 一次裁到水位线** —— 长跑服务里所有「记住最近 N 个 id」的场景，摊销 O(1)
5. **配置来源溯源**（`config/provenance.py` + `sgl-omni config explain PATH`）—— 多层覆盖的配置系统必备。能回答「这个值是谁设的、覆盖了什么」，排查成本降一个数量级

### 需要想清楚再搬的

- **`__getattr__` 组合上游类** —— 只在依赖版本可以 pin 死时才划算
- **注册表自动发现 + 宽容 import** —— 需要配套 `strict=True` 的测试，否则依赖问题会静默变成「不支持该架构」

---

## 四、潜在优化点

按「证据强度」排，前三条有直接证据。

### 1. 硬编码的批处理等待窗口（有 issue 佐证）

`models/qwen3_omni/stages.py:852` 和 `:922` 把 encoder 的 `max_batch_wait_ms` 写死成 50。低并发下这 50 ms 纯粹是 TTFT 的净损失。社区已经把它开成 [#1147](https://github.com/sgl-project/sglang-omni/issues/1147)，且明确说这个值从来没被扫过参数。

**推广一点**：任何 `factory_args` 里没暴露的性能常数都值得查一遍。工厂参数不可配置，就等于这个值永远不会被调优。

### 2. 模型接入层的重复（维护者已在处理）

19 个模型各有一份 `stages.py`，`request_builders.py` 从 117 行到 1398 行不等。README 的 2026/08 条目写着「TTS architecture refactor: shared pipeline state, engine construction, reference encoding, capability metadata, and vocoder scheduling」，说明维护者认同这里有重复。

`models/` 占全项目 66%（95k / 145k）。这个比例本身不是问题——模型代码本来就该多——但同形状的 TTS 三阶段流水线被写了十几遍，抽象化的空间很明确。

### 3. 单类过重

| 文件 | 行数 |
| --- | --- |
| `scheduling/omni_scheduler.py` | 2677 |
| `pipeline/stage/runtime.py` | 1843 |
| `comm/engine.py` | 1136 |
| `relay/cuda_ipc.py` | 1209 |
| `model_runner/base.py` | 1007 |

`Stage` 一个类里同时管：控制消息分发、relay 读写、扇入聚合、流式路由、outbox 排空、abort、profiler 控制、TP leader/follower。它的「IO 壳」定位是清楚的，但 1843 行意味着新人改任何一处都要理解全部。

### 4. 非 CUDA 后端覆盖薄（这也是机会）

`platforms/` 只有 754 行要覆盖 cuda / rocm / xpu / npu / musa / apple / cpu 七个后端，`tests/unit_test/platforms/` 下只有一个 `test_apple.py`。Apple Silicon 路径目前只有 Qwen3-ASR 能端到端跑通，且限制很多：`tp_size=1`、只有贪心解码、无 radix cache / chunked prefill / CUDA graph，Torch MPS 路径 KV 预算只有 2048 token。

**对我来说这是最好的切入点**：维护者主力在 H100 上，Mac 上的问题他们复现不了。

### 5. 贡献规则没有单一入口

仓库没有 `CONTRIBUTING.md`。实际起作用的规则散在三处：`docs/developer_reference/main.md`（架构与目录约定）、`.github/pull_request_template.md`（CI 触发靠 maintainer 打 `run-ci` 标签、斜杠命令选模型）、`.github/CODEOWNERS`（按目录分派 review）。新人第一次提 PR 大概率会卡在「CI 为什么不跑」。

### 6. 有界集合裁剪不保序（小）

`_record_bounded_request_id` 用 `set` 迭代顺序删元素，被裁掉的不一定是最老的。在当前场景（只需要「近期的大概率还在」）是可接受的取舍，但如果哪天需要严格的时间窗口语义，这里会是个坑。

---

## 五、下一步

### 读代码

- [ ] `pipeline/stage/runtime.py` 的 `_on_stream_chunk` 和 `_send_stream_to_target` 两段——流式路径是我目前理解最浅的部分
- [ ] `relay/cuda_ipc.py` 的发送侧显存池：slot 分配、多 slot 组成一个逻辑传输、一个 ACK 释放整段
- [ ] `config/topology.py` 的 `compile_logical_processes`：副本、TP、进程共置怎么算出来
- [ ] 对照读 `models/qwen3_tts/config.py`（3 阶段）和 `models/qwen3_omni/config.py`（7 阶段）

### 动手

- [ ] 本机跑通 CPU 单测 lane：`CUDA_VISIBLE_DEVICES="" pytest tests/ -m "not benchmark and not accelerator"`
- [ ] M2 Max 上跑通 Qwen3-ASR MLX 路径（`mlx-community/Qwen3-ASR-0.6B-4bit`）
- [ ] 用 `sgl-omni config resolve --show provenance` 看一个 YAML 的完整合并结果
- [ ] 挑 `tests/unit_test/pipeline/` 下一个测试，改坏一行代码看它怎么失败——这是验证「我真的读懂了」的最快办法

### 可能的第一个 PR

按门槛从低到高：

1. 文档 / cookbook（`docs-check.yaml` 跑在无 GPU 的 GitHub runner 上）
2. Rust Router（`sglang_omni_router/rust/`，76 KB，是全项目唯一读得完的模块，CI 也无 GPU）
3. Apple Silicon / MPS 路径的 bugfix——本机能复现，维护者不能
4. [#1147](https://github.com/sgl-project/sglang-omni/issues/1147) 的第一半：把 `max_batch_wait_ms` 从工厂硬编码改成可配置（默认不变），扫参数那半留给有 GPU 的时候
