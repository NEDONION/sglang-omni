# 01 · 认识项目

> 系列导航：[索引](./README.md) · **本篇** → [02 配置环境](./02-environment.md)

读代码之前先建立心智模型，否则很难判断一个 issue 该改哪一层。

## 本页目录

- [一、它和 SGLang 的分工](#一它和-sglang-的分工)
- [二、请求走过的链路](#二请求走过的链路)
- [三、目录与职责对照](#三目录与职责对照)
- [四、模型代码的硬性约定](#四模型代码的硬性约定)
- [五、先读哪三份文档](#五先读哪三份文档)

---

## 一、它和 SGLang 的分工

SGLang-Omni 是 **omni / 语音 / TTS 模型的多阶段推理服务运行时**，分工是明确的：

- **SGLang** 负责高性能自回归调度与模型执行——一段 AR 怎么算得快。
- **SGLang-Omni** 负责流水线拓扑、阶段生命周期、阶段间数据传输、模型 family 接入层，以及 OpenAI 兼容的服务接口——七段怎么连、怎么放、怎么传、怎么一起中止。

设计目标叫「多阶段解码」：一次生成被拆成计算模式、依赖结构、资源需求都不同的若干阶段——预处理、encoder、AR 引擎、talker、decoder、vocoder、聚合器。每个阶段配一个匹配自身负载的 scheduler，阶段间数据走 relay 数据面（shared-memory / NCCL / NIXL / Mooncake）。

目前服务的模型横跨四类：omni 对话（Qwen3-Omni、Ming-Omni）、音乐生成（MiniMax Music 3）、语音合成（Higgs Audio v3、MOSS-TTS、Fish Speech S2-Pro、Qwen3-TTS、dots.tts、ZONOS2 等）、转写与说话人分离（Qwen3-ASR、Fun-ASR、MOSS-Transcribe-Diarize），外加一个 Rust 写的 Router。

## 二、请求走过的链路

```
HTTP API → Client → Coordinator → Stage → Scheduler → ModelRunner → model forward
```

中间三层（Coordinator / Stage / Scheduler）是框架自有逻辑，**也是无 GPU 贡献者最容易切入的地方**。

| 层 | 职责 |
| --- | --- |
| HTTP API | OpenAI 兼容的请求响应 schema、SSE 分帧、HTTP 错误 |
| Client | `GenerateRequest` → `OmniRequest`、结果聚合、音频编码 |
| Coordinator | 请求生命周期、入口阶段提交、终态结果收集、abort 广播 |
| Stage | 控制面 IO、relay IO、fan-in、流式路由、scheduler 收发桥接 |
| Scheduler | 每阶段的执行循环，以及失败向 stage outbox 的传播 |
| ModelRunner | AR forward 准备、forward 派发、输出提取 |

## 三、目录与职责对照

| 目录 | 职责 | 典型改动 |
| --- | --- | --- |
| `pipeline/` | 阶段编排、coordinator、进程管理 | 生命周期、abort 广播、流式路由 |
| `scheduling/` | 各阶段调度循环、inbox/outbox 消息类型 | 批处理策略、队列语义 |
| `model_runner/` | AR 阶段的模型运行器抽象 | forward 准备、输出提取 |
| `models/` | 每个模型 family 的 config / stages / 请求构造 | 接入新模型、修某个模型的行为 |
| `config/` | PipelineConfig、StageConfig、配置管理、拓扑 | 参数解析、默认值、校验 |
| `relay/` | 数据面传输后端 | 传输实现、张量搬运 |
| `serve/` | HTTP server 与 OpenAI 兼容适配 | 接口字段、SSE 分帧、错误码 |
| `client/` | API adapter 使用的内部客户端 | 请求转换、结果聚合 |
| `proto/` | 请求、载荷、阶段与控制面消息类型 | 协议字段增删 |

## 四、模型代码的硬性约定

模型专属代码一律放在 `sglang_omni/models/<model>/` 下，按这套布局组织：

```
models/<model>/
├── config.py             # PipelineConfig 子类和 StageConfig 列表
├── stages.py             # stage 工厂
├── routing.py            # 可选：数据驱动的路由辅助
├── request_builders.py   # 阶段间载荷变换
├── payload_types.py      # 该模型专属的载荷状态类型
├── callbacks.py          # 需要时的反馈回调或策略
└── components/           # 模型模块、processor、vocoder、adapter
```

**只有模型局部行为属于这里。** 框架层（Stage、Coordinator、各 scheduler、model-runner 基类、relay、运行时准备、runner）不接受模型专属逻辑——这是评审时最常被打回的一类改动。

## 五、先读哪三份文档

| 文档 | 内容 |
| --- | --- |
| `docs/developer_reference/main.md` | 架构总览，就是本页的权威版本 |
| `docs/developer_reference/pipeline.md` | 阶段与调度的细节 |
| `docs/developer_reference/communication.md` | 控制面消息与 relay 数据传输 |

要接 TTS 模型再加 `docs/developer_reference/tts_model_integration.md`，里面是逐项 checklist 和生命周期规则。

想更深入的话，本仓库 `notes/` 下的代码研读系列从 [architecture.md](../architecture.md) 开始，
[modules.md](../modules.md) 建议从 `proto/` 目录读起——那是整个系统的词汇表。

---

> 下一篇：[02 · 配置环境](./02-environment.md)
