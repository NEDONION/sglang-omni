# 02 · 模块拆分与职责

按「读的时候要先读哪个」排序，不按字母序。

---

## 一、契约层：`proto/`（0.8k 行，先读这个）

整个系统的词汇表。**所有跨进程消息都在这里定义，读完这个目录，后面所有代码的名词就都认识了。**

| 文件 | 内容 |
| --- | --- |
| `request.py` | `OmniRequest`（用户请求）、`StagePayload`（阶段间载荷）、`RequestState`、`RequestInfo` |
| `messages.py` | 控制面消息：`SubmitMessage` `DataReadyMessage` `DataAckMessage` `StreamMessage` `CompleteMessage` `AbortMessage` `ShutdownMessage` `ProfilerStart/Stop` |
| `admin.py` | 运维/RL 操作：`AdminOperation` / `AdminResult` |
| `kv_transfer.py` | KV cache 跨阶段搬运的握手消息 |
| `stage.py` | `StageInfo`：12 行，就是 `(name, control_endpoint)` |

两个必须记住的类型：

```python
@dataclass
class StagePayload:
    request_id: str
    request: OmniRequest    # 原始请求，一路带着走
    data: Any               # 本阶段的产物，模型自定义
```

```python
@dataclass
class DataReadyMessage:
    request_id: str
    from_stage: str
    to_stage: str
    data_ref: dict | None   # ← 只有引用，没有张量
    chunk_id: int | None    # 流式时的块序号
    is_done: bool           # 流结束信号
    error: str | None       # 流错误信号
```

`DataReadyMessage.to_dict()` 里塞了一堆断言：`is_done` 和 `error` 不能同时出现、信号消息不许带 `data_ref` 和 `chunk_id`。这是把「流控制协议」编码进了序列化函数——**契约在类型里，不在文档里**。

---

## 二、编排层：`pipeline/`（6.1k 行，核心中的核心）

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `stage/runtime.py` | 1843 | `Stage` —— 纯 IO 壳，全项目最重的一个类 |
| `coordinator.py` | 875 | 全局请求路由与状态机 |
| `stage_workers.py` | 911 | 进程 spawn、env 打补丁、stage 构造 |
| `mp_runner.py` | 815 | `MultiProcessPipelineRunner`，启动编排总入口 |
| `control_plane.py` | 446 | ZMQ socket 封装 + msgpack 序列化 |
| `replicas.py` | 278 | 副本拓扑与绑定策略（round-robin） |
| `runtime_config.py` | 289 | `prepare_pipeline_runtime()`：解析出 placement / process plan / endpoints |
| `tp_control.py` | 188 | 张量并行组的 leader/follower 控制 |
| `local_dispatch.py` | 88 | 同进程 Python 对象直传 |

### `Coordinator` 的五件事

1. 注册 stage 端点（`register_stage`）
2. 把新请求投给 `entry_stage`
3. 追踪请求状态：pending / running / completed / failed / aborted
4. 收集**多个**终态阶段的完成消息并合并（比如 `decode` 出文本、`code2wav` 出音频，两个都到齐才算完）
5. 广播 abort

有一条容易漏掉的性质：**Coordinator 对调度器实现完全不可知**，在张量并行的 stage 组里它只跟 rank 0 说话，其余 rank 对它不可见。

### `Stage` 的不变式

`Stage` 是个 IO 壳，它**不根据 scheduler 类型分支**。`SimpleScheduler`、`OmniScheduler`、各种流式调度器对它呈现的是同一个接口：

```python
class Scheduler:
    inbox:  Queue[IncomingMessage]   # type: new_request | stream_chunk | stream_done
    outbox: Queue[OutgoingMessage]   # type: result | stream | error
    def start(self); def stop(self); def abort(self, request_id)
```

这条不变式是整个可扩展性的地基：加一种新调度器，`Stage` 一行都不用改。

---

## 三、通信层：`comm/` + `relay/`（5.8k 行）

**`comm/` 决定「怎么送」，`relay/` 负责「送」。**

| 文件 | 职责 |
| --- | --- |
| `comm/router.py` | `CommRouter`：按局部性和 placement 推导传输方式，**没有公开的后端选择器** |
| `comm/engine.py` | `CommEngine`：面向 Stage 的门面，1136 行，含发送 worker、pending 追踪 |
| `comm/stage_io.py` | 张量的打包/拆包：递归抽出 `payload.data` 里的张量，替成占位符，剩下的 pickle |
| `comm/data_ref.py` | `DataRef` / `TransportKind` 类型定义 |
| `relay/base.py` | `Relay` 抽象基类 + `RELAY_REGISTRY` 装饰器注册表 |
| `relay/{shm,cuda_ipc,nccl,nixl,mooncake}.py` | 五种后端实现，`cuda_ipc.py` 最重（1209 行，自带发送侧显存池） |

### 传输选择规则（`CommRouter.outbound`）

| 传输 | 触发条件 |
| --- | --- |
| `local_object` | 源和目标在同一个 OS 进程，且载荷可直传 |
| direct PyTorch CUDA IPC | 同 placement、不同进程，且能证明 CUDA ordinal 兼容 |
| `cuda_ipc`（池化） | 同节点 GPU 边，但不满足直传条件 |
| `shm` | 同节点、非 GPU-to-GPU |
| `mooncake` | 跨节点 |

**这是推导出来的，不是配出来的。** `CommConfig` 只能调 slot 大小、credit 数、Mooncake 连接参数，不能指定用哪个后端。

### `Relay` 接口只有四个方法

```python
class Relay:
    async def put_async(tensor, request_id=None, dst_rank=None) -> RelayOperation
    async def get_async(metadata, dest_tensor, request_id=None) -> RelayOperation
    def cleanup(request_id): ...
    def close(): ...
```

后端只需要「搬一个扁平张量缓冲区，返回一段另一端能用来 get 的元数据」。**格式是后端中立的**——这就是为什么加一个新传输后端（比如 RDMA）只要实现这四个方法。

---

## 四、调度层：`scheduling/`（8.2k 行）

一个 stage 配一个 scheduler，按 stage 的负载形状选：

| 调度器 | 行数 | 用在哪 | 有没有 KV cache |
| --- | --- | --- | --- |
| `OmniScheduler` | 2677 | 自回归阶段（thinker / talker / tts_engine） | 有（复用 SGLang） |
| `SimpleScheduler` | 327 | preprocessing、encoder、decode、非流式 vocoder | 无 |
| `ThreadedSimpleScheduler` | 226 | CPU 密集的简单阶段 | 无 |
| `StreamingSimpleScheduler` | 551 | 需要处理流式输入块的简单阶段 | 无 |
| `StreamingVocoderBase` | 513 | 流式声码器的共享基类 | 无，但有 per-request 累积状态 |
| `DllmScheduler` | 363 | Diffusion LLM 阶段 | 有 |

配套设施：

- `bootstrap.py` — `create_sglang_infrastructure()`：起一个 SGLang worker，返回 `(model_worker, tree_cache, req_to_token_pool, token_to_kv_pool_allocator, model_config)`
- `sglang_backend/server_args_builder.py` — `build_sglang_server_args()`
- `generation_batch_policy.py` — prefill 准入门（16 个请求就绪，或最老的等满 40ms 就放行）
- `reference_encoder.py` / `pre_lm_encoder.py` — 参考音频编码服务的共享骨架
- `speaker_cache.py` / `stage_cache.py` — 阶段级输出缓存

---

## 五、模型执行层：`model_runner/`（3.6k 行）

统一的 AR 前向路径：

```
ForwardBatch 构造 → before hooks → model.forward() → post hooks → 采样 → 输出提取
```

| 文件 | 职责 |
| --- | --- |
| `base.py`（1007） | `ModelRunner` 基类：ForwardBatch、采样、logit 后处理、`execute_launch`/`execute_resolve` 两段式异步 decode |
| `sglang_model_runner.py`（491） | 桥到 SGLang 的执行；`_register_omni_model()` 是新模型必须改的那一行 |
| `thinker_model_runner.py`（490） | Qwen-omni thinker 类：前向前注入多模态 embedding（image/video/audio/deepstack） |
| `model_worker.py`（555） | worker 封装 |
| `mlx_model_worker.py`（277） | Apple Silicon 的 MLX 路径 |
| `weight_checker.py` | 权重校验（RL 场景下热更新后用） |

设计文档里还提到一个 `FeedbackARModelRunner` 角色：**下一步 decode 依赖上一步在同一个 runner 内产生的反馈**（Qwen3-Omni talker、Fish S2-Pro 都是这形状），抽成 `FeedbackStrategy` 策略对象。注意它的边界——**只覆盖自包含的反馈环**，跨调度器、走 relay 的反馈不在范围内。

---

## 六、服务层：`serve/`（8.8k 行）+ `client/`（1.4k 行）

### 路由表

| 路由 | 文件 |
| --- | --- |
| `POST /v1/chat/completions` | `openai_api.py` |
| `POST /v1/audio/speech` `/speech/batch` | `speech_service.py` |
| `WS /v1/audio/speech/stream` | `speech_ws.py` |
| `POST /v1/audio/transcriptions` | `transcriptions.py` + `transcription_adapters/` |
| `POST /v1/audio/translations` | `translations.py` |
| `WS /v1/realtime` | `realtime/` |
| `GET/POST /v1/audio/voices` | `speech_voices.py`（上传音色） |
| `GET /health` `/v1/models` | `openai_api.py` |
| `POST /pause_generation` `/update_weights_from_distributed` … | RL rollout 控制面，见 `docs/developer_reference/rl_admin_control.md` |
| `POST /start_profile` `/stop_profile` | `launcher.py` 挂载 |

### 两个入口的区别（最容易搞混的一点）

- `create_app(client, ...)` —— **只**建 FastAPI app 和注册路由。不编译流水线、不起运行时、不建 coordinator、不挂 profiling、不跑 uvicorn。用于「我已经有一个活的 Client，想自己嵌 HTTP 层」。
- `launch_server(pipeline_config, ...)` —— 完整生命周期：编译配置 → 起 `MultiProcessPipelineRunner` → 建 `Client` → 建 app → 挂 profiling → 跑 uvicorn → 关停时停运行时。

`Client` 的职责是**请求形态转换 + 结果聚合**：`GenerateRequest → OmniRequest` 下去，文本片段、音频块、usage、finish_reason 上来聚合成 `CompletionResult`。

---

## 七、配置层：`config/`（4.0k 行）

| 文件 | 职责 |
| --- | --- |
| `schema.py`（783） | `PipelineConfig` / `StageConfig` / `EngineStageConfig` / `FactoryArgs` |
| `path.py`（867） | 点号路径语言的解析与写入（`--vocoder.factory.dtype` 这种） |
| `sources.py`（497） | 来源合并：默认值 < YAML < `shared` 展开 < 广播 flag < 显式点号 flag |
| `topology.py`（464） | 编译逻辑进程计划，校验进程内边、GPU 共置 |
| `placement.py` / `resolver.py` / `runtime.py` | 设备放置、字符串导入解析、运行时视图 |
| `provenance.py`（146） | 记录每个值是谁设的——`sgl-omni config explain PATH` 靠它 |

三个消费者组：

| 组 | 谁消费 | 例子 |
| --- | --- | --- |
| stage 顶层 | 父进程做放置和进程规划 | `gpu` `tp_size` `process` `gpu_memory_fraction` |
| `engine.*` | SGLang `ServerArgs`（仅 `EngineStageConfig`） | `mem_fraction_static` `max_running_requests` |
| `factory.*` | stage 工厂函数的签名 | `dtype` `max_seq_len` `enable_async_decode` |

**未声明的 key 原样透传**，由消费方裁决。工厂不接受的参数会在 stage 构造时报错，而不是静默忽略。

---

## 八、模型接入层：`models/`（95k 行，占 66%）

20 个模型。目录约定：

```
models/<name>/
├── config.py            # PipelineConfig 子类 + stages 列表 + EntryClass（必须）
├── stages.py            # StageConfig.factory_path 指向的工厂函数
├── request_builders.py  # 阶段间 payload 变换、路由函数、扇入合并
├── payload_types.py     # 类型化的模型专属状态
├── routing.py           # 可选：数据驱动的路由
├── callbacks.py         # 可选：反馈回调
├── sglang_model.py      # SGLang 侧的模型类
└── components/          # 模型模块、processor、vocoder、adapter
```

`registry.py` 用 `pkgutil.iter_modules` 遍历 `models/` 的每个子包，import `<pkg>.config`，取 `EntryClass`，把它的 `architecture` ClassVar 和 HF `config.json::architectures[0]` 对上。**没有任何手写清单要维护**——加模型就是加一个目录。

代码量排序（前五）：qwen3_omni 13.5k、ming_omni 10.7k、moss_tts 8.3k、qwen3_tts 7.2k、ming_tts 6.7k。

---

## 九、周边

| 目录 | 说明 |
| --- | --- |
| `platforms/` | `cuda` `rocm` `xpu` `npu` `musa` `apple` `cpu` 七个后端的能力探测与差异隔离 |
| `mps/` | NVIDIA Multi-Process Service 管理（多进程共享一张卡时的 GPU 时间片），**不是 Apple Metal** |
| `profiler/` | torch trace + 请求级事件 JSONL；`python -m sglang_omni.profiler <dir>` 出 timeline/stage/hop 报表 |
| `preprocessing/` | 音/视频/图像/文本的规范化、缓存键、资源连接器 |
| `diagnostics/` | GPU 自检 |
| `vendor/` | 对 SGLang 的猴子补丁（`sglang_layers_patch`、`sglang_server_args`） |
| `sglang_omni_router/` | 多 worker 前门。Rust（`lifecycle.rs` `server/`）+ Python（`supervisor.py` `selector.py` `voice_routing.py` `admission_shm.py`） |
| `benchmarks/` | `eval/` 下 16 个基准脚本 + `benchmarker/` `dataset/` `metrics/` |
| `examples/configs/` | 25+ 个可直接 `--config` 的 YAML |
