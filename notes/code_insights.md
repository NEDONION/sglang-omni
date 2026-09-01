# 04 · 核心代码讲解与难点

每节：**贴原代码 → 加中文注释 → 讲设计意图 → 指出边界**。

> 说明：注释写在这里而不是直接改源码。这个分支要长期跟 upstream 同步，动源码会让每次 rebase 都冲突。需要给某个具体文件加行内注释时单独开分支做。

---

## 一、不继承的继承：`OmniScheduler.__getattr__`

`sglang_omni/scheduling/omni_scheduler.py:707`

```python
def __getattr__(self, name: str):
    """在上游 SGLang Scheduler 类上查找方法。

    这样我们拿到完整的调度 MRO（批次选择、结果处理、显存检查……），
    却不用继承 SGLangScheduler。
    """
    # grammar_* 是上游会访问但我们不支持的属性，给出空实现避免 AttributeError
    if name == "grammar_queue":
        value = []
        self.__dict__[name] = value          # 写进实例字典，下次不再走 __getattr__
        return value
    if name == "grammar_backend":
        self.__dict__[name] = None
        return None

    try:
        attr = getattr(_Upstream, name)      # ← 从「类」上取，拿到的是未绑定函数
    except AttributeError:
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute {name!r}"
        ) from None

    # 关键一步：把未绑定方法绑到「本实例」上，
    # 于是上游代码里所有的 self.xxx 访问的都是 OmniScheduler 的状态
    if callable(attr):
        return types.MethodType(attr, self)
    return attr
```

### 这在干什么

`OmniScheduler` **不是** SGLang `Scheduler` 的子类。但当你调用 `omni_scheduler.get_next_batch_to_run()`：

1. 实例字典和 `OmniScheduler` 类上都没有 → 触发 `__getattr__`
2. 从上游 `Scheduler` 类上取到未绑定的函数对象
3. `types.MethodType(attr, self)` 把它绑到 `OmniScheduler` 实例
4. 上游函数体内部调 `self.get_new_batch_prefill()` —— 又走一遍上面的流程

结果是：**上游的整条调用链都在 SGLang-Omni 的状态上运行**，而 SGLang-Omni 自己定义的方法（`recv_requests`、`process_input_requests`、`run_batch`、`send_to_tokenizer`）因为在实例/类上找得到，永远优先。

### 为什么不直接继承

继承的问题是**版本耦合**：上游 `Scheduler.__init__` 会初始化一大堆 SGLang-Omni 用不上的东西（tokenizer manager、detokenizer IPC、grammar backend、metrics collector），而且 SGLang 每次改 `__init__` 签名，子类就得跟着改。用 `__getattr__` 组合：只借方法，不借构造过程；`_init_upstream_scheduler_components()` 里自己挑需要的组件初始化。

### 边界在哪

- **overlap scheduling 明确不支持**。`OmniScheduler._event_loop_overlap` 直接拒绝运行，原因是 `Req.inflight_middle_chunks` 的自减在那个循环上会滞后一个 iteration。
- `__getattr__` 只在**属性查找失败时**触发，所以任何上游用 `super()`、`type(self)`、`isinstance` 做的事情都可能出问题。代码里为此打了一批 `_init_upstream_compat_flags` 的兼容标志位。
- 调试成本高：栈里会出现上游文件的行号，但 `self` 是本地类型，IDE 跳转基本失效。

**这个技巧值得学，但要清楚代价**：它换来的是「上游随便升级，我只维护自己重写的四个方法」，代价是「静态分析全废」。项目愿意付这个代价，是因为 SGLang 版本被死死 pin 在 0.5.18。

---

## 二、把协议编码进序列化函数：`DataReadyMessage`

`sglang_omni/proto/messages.py:19`

```python
def to_dict(self) -> dict[str, Any]:
    _require_str(self.request_id, "request_id")
    _require_str(self.from_stage, "from_stage")
    _require_str(self.to_stage, "to_stage")
    _require_bool(self.is_done, "is_done")

    # 约束 1：结束和错误互斥——一个流不能既正常结束又报错
    if self.is_done and self.error is not None:
        raise ValueError("stream signal cannot be both done and error")

    # 约束 2：信号消息（done / error）是纯控制消息，不许携带数据
    if self.is_done or self.error is not None:
        if self.data_ref is not None:
            raise ValueError("stream signal must not carry data_ref")
        if self.chunk_id is not None:
            raise ValueError("stream signal must not carry chunk_id")

    # 约束 3：数据消息必须带 dict 形态的 data_ref
    elif not isinstance(self.data_ref, dict):
        raise TypeError(...)
```

`from_dict()` 里**同样的三条约束又写了一遍**。看着像重复，其实不是：`to_dict` 防的是本进程写出坏消息，`from_dict` 防的是对端发来坏消息。一个是自检，一个是不信任边界。

**设计思想**：这是「让非法状态不可表示」的工程版本。`DataReadyMessage` 一个类承担了三种角色（数据消息、结束信号、错误信号），用 dataclass 表达不出「这三种角色字段互斥」，就在序列化边界上强制。跨进程系统里这类校验的价值极高——**没有它，一个字段错误会变成对端某个 `None` 解引用，栈完全指错地方。**

---

## 三、声明式拓扑：读懂一个 `StageConfig`

`sglang_omni/models/qwen3_omni/config.py:135`（略作精简）

```python
return EngineStageConfig(              # Engine 版才有 engine.* 组（映射到 SGLang ServerArgs）
    name="thinker",
    process=process,                   # 进程归属：同名 = 同一个 OS 进程
    factory_path=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
    factory=FactoryArgs(               # factory.* 组：传给上面那个工厂函数的签名
        max_seq_len=8192,
        enable_async_decode=True,      # 一步前瞻，把 host 侧结果处理和下一次 GPU decode 重叠
    ),
    gpu=gpu,

    # ── 扇入：语音模式下 thinker 自己承担聚合，省掉一个 mm_aggregate 阶段 ──
    wait_for=["preprocessing", "image_encoder", "audio_encoder"],   # 静态上界
    wait_for_fn=f"{_PKG}.request_builders.resolve_mm_aggregate_wait_sources",  # 运行时收窄
    merge_fn=f"{_PKG}.merge.merge_for_thinker",

    # ── 扇出：完整 payload 走 next，逐块 hidden states 走 stream_to ──
    next="decode",
    stream_to=["talker_ar", "decode"] if speech_enabled else ["decode"],
    route_fn=f"{_PKG}.request_builders.resolve_thinker_next_stages",
    stream_done_to_fn=f"{_PKG}.request_builders.resolve_thinker_stream_done_targets",

    # ── 每条出边一个投影函数：下游只拿它需要的字段 ──
    project_payload={
        "decode": f"{_PKG}.request_builders.project_thinker_to_decode",
    },
)
```

### 四个必须分清的概念

| 字段 | 含义 | 什么时候发 |
| --- | --- | --- |
| `next` | 完整 payload 的下游 | 本阶段全部算完 |
| `stream_to` | 流式块的下游 | 边算边发 |
| `project_payload` | 每条边的投影函数 | 发之前调，裁剪字段 |
| `wait_for` | 扇入等待源 | 收齐才进 scheduler |

`project_payload` 这层容易被忽略但很关键。文档里有一句约束：*"Request and control objects should retain parameters needed downstream, but they should not retain consumed bulk media across later stage hops."* —— **谁把原始媒体转成了规范化状态，谁负责释放那份引用**。否则一个 30 帧的视频请求会一路被拖到 code2wav 阶段。

### `_PKG` 字符串路径而非直接 import

所有 `factory_path` / `merge_fn` / `route_fn` 都是**字符串**，运行时用 `import_string()` 解析。原因：配置对象要能被 pickle 传到 spawn 出来的子进程里，而函数对象跨 spawn 不可靠；而且这让 stage 进程只 import 自己需要的模块——preprocessing 进程不用把 CUDA 模型代码拉进来。

---

## 四、零维护的模型注册：`registry.py`

```python
package = importlib.import_module(package_name)          # sglang_omni.models

for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
    if not ispkg:                                        # 只要子包，跳过 registry.py 这类模块
        continue
    try:
        importlib.import_module(name)
    except Exception as exc:
        if strict:
            raise
        logger.warning(f"Ignore import error when loading {name}: {exc}")
        continue                                         # ← 关键：一个模型的依赖缺失不影响其他模型

    config_module = importlib.import_module(f"{name}.config")
    if not hasattr(config_module, "EntryClass"):
        raise AssertionError(f"Config module {name}.config must have an EntryClass")

    for arch in _iter_config_architectures(config_module.EntryClass):
        # arch 来自 EntryClass.architecture（+ architecture_aliases）
        # 之后按 HF config.json::architectures[0] 匹配
        ...
```

**`continue` 那一行是这个设计成立的前提。** 二十个模型有二十套依赖（`dots.tts` 要 Pynini、`fun_cosyvoice3` 要 lightning、Apple 上一堆装不了），如果一个 import 失败就整体崩溃，这个自动发现机制根本没法用。它选择的是：**默认宽容 + `strict=True` 给测试用**。

代价也很明确：**依赖装错时你得到的是一条 WARNING 日志和一句"unsupported architecture"，而不是真正的 ImportError**。排查这类问题的正确姿势是用 `strict=True` 重跑一次注册。

---

## 五、批处理等待窗口：延迟与吞吐的显式取舍

`sglang_omni/scheduling/simple_scheduler.py:106`

```python
def _collect_batch(self, first_msg: IncomingMessage) -> list[IncomingMessage]:
    batch = [first_msg]
    if self._batch_fn is None or self._max_batch_size <= 1:
        return batch                                     # 不支持批处理，直接走

    deadline: float | None = (
        time.monotonic() + self._max_batch_wait_s
        if self._batch_wait_when_idle                    # ← 这个开关决定队列空时等不等
        else None
    )
    while len(batch) < self._max_batch_size:
        try:
            msg = self.inbox.get_nowait()                # 先非阻塞捞
        except _queue_mod.Empty:
            if deadline is None:
                break                                    # 不等：单请求零额外延迟
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = self.inbox.get(timeout=remaining)   # 等：赌后面还有请求能凑批
            except _queue_mod.Empty:
                break
```

**这是 TTFT 和吞吐之间那个开关的物理位置。** `_batch_wait_when_idle=True` 时，一个孤零零的请求也要把整个窗口等满才开始算。Qwen3-Omni 的两个 encoder 工厂把窗口硬编码成 `max_batch_wait_ms=50`（`models/qwen3_omni/stages.py:852` 和 `:922`），这 50 ms 直接坐在 TTFT/TTFA 的关键路径上——它是为了让 c16 的视频基准能凑批而选的，从来没被扫过参数。

社区把这件事开成了 good first issue [#1147](https://github.com/sgl-project/sglang-omni/issues/1147)：先把它从工厂硬编码改成可配置（默认不变），再在 c1/c8/c16 上扫 {0,1,5,10,50}。

**通用教训**：任何「为了凑批而等」的窗口，都必须能被配置，且必须有低并发下的快路径。硬编码的等待窗口是分布式推理系统里最常见的隐性延迟来源。

---

## 六、有界集合：跨进程系统里的迟到消息

`sglang_omni/pipeline/stage/runtime.py:1775`

```python
@staticmethod
def _record_bounded_request_id(ids: set[str], request_id: str) -> None:
    ids.add(request_id)
    if len(ids) > 10000:              # 上限
        excess = len(ids) - 5000      # 一次裁到水位线，而不是每次删一个
        it = iter(ids)
        to_remove = [next(it) for _ in range(excess)]
        ids -= set(to_remove)
```

看起来平平无奇，但它解决的是分布式系统的一个真问题：**abort 广播出去之后，在途的流块还会继续到达**。接收方必须能识别「这个 request 已经死了，丢掉这块，并且不要因为这块又把状态重建出来」。所以要记住已中止的 id。

但记忆不能无界——一个跑几周的服务会攒下几百万个 id。这里的方案是**上限 10000、触发时一次裁到 5000**（摊销 O(1)，而不是每次插入都淘汰一个）。

代价写在名字里：`set` 迭代顺序不是插入顺序，所以被裁掉的**不一定是最老的**。这是刻意的取舍——用 `OrderedDict` 能保序但更重，而这里只需要「近期的大概率还在」。

同类常量在 `omni_scheduler.py` 里还有一组：`_ABORTED_REQUEST_ID_LIMIT`、`_COMPLETED_REQUEST_ID_LIMIT`、`_PENDING_STREAM_REQUEST_LIMIT`，都是 10000/5000。

---

## 七、控制消息先发，再等传输完成

`docs/developer_reference/communication.md` 的原话是：

> The control-before-wait ordering is important for NIXL and other credit-based backends. If the sender waited for completion before notifying the receiver, the receiver would never start the read that releases the sender's credit.

用图说清这个死锁：

```
❌ 错误顺序                          ✅ 正确顺序
发送方: put() → await 完成           发送方: put() → 发 DataReadyMessage → await 完成
        ↑ 永远等不到                          ↓
        │                            接收方: 收到 → get() → 释放 credit
        │                                    ↓
接收方: 什么都没收到，不会 get       发送方: await 返回
        └─────── 死锁 ───────┘
```

**这类顺序约束是所有基于 credit / 流控的传输层的共同陷阱**：完成信号依赖对端动作，而对端动作依赖你先通知它。凡是「A 等 B，B 等 A 的通知」的结构，通知必须先于等待。

---

## 八、`local_object` 的只读契约

同进程的两个 stage 之间不做序列化，直接传 Python 对象引用。文档给的规则：

> Receivers must treat the payload, nested data containers, tensors, stream chunks, and metadata as read-only. The object must also stay valid for the receiver's scheduler queue lifetime. Senders and projection functions must not mutate or recycle objects after dispatch.

代码里为此有一套防御：

```python
def _is_isolated_projected_payload(...)      # 检查投影出来的 payload 是不是独立容器
def _shares_mutable_container(original, projected)  # 递归比对可变容器的 id
def _collect_mutable_container_ids(...)
def _can_send_full_payload_locally(...)
```

规则是：**单目标同进程边可以直传完整 payload；扇出时，只有每个投影产物都是带自己 `data` 容器的独立 `StagePayload` 才允许**，否则下游会共享可变状态。张量叶子可以有意共享，但必须当只读。

这是一个典型的「为了性能开后门，然后用运行时检查把后门守住」的模式。**值得注意的是它守的是可变容器，不守张量**——张量共享是被允许甚至鼓励的，因为拷贝一份几百 MB 的 hidden states 才是真正的浪费。

---

## 九、扇入的静态上界 + 运行时收窄

这个模式在项目里出现了至少四次，值得单独拎出来：

| 静态字段 | 运行时函数 | 收窄什么 |
| --- | --- | --- |
| `wait_for` | `wait_for_fn` | 纯文本请求不等 `image_encoder` |
| `next` | `route_fn` | 不出音频的请求不走 `talker_ar` |
| `stream_to` | `stream_done_to_fn` | 流结束信号只发给真的开了流的下游 |
| `terminal_stages` | `terminal_stages_fn` | 纯文本请求只等 `decode`，不等 `code2wav` |

**为什么要两层？** 静态字段是给**编译期**用的——`config/topology.py` 要靠它算进程拓扑、校验进程内边、建 relay 通道。运行时函数是给**每个请求**用的——同一个 pipeline 要同时服务纯文本和多模态请求。

如果只有静态字段，纯文本请求会永远卡在 `mm_aggregate` 等一个永远不来的 `image_encoder` 输出；如果只有运行时函数，启动时根本不知道要建哪些通道。

---

## 十、启动时的四个顺序约束

`pipeline/mp_runner.py:489` 的 `start()` 里，每一步的顺序都是被 bug 逼出来的：

```python
port = _find_available_port(host, port)      # ① 端口先占，别加载完 60GB 权重才发现被占

prep = prepare_pipeline_runtime(config)      # ② 先解析计划，再动进程
groups = _build_stage_groups(...)

await self._coordinator.start()              # ③ Coordinator 先起来收消息

# note (Jiaxin Deng): daemons must predate the first CUDA init
if self._config.mps != "off":                # ④ MPS daemon 必须早于第一次 CUDA init
    self._mps = create_for_pipeline(...)
    await self._mps.start()

for group in self._groups:
    group.spawn(ctx)                         # ⑤ 现在才 spawn

await asyncio.gather(*(g.wait_ready(timeout) for g in self._groups))

for group in self._groups:                   # ⑥ 就绪之后才注册，避免请求打到没加载完的进程
    for stage_name, endpoint in group.stage_control_endpoints.items():
        self._coordinator.register_stage(stage_name, endpoint)
```

异常处理写了两层——`except Exception` 和 `except BaseException`——保证 Ctrl-C（`KeyboardInterrupt` 不是 `Exception`）也能把 MPS daemon 收干净。**在有外部守护进程的系统里，只 catch `Exception` 就等于漏掉了用户中断这条最常见的退出路径。**
