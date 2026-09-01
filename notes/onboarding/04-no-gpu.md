# 04 · 没有 GPU 怎么办

> 系列导航：[索引](./README.md) · [03 跑通测试](./03-tests.md) → **本篇** → [05 找第一个任务](./05-first-task.md)

结论先放这儿：**能参与，而且路子不窄**。这不是安慰式的说法——项目的 CI 结构本身就是这么设计的。

## 本页目录

- [一、证据：CI 本身就是这么设计的](#一证据ci-本身就是这么设计的)
- [二、无卡能完整做完的六个方向](#二无卡能完整做完的六个方向)
- [三、必须借卡的四类工作](#三必须借卡的四类工作)
- [四、四条免费 runner CI](#四四条免费-runner-ci)

---

## 一、证据：CI 本身就是这么设计的

`.github/workflows/test.yaml` 的 `unit-test` 作业里有这么一行：

```yaml
env:
  # note (Jingwen): Hide CUDA from the default unit-test process. Tests
  # that need an accelerator belong in the unit-test-accelerator job below.
  CUDA_VISIBLE_DEVICES: ""
```

把 CUDA 从默认 unit-test 进程里藏起来，需要加速器的测试属于下面单独的
`unit-test-accelerator` 作业。换句话说，**主单测套件是被设计成无 GPU 运行的**。

量级上：`tests/` 下 343 个测试文件，只有 49 个提到 accelerator。

## 二、无卡能完整做完的六个方向

| 方向 | 位置 | 说明 |
| --- | --- | --- |
| **框架逻辑** | `pipeline/` `scheduling/` `config/` `proto/` `relay/` | coordinator 生命周期、stage 消息路由、inbox/outbox 语义、配置解析。用 `fixtures/pipeline_fakes.py` 造场景，这是最大的一块 |
| **Apple Silicon 路径** | `platforms/apple.py`、`model_runner/mlx_model_worker.py`、`models/qwen3_asr/mlx/` | MLX 与 Torch MPS 还是 Experimental，全仓库只有 Qwen3-ASR 有 MLX runner。**这块必须有 Mac 才能验证**，有卡的人反而做不了 |
| **设备抽象层** | `sglang_omni/platforms/`（cuda / cpu / xpu / npu / rocm / musa / apple） | XPU / NPU / CPU 后端的分支逻辑，测试在 `tests/unit_test/xpu/test_device_layer.py` |
| **Rust Router** | `sglang_omni_router/` | 多 worker OpenAI 兼容网关：健康检查、就绪探测、生命周期、能力发现。纯 CPU 侧工程，独立 CI |
| **文档与 cookbook** | `docs/` | 全是 Markdown，本地 `cd docs && bash serve.sh` 预览（需要 `pandoc parallel retry` 和 `docs/requirements.txt`） |
| **Benchmark 工具链** | `benchmarks/` | 数据集处理、指标计算、回归检查，测试在 `tests/unit_test/benchmarks/` |

## 三、必须借卡的四类工作

| 类别 | 为什么 |
| --- | --- |
| 精度验证 | WER、说话人相似度这类指标；PR 模板要求模型侧改动附精度结果 |
| 性能数据 | 吞吐 / 延迟 benchmark、CUDA Graph 命中率这类遥测 |
| 模型端到端 | `tests/test_model/` 需要真权重和显存 |
| CUDA kernel | 写得了，验不了 |

**可行的办法**：在 issue 或 PR 里直说你没有卡，请 maintainer 帮忙跑一轮 CI。
反正 GPU CI 本来就必须由 maintainer 打 `run-ci` 标签才会触发，这在本仓库是常规操作。

## 四、四条免费 runner CI

这四个 workflow 跑在 GitHub 托管的 ubuntu runner 上，**不需要 `run-ci` 标签**，PR 一开就自动跑。
也就是说无卡贡献者照样能拿到真实 CI 信号：

| Workflow | 查什么 | Runner |
| --- | --- | --- |
| `Lint` | pre-commit 全量 + `playground/qwen-omni` 的 Node 播放状态测试 | `ubuntu-latest` |
| `Docs Check` | 用 pandoc 完整构建一遍文档，链接和 toctree 都会查 | `ubuntu-latest` |
| `Test Layout` | 测试文件是否放在四条合法 lane 里 | `ubuntu-latest` |
| `Rust Router` | Router 的 Rust 侧检查（按路径过滤触发） | `ubuntu-24.04` |

对照之下，`Omni CI`（`omni-ci.yaml` 及其调起的 `test.yaml`、`test-asr-ci.yaml`、
`test-tts-ci.yaml`、`test-qwen3-omni-ci.yaml`）跑在 `[self-hosted, h100]` 上，需要标签才触发。

---

> 下一篇：[05 · 找第一个任务](./05-first-task.md)
