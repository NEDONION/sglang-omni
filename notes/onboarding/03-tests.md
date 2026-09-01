# 03 · 跑通测试

> 系列导航：[索引](./README.md) · [02 配置环境](./02-environment.md) → **本篇** → [04 没有 GPU 怎么办](./04-no-gpu.md)

在改任何代码之前，先确认基线是绿的。基线不绿，后面所有失败都无法归因。

## 本页目录

- [一、和 CI 同款的一条命令](#一和-ci-同款的一条命令)
- [二、三个 marker 的含义](#二三个-marker-的含义)
- [三、测试文件放哪](#三测试文件放哪有-ci-强制检查)
- [四、macOS 上的预期 skip](#四macos-上的预期-skip)

---

## 一、和 CI 同款的一条命令

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
  python -m pytest tests/unit_test -q -m "not benchmark and not accelerator"
```

这和 `.github/workflows/test.yaml` 里 `unit-test` 作业跑的是同一套东西
（CI 那边是 `pytest tests/ -v -m "not benchmark and not accelerator" -x`）。

CI 还包了一层 `.github/scripts/run_flaky_pytest.sh` 做重试，本地不需要。

## 二、三个 marker 的含义

定义在 `pyproject.toml` 的 `[tool.pytest.ini_options]`：

| Marker | 含义 | 无 GPU 时 |
| --- | --- | --- |
| `accelerator` | 需要加速器硬件 | 排除掉 |
| `benchmark` | 性能基准，可能要 GPU 且耗时长 | 排除掉 |
| `tts_stage(name)` | 选择文件内的某个 TTS benchmark CI 阶段 | 不涉及 |

你新写的测试如果依赖真实设备，**记得打 `@pytest.mark.accelerator`**——
否则会掉进无卡的 `unit-test` 作业里，在 CI 上炸给所有人看。

## 三、测试文件放哪（有 CI 强制检查）

`Test Layout` 这个 workflow 会遍历 `tests/` 下所有 `.py`，
不在四条合法 lane 里的文件一律判失败（只有 `tests/__init__.py` 和 `tests/utils.py` 例外）：

| 目录 | 用途 | 需要 GPU |
| --- | --- | --- |
| `tests/unit_test/` | 纯逻辑单测，用 `fixtures/` 里的 fake 对象搭场景 | 大部分不需要 |
| `tests/test_model/` | 真模型端到端 | 需要，还要权重 |
| `tests/test_ci/` | CI 专用套件 | 视套件而定 |
| `tests/utils/` | 测试辅助代码 | — |

新人最常犯的错就是随手把测试丢在 `tests/` 根目录。

`tests/unit_test/` 下按子系统分目录：`pipeline/`、`relay/`、`preprocessing/`、`sampling/`、
`quantization/`、`model_runner/`、`xpu/`、`client/`、`benchmarks/`，以及各模型自己的目录
（`qwen3_omni/`、`audar_tts/` 等）。**新测试放进对应子目录，不要在 `unit_test/` 根上堆文件。**

写单测时先翻 `tests/unit_test/fixtures/`：`pipeline_fakes.py`、`qwen_fakes.py`、`fish_fakes.py`
已经把「造一个假 stage / 假 scheduler / 假模型」这件事做好了，这是无卡测框架逻辑的关键基建。

## 四、macOS 上的预期 skip

`dots.tts` 和 ZONOS2 的文本归一化依赖 Pynini / OpenFst，在 Apple arm64 上没有可用 wheel，
`pyproject.toml` 里对此有显式的平台标记。相关测试被 skip 是**预期行为**，不是你环境坏了。

---

> 下一篇：[04 · 没有 GPU 怎么办](./04-no-gpu.md)
