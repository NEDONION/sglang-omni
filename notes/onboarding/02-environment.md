# 02 · 配置环境

> 系列导航：[索引](./README.md) · [01 认识项目](./01-orientation.md) → **本篇** → [03 跑通测试](./03-tests.md)

这一步决定后面所有命令。选错路线会在装依赖时卡很久。

## 本页目录

- [一、按机器选路线](#一按机器选路线)
  - [1. 四条路线对照](#1-四条路线对照)
  - [2. 纯 CPU 非 Mac 机器的陷阱](#2-纯-cpu-非-mac-机器的陷阱)
- [二、拉仓库（所有平台通用）](#二拉仓库所有平台通用)
- [三、macOS Apple Silicon](#三macos-apple-silicon)
- [四、Linux + NVIDIA（Docker）](#四linux--nvidiadocker)
- [五、pre-commit（所有平台必做）](#五pre-commit所有平台必做)

---

## 一、按机器选路线

### 1. 四条路线对照

| 你的机器 | 安装方式 | 能做到哪一步 |
| --- | --- | --- |
| Linux + NVIDIA GPU | Docker 镜像 `hongccc/sglang-omni:dev` | 全部：本地起服务、模型端测试、benchmark、精度验证 |
| Apple Silicon Mac | 仓库自带 `./install.sh` | 框架逻辑 + 全套 CPU 单测 + 文档 + **MLX / Torch MPS 路径**（Qwen3-ASR 可真跑） |
| Intel Arc GPU | `pyproject_xpu.toml` + PyTorch XPU wheel | Qwen3-ASR / Qwen3-TTS / Qwen3-Omni 端到端（实验性），后端自动探测 |
| 纯 CPU 的 Linux / WSL | 见下条 | 文档、Rust Router、CI 脚本；Python 侧建议在 Docker 里做 |

Python 版本要求 `>=3.10,<3.13`，各路线都用 3.12。

### 2. 纯 CPU 非 Mac 机器的陷阱

`pyproject.toml` 里 `sglang`、`flash-attn-4`、`nixl-cu13`、`mooncake-transfer-engine-cuda13`
这些依赖的平台标记是「非 Apple arm64」，也就是说在一台没有卡的 Linux 上
`uv pip install -e .` 会把整套 CUDA 专用依赖拉进来，安装或 import 阶段都很容易失败。

务实的做法有两个：

1. 用 Docker 镜像但**不传 `--gpus`**，只在里面跑 CPU 单测。
   ⚠️ 这条路仓库文档没有明确背书，需要你自己验证。
2. 先从**不需要 import 本包**的方向切入——文档、`sglang_omni_router/` 的 Rust 代码、`.github/` 里的 CI 脚本。

如果你手上是 Apple Silicon Mac，直接走第三节的官方安装脚本，比什么都省事。

## 二、拉仓库（所有平台通用）

先在 GitHub 上 fork `sgl-project/sglang-omni`，然后：

```bash
git clone git@github.com:<你的账号>/sglang-omni.git
cd sglang-omni
git remote add upstream https://github.com/sgl-project/sglang-omni.git
git remote -v
```

`origin` 指向你的 fork，`upstream` 指向官方仓库。以后同步主线：

```bash
git fetch upstream && git rebase upstream/main
```

## 三、macOS Apple Silicon

前提：已经装好 [Homebrew](https://brew.sh)。安装脚本**刻意不会**帮你装 Homebrew，
也不会调用 `sudo`——因为 Homebrew 官方引导程序可能需要管理员授权并改动系统状态。

### 1. 一条命令装完

```bash
./install.sh
source .venv-apple/bin/activate
```

脚本是幂等的，重复跑安全。仅支持 macOS `arm64`。

### 2. 脚本到底做了什么

1. 创建（或复用）虚拟环境 `.venv-apple`，Python 3.12
2. 用 Homebrew 装 `ffmpeg@7` 和 `uv`（只有系统缺可用 git 时才装 `git`）
3. 从源码装 SGLang `v0.5.18` 及其 `all_mps` extra，缓存在 `~/.cache/sglang-omni/sglang-v0.5.18`
4. 用 `uv pip` 安装当前 checkout

SGLang 的可选 Rust 扩展在这条路径上用不到，会被跳过。

### 3. DYLD_LIBRARY_PATH

每次开新终端要跑音频相关服务前，都需要暴露 ffmpeg 的库路径：

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

建议直接写进 `~/.zshrc`。

**为什么钉在 `ffmpeg@7`**：Apple arm64 上钉的是 `torchcodec==0.11.1`，
它不支持 Homebrew 当前无版本号的 FFmpeg 9 formula。这不是疏漏，是刻意的。

### 4. 可选环境变量

| 变量 | 用途 |
| --- | --- |
| `SGLANG_OMNI_VENV` | 换虚拟环境路径 |
| `SGLANG_OMNI_EXTRAS` | 开可选 extras，如 `audar-tts,fun-cosyvoice3` |
| `SGLANG_SOURCE_DIR` | 换 SGLang 源码 checkout 位置 |
| `NONINTERACTIVE=1`（或 `--non-interactive`） | 关掉 Homebrew 自动更新，CI 用 |
| `UV_HTTP_TIMEOUT` / `UV_HTTP_RETRIES` | 网络慢或走代理时调大 |

常见失败就三种：Homebrew / uv 不在 `PATH` 上、Python 3.12 工具链不可用、忘了导 `DYLD_LIBRARY_PATH`。

## 四、Linux + NVIDIA（Docker）

官方推荐这条路——镜像里 UCX、flash-attn、SGLang、CUDA 都是预编译好的，省掉最痛的一段。

```bash
docker pull hongccc/sglang-omni:dev

docker run -it \
    --shm-size 32g \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    hongccc/sglang-omni:dev \
    /bin/zsh
```

容器内做源码可编辑安装：

```bash
pip install --upgrade pip && pip install uv
uv venv .venv -p 3.12
source .venv/bin/activate

# 开发用可编辑安装；只是想用的话把 -e . 换成 "sglang-omni==0.1.4"
uv pip install --prerelease=allow -v -e .
```

目前只发布了 `dev` 这一个 tag，它跟着 main 走。要可复现就按 digest 钉版本。

不用 Docker 的手工安装需要自己准备两个前置：**UCX 1.20.x**（带 CUDA + verbs）和
**flash-attn-4 `>=4.0.0b18`**，后者要和 `torch==2.13.0` 及 SGLang 0.5.18 的
`nvidia-cutlass-dsl` 4.6.2 pin 对齐。编译参数可以直接抄 `docker/Dockerfile`。

## 五、pre-commit（所有平台必做）

这是 CI 第一道关卡，本地不装的话 PR 一提就红。

```bash
pip install pre-commit
pre-commit install

# 提 PR 前跑一次全量；第一遍会自动改文件，改完要再跑一次并把改动 commit 掉
pre-commit run --all-files
```

挂着的 hook：

| 类别 | hook |
| --- | --- |
| 格式化 | `black-jupyter`、`isort`、`clang-format`（C++/CUDA）、`nbstripout` |
| 清理 | `autoflake`（删无用 import）、`trailing-whitespace`、`end-of-file-fixer` |
| 静态检查 | `ruff`（只对 `benchmarks/`、`docs/`、`examples/` 查 F401）、`check-ast`、`check-yaml`、`check-toml` |
| 安全与卫生 | `detect-private-key`、`check-added-large-files`、`check-merge-conflict`、`debug-statements` |
| 流程 | `no-commit-to-branch`——直接拦住你往 main 上提交 |

---

> 下一篇：[03 · 跑通测试](./03-tests.md)
