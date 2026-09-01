# 07 · 常见坑

> 系列导航：[索引](./README.md) · [06 提交 PR](./06-pull-request.md) → **本篇（终）**

按出现频率排序。症状 → 原因与解法。

---

## 环境类

| 症状 | 原因与解法 |
| --- | --- |
| 激活了虚拟环境但 import 失败 | Apple 安装脚本建的是 `.venv-apple`，不是 `.venv`。目录下同时存在两个环境时最容易搞混，先 `which python` 确认 |
| 起服务时报音频库找不到 | 忘了 `export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib..."`。写进 `~/.zshrc` 一劳永逸 |
| `No module named pytest` | `pytest` 是 `pyproject.toml` 的核心依赖，报这个说明当前虚拟环境不是用完整依赖装的。重跑安装，或临时 `pip install pytest pytest-asyncio` |
| `install.sh` 报 brew 找不到 | 脚本不会帮你装 Homebrew。自己从 [brew.sh](https://brew.sh) 装完再跑 |
| 装依赖卡在网络上 | `UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 ./install.sh` |
| 纯 CPU Linux 上 `-e .` 装不上 | 平台标记会拉进 CUDA 专用依赖，见 [02 · 一.2](./02-environment.md#2-纯-cpu-非-mac-机器的陷阱) |

## 测试类

| 症状 | 原因与解法 |
| --- | --- |
| 一堆测试在 macOS 上 skip | `dots.tts` / ZONOS2 依赖的 Pynini 在 Apple arm64 没有 wheel，是预期行为 |
| 本地过了 CI 挂了 | 本地漏了 `-m "not benchmark and not accelerator"`，或者新测试没打 `@pytest.mark.accelerator` |
| `Test Layout` 红了 | 测试文件放在了 `tests/` 根目录。挪进 `unit_test/`、`test_model/`、`test_ci/`、`utils/` 之一 |

## 提交与 CI 类

| 症状 | 原因与解法 |
| --- | --- |
| commit 被 hook 拒绝 | `no-commit-to-branch` 在拦你提 main。开分支 |
| pre-commit 一直失败 | 第一遍会自动修改文件（black / isort / autoflake），需要**再跑一次**确认全绿，并把自动改动一起 commit |
| PR 开着但没有 GPU CI 结果 | 要么是 Draft，要么是 maintainer 还没打 `run-ci` 标签。在 PR 里礼貌 ping 一下 |
| `Docs Check` 失败 | 新页面没登记进 `index.rst` 或对应 toctree；另外文档里用相对链接（`../get_started/installation.md`），不要用绝对 docs URL |
| rebase 冲突一大片 | 别直接改上游同步过来的源码文件加注释——这个仓库要长期跟 upstream 同步，注释写在 `notes/` 里 |

---

## 一页速查

```bash
# 装（macOS）
./install.sh && source .venv-apple/bin/activate
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

# 查
pre-commit run --all-files
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD python -m pytest tests/unit_test -q \
  -m "not benchmark and not accelerator"

# 同步主线
git fetch upstream && git rebase upstream/main
```

---

> 回到 [系列索引](./README.md)
