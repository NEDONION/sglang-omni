# 新人上手系列 · 索引

从零开始参与 SGLang-Omni 开发：认识项目 → 配环境 → 跑测试 → 找任务 → 提 PR。
全程假设读者第一次接触这个仓库，并**专门标注了没有 GPU 时该走哪条路**。

> 依据：本系列所有命令、路径与 CI 规则均取自 main 分支 v0.1.4 的实际文件
> （`install.sh`、`pyproject.toml`、`.pre-commit-config.yaml`、`.github/workflows/`、`docs/`）。
> 仓库演进后以代码为准。

---

## 阅读顺序

| # | 文档 | 讲什么 | 要多久 |
| --- | --- | --- | --- |
| 01 | [认识项目](./01-orientation.md) | 它解决什么问题、请求链路、目录职责对照 | 20 min |
| 02 | [配置环境](./02-environment.md) | 路线选择 → 平台安装 → pre-commit | 30–90 min |
| 03 | [跑通测试](./03-tests.md) | 基线验证、marker 语义、测试放哪 | 15 min |
| 04 | [没有 GPU 怎么办](./04-no-gpu.md) | 能做 / 不能做，以及免费 CI 信号 | 10 min |
| 05 | [找第一个任务](./05-first-task.md) | issue 怎么挑、怎么确认没人在做、无卡高产路线 | 15 min |
| 06 | [提交 PR](./06-pull-request.md) | 分支 → 评审 → `run-ci` 标签 → slash command | 15 min |
| 07 | [常见坑](./07-pitfalls.md) | 症状 → 原因 → 解法速查 | 按需查 |

---

## 分级目录

### 01 · 认识项目
- 一、它和 SGLang 的分工
- 二、请求走过的链路
- 三、目录与职责对照
- 四、模型代码的硬性约定
- 五、先读哪三份文档

### 02 · 配置环境
- 一、按机器选路线
  - 1. 四条路线对照表
  - 2. 纯 CPU 非 Mac 机器的陷阱
- 二、拉仓库（所有平台通用）
- 三、macOS Apple Silicon
  - 1. 一条命令装完
  - 2. 脚本到底做了什么
  - 3. DYLD_LIBRARY_PATH
  - 4. 可选环境变量
- 四、Linux + NVIDIA（Docker）
- 五、pre-commit（所有平台必做）

### 03 · 跑通测试
- 一、和 CI 同款的一条命令
- 二、三个 marker 的含义
- 三、测试文件放哪（有 CI 强制检查）
- 四、macOS 上的预期 skip

### 04 · 没有 GPU 怎么办
- 一、证据：CI 本身就是这么设计的
- 二、无卡能完整做完的六个方向
- 三、必须借卡的四类工作
- 四、四条免费 runner CI

### 05 · 找第一个任务
- 一、从 issue 列表进
- 二、先查清楚有没有人在做（三层查法 + 批量筛选脚本）
- 三、真正的高产路线：自己在 Apple Silicon 上找 bug
- 四、动手前先说一声

### 06 · 提交 PR
- 一、六步流程
- 二、GPU CI 的执行顺序
- 三、用 slash command 选 CI 模型
- 四、Draft PR 不跑 CI

### 07 · 常见坑
- 症状 → 原因与解法速查表

---

## 和代码研读系列的关系

`notes/` 下另有一组更深的代码研读笔记，建议**先读完本系列 01，再按需跳进去**：

| 文档 | 深度 | 什么时候读 |
| --- | --- | --- |
| [architecture.md](../architecture.md) | 整体架构与技术栈 | 本系列 01 读完还想深挖时 |
| [modules.md](../modules.md) | 模块拆分与职责，从 `proto/` 开始 | 准备动手改代码前 |
| [core_flows.md](../core_flows.md) | 启动 / 请求 / 流式 / 中止四条主线 | 定位一个 bug 在哪一层时 |
| [code_insights.md](../code_insights.md) | 核心代码逐段讲解与难点 | 改到具体文件时 |
| [learning_summary.md](../learning_summary.md) | 学习总结 | 回顾时 |
