# SGLang-Omni 笔记索引

本目录下的笔记分两条线：**上手线**（怎么参与开发）和**研读线**（代码本身怎么设计的）。
第一次接触这个仓库的话，从上手线 01 开始。

```
notes/
├── README.md              ← 你在这里
├── onboarding/            【上手线】从零到第一个 PR
│   ├── README.md              系列索引 + 三级目录
│   ├── 01-orientation.md      认识项目
│   ├── 02-environment.md      配置环境
│   ├── 03-tests.md            跑通测试
│   ├── 04-no-gpu.md           没有 GPU 怎么办
│   ├── 05-first-task.md       找第一个任务
│   ├── 06-pull-request.md     提交 PR
│   └── 07-pitfalls.md         常见坑
└── 【研读线】代码设计与实现
    ├── architecture.md        01 · 整体架构与技术栈
    ├── modules.md             02 · 模块拆分与职责
    ├── core_flows.md          03 · 核心流程
    ├── code_insights.md       04 · 核心代码讲解与难点
    └── learning_summary.md    05 · 学习总结
```

---

## 一、上手线 · [onboarding/](./onboarding/README.md)

目标：**跑起来、改得动、提得出 PR**。全程标注了没有 GPU 时该走哪条路。

| # | 文档 | 讲什么 |
| --- | --- | --- |
| 01 | [认识项目](./onboarding/01-orientation.md) | 它和 SGLang 的分工、请求链路、目录职责对照、模型代码约定 |
| 02 | [配置环境](./onboarding/02-environment.md) | 路线选择 · fork/clone · macOS `install.sh` · Docker · pre-commit |
| 03 | [跑通测试](./onboarding/03-tests.md) | 和 CI 同款的命令、三个 marker、测试文件放哪 |
| 04 | [没有 GPU 怎么办](./onboarding/04-no-gpu.md) | 能做的六个方向、要借卡的四类活、四条免费 CI |
| 05 | [找第一个任务](./onboarding/05-first-task.md) | issue 怎么挑、怎么确认没人在做、无卡高产路线 |
| 06 | [提交 PR](./onboarding/06-pull-request.md) | 六步流程、`run-ci` 标签、slash command、CODEOWNERS |
| 07 | [常见坑](./onboarding/07-pitfalls.md) | 症状 → 解法速查 + 一页命令速查 |

## 二、研读线

目标：**理解设计意图**。建议在上手线 01 之后按需切入。

| # | 文档 | 讲什么 |
| --- | --- | --- |
| 01 | [architecture.md](./architecture.md) | 整体架构与技术栈，多阶段推理为什么是个独立问题 |
| 02 | [modules.md](./modules.md) | 模块拆分与职责，从 `proto/`（系统词汇表）读起 |
| 03 | [core_flows.md](./core_flows.md) | 启动 / 请求 / 流式 / 中止四条主线 |
| 04 | [code_insights.md](./code_insights.md) | 核心代码逐段讲解与难点 |
| 05 | [learning_summary.md](./learning_summary.md) | 学习总结与可迁移的设计经验 |

---

## 三、两条线怎么配合

| 你现在想做什么 | 读哪儿 |
| --- | --- |
| 刚 clone 完，想跑起来 | onboarding [02](./onboarding/02-environment.md) → [03](./onboarding/03-tests.md) |
| 想知道这项目在解决什么问题 | onboarding [01](./onboarding/01-orientation.md) → [architecture.md](./architecture.md) |
| 准备动手改代码，但不知道改哪一层 | [modules.md](./modules.md) → [core_flows.md](./core_flows.md) |
| 定位一个 bug 在哪个阶段 | [core_flows.md](./core_flows.md) → [code_insights.md](./code_insights.md) |
| 没有卡，想知道能干什么 | onboarding [04](./onboarding/04-no-gpu.md) → [05](./onboarding/05-first-task.md) |
| 准备提 PR 了 | onboarding [06](./onboarding/06-pull-request.md) |
| CI 红了 / 装不上 | onboarding [07](./onboarding/07-pitfalls.md) |

---

> 上手线的所有命令、路径与 CI 规则均取自 main 分支 v0.1.4 的实际文件：
> `install.sh`、`pyproject.toml`、`.pre-commit-config.yaml`、`.github/workflows/`、`docs/`。
> 仓库演进后以代码为准。
