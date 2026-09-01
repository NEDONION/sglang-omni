# 05 · 找第一个任务

> 系列导航：[索引](./README.md) · [04 没有 GPU 怎么办](./04-no-gpu.md) → **本篇** → [06 提交 PR](./06-pull-request.md)

选题比写代码更影响你的第一次体验。

## 本页目录

- [一、从 issue 列表进](#一从-issue-列表进)
- [二、先查清楚有没有人在做](#二先查清楚有没有人在做)
- [三、真正的高产路线：自己在 Apple Silicon 上找 bug](#三真正的高产路线自己在-apple-silicon-上找-bug)
- [四、动手前先说一声](#四动手前先说一声)

---

## 一、从 issue 列表进

- **[`good first issue` 标签](https://github.com/sgl-project/sglang-omni/labels/good%20first%20issue)**
  ——挂着的几个都是 Qwen3-Omni 的性能调优，而且**大多已经有开放 PR 了**（例如 #1147 已被 #1177 认领）。
  先按下一节的方法查过再动手。
- **标题带 `[Bug]` 的**：范围清楚、有复现步骤、容易配单测，最适合第一个 PR。
- **标题带 `[Roadmap]` / `[Tracking]` 的**：是长期跟踪贴，里面常常列着已拆好的子任务，
  可以直接在贴里认领一条。
- **标题带 `[Perf]` / `[Runtime Profiling]` 的**：需要卡，暂时跳过。

## 二、先查清楚有没有人在做

这个仓库非常活跃：常年 90+ 个开放 PR，一个可做的 issue 往往几天内就被认领。
**贴标签的 `good first issue` 基本都已经有 PR 了**，光看 issue 列表会白忙一场。

认领前必查这三层，缺一不可：

```bash
# 1) 有没有指派给别人
gh issue view <N> --repo sgl-project/sglang-omni --json assignees

# 2) 有没有 PR 关联它（最关键：光 grep "#N" 会漏，要看 timeline 的交叉引用）
gh api "repos/sgl-project/sglang-omni/issues/<N>/timeline?per_page=100" \
  --jq '.[] | select(.event=="cross-referenced")
        | select(.source.issue.pull_request != null)
        | "\(.source.issue.number) \(.source.issue.state) \(.source.issue.title)"'

# 3) 评论里有没有人说"我来做"
gh issue view <N> --repo sgl-project/sglang-omni --comments
```

批量筛「未指派 + 无 PR 关联」的完整脚本：

```bash
# 拉所有未指派的开放 issue
for p in 1 2 3 4; do
  gh api "repos/sgl-project/sglang-omni/issues?state=open&per_page=100&page=$p" \
    --jq '.[] | select(.pull_request == null) | select((.assignees|length)==0)
          | [(.number|tostring), .title] | @tsv'
done > open_free.tsv

# 逐个查 timeline，挑出零 PR 关联的
while IFS=$'\t' read -r num title; do
  refs=$(gh api "repos/sgl-project/sglang-omni/issues/$num/timeline?per_page=100" \
    --jq '[.[] | select(.event=="cross-referenced")
           | select(.source.issue.pull_request != null)] | length')
  [ "$refs" = "0" ] && printf '%s\t%s\n' "$num" "$title"
done < open_free.tsv
```

## 三、真正的高产路线：自己在 Apple Silicon 上找 bug

Apple Silicon 路径是 Experimental，全仓库只有 Qwen3-ASR 有 MLX runner
（`sglang_omni/models/qwen3_asr/mlx/`）。**在 Mac 上真跑一遍，问题是自己会冒出来的**，
而且这类 bug 有卡的贡献者复现不了，不存在被抢的问题。

这条路已经被验证过：贡献者 jzheng17 的一串 PR 都是在 Apple M4 上跑 `Qwen3-ASR-0.6B-4bit`
量出来的——空转时 scheduler 轮询吃 CPU（#1809）、权重全在缓存里还是要跟 Hub 校验（#1808）、
`prompt` 字段没传进去（#1807）、`hotwords` key 没人 emit（#1874）。
每一个都是「跑起来 → 发现不对 → 开 issue → 提 PR」。

## 四、动手前先说一声

在 issue 下留言说明**你打算怎么改**、**需不需要人帮忙跑 GPU CI**，等一个回应再动手。

大改动尤其如此——先对齐方案，比写完两百行再被要求重构划算得多。
日常讨论在 [SGLang Slack](https://slack.sglang.io)。

按 README 的说法，项目欢迎的贡献方向是：推理系统、kernel、调度、阶段间通信、
模型 runner 与缓存效率、模型接入、benchmark、生产部署。

---

> 下一篇：[06 · 提交 PR](./06-pull-request.md)
