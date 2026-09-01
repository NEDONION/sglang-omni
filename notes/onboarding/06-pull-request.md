# 06 · 提交 PR

> 系列导航：[索引](./README.md) · [05 找第一个任务](./05-first-task.md) → **本篇** → [07 常见坑](./07-pitfalls.md)

这个仓库的 CI 触发方式和大多数开源项目不一样，先看懂再提。

## 本页目录

- [一、六步流程](#一六步流程)
- [二、PR 描述怎么写](#二pr-描述怎么写)
- [三、GPU CI 的执行顺序](#三gpu-ci-的执行顺序)
- [四、用 slash command 选 CI 模型](#四用-slash-command-选-ci-模型)
- [五、Draft PR 不跑 CI](#五draft-pr-不跑-ci)

---

## 一、六步流程

1. **开分支**
   ```bash
   git checkout -b fix/asr-vocabulary-hint
   ```
   pre-commit 的 `no-commit-to-branch` 会拦住直接提 main。

2. **改代码，配单测**
   放进 `tests/unit_test/` 对应子目录，尽量做到无 GPU 可跑；需要设备的打 `@pytest.mark.accelerator`。

3. **本地自查**
   ```bash
   pre-commit run --all-files
   CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD python -m pytest tests/unit_test -q \
     -m "not benchmark and not accelerator"
   ```

4. **推分支、开 PR**，按模板填描述。

5. **等 maintainer 打 `run-ci` 标签**
   GPU CI 跑在自建 runner 上，不打标签不会触发。打上之后，只要标签还在，后续每次 push 都会重跑。

6. **过评审**
   `.github/CODEOWNERS` 会按你改的路径自动请求评审人：

   | 路径 | 评审人 |
   | --- | --- |
   | `/sglang_omni/pipeline` | @FrankLeeeee @shuaills |
   | `/sglang_omni/models` | @FlamingoPg @shuaills @ocss884 |
   | `/sglang_omni/relay` | @sleepcoo |
   | `/sglang_omni/serve` `/sglang_omni/client` | @shuaills |
   | `/sglang_omni/config` `/sglang_omni/proto` `/.github` `/docker` | @FrankLeeeee |

## 二、PR 描述怎么写

模板在 `.github/pull_request_template.md`，四段加一个 checklist：

| 段落 | 写什么 |
| --- | --- |
| Motivation | 这个 PR 的目的和要达成的目标 |
| Modifications | 具体改了什么 |
| Related Issues | 关联 issue，写 `Fixes #1872` 能自动关闭 |
| Accuracy Test | 影响模型侧代码（kernel、模型结构）时必须给精度结果 |
| Benchmark & Profiling | 预期影响性能时给 benchmark 和 profiling 结果 |

Checklist 四条：pre-commit 过、加了单测、更新文档 / docstring / 示例、按需给吞吐延迟与精度数据。

没有卡跑不了后两段的，在 PR 里直说，请 maintainer 协助。

## 三、GPU CI 的执行顺序

```
preflight → setup → pr-test → asr-ci → tts-ci → qwen3-omni-ci
```

- 只有 `setup` 会阻断所有下游；某个 benchmark 套件失败不影响后面的套件继续跑（用了 `always()`）。
- `pr-test` 就是 `test.yaml`，里面又分 `unit-test`（无 GPU）和 `unit-test-accelerator` 两个作业。
- 触发时机：`opened`、`labeled`、`synchronize`、`reopened`、`ready_for_review`。

## 四、用 slash command 选 CI 模型

直接在 PR 评论里发，可以指定这轮 CI 跑哪个 TTS / ASR 模型。两个 family 各选一个可以组合：

```
/tag-and-rerun-ci higgs         # TTS: higgs | moss | qwen3-tts
/tag-and-rerun-ci fun-asr       # ASR: fun-asr | qwen3-asr | whisper-asr
/tag-and-rerun-ci moss fun-asr  # 组合
```

不指定的话会随机选一个模型跑。

## 五、Draft PR 不跑 CI

即使打了 `run-ci` 标签，Draft 状态的 PR 依然会被跳过。想拿 CI 结果就先 **Ready for review**。

---

> 下一篇：[07 · 常见坑](./07-pitfalls.md)
