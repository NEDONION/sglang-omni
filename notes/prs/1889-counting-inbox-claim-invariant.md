# PR #1889 · 给调度器的一条记账规则立个绊线

| | |
| --- | --- |
| **PR** | https://github.com/sgl-project/sglang-omni/pull/1889 |
| **关联 issue** | [#1680](https://github.com/sgl-project/sglang-omni/issues/1680) |
| **分支** | `test/counting-inbox-claim-invariant` |
| **类型** | 测试 + 文档（零行为改动） |
| **改动量** | 2 个文件，+60/-1（2 个 commit）|
| **状态** | 🟡 待评审 |
| **开始时间** | 2026-09-02 |

---

## 一、这事从哪来的

我想找个 issue 练手，两个硬条件：**没人在做**，而且**没有 GPU 也能做完**。

先按常规办法筛：95 个未指派的开放 issue。然后发现光看「有没有指派」根本不够——这个仓库常年 90+ 个开放 PR，很多 issue 早有人在提 PR 了，只是没人去点 assign。于是又加一层：查 GitHub timeline 的交叉引用，看有没有 PR 关联它。

筛完只剩 24 个真正零关联的。再排掉需要 GPU 的（6 个 profiling 系列、8 个要复现的 GPU bug、5 个大 Roadmap……），**能无卡做的只剩这一个**。

好处是范围极小、纯 CPU、逻辑清楚；坏处是价值有限，评审可能会问「为什么现在做」。作为第一个 PR 熟悉流程，够用了。

> 这套筛法写进了 [onboarding/05-first-task.md](../onboarding/05-first-task.md)：
> 光 grep `#编号` 会漏，因为有的 PR 用完整 URL 引用 issue，必须查 timeline。

## 二、背景：`_CountingInbox` 到底在解决什么

代码在 [`sglang_omni/scheduling/threaded_simple_scheduler.py`](../../sglang_omni/scheduling/threaded_simple_scheduler.py)。

调度器收到一个「取消请求」（abort）时，要分三种情况：

| 情况 | 怎么判断 | 做什么 |
| --- | --- | --- |
| 请求**正在跑** | `_pending` 里有它 | 直接 cancel |
| 请求**还没跑但够得着** | `is_reachable()` 为真 | 记个「墓碑」，等调度循环取到它时丢弃 |
| **完全不认识**它 | 以上都不是 | 记个「预防性 abort」，等它将来到达时丢弃 |

麻烦出在**第二种和第三种之间那道缝**。调度循环长这样：

```python
msg = self.inbox.get(timeout=0.1)      # ← 请求已经离开队列了
...
with self._lock:
    try:
        ...
        self._pending[request_id] = future     # ← 到这里才算「正在跑」
    finally:
        self.inbox.release_claim(request_id)
```

从队列取出来、到登记进 `_pending` 之间，这个请求**既不在队列里，也不在正在跑里**——悬在半空。

如果这时来一个 abort，而系统只统计「还在队列里的」，就会判成第三种（完全不认识），记个预防性 abort。可请求马上就要被派发了，这个预防性 abort 永远等不到它的请求 → **abort 丢失，请求照跑不误**。

`_CountingInbox` 就是来堵这道缝的。它维护两本账：

```
_request_counts   →   _claimed_counts   →   _pending
  （在队列里）        （出队但没派发）        （在跑）
```

- 进队列 → 第一本账 +1
- 出队列 → 从第一本**搬到**第二本
- 派发完成 → 第二本清掉（就是上面那个 `finally` 干的事）
- `is_reachable()` → **两本账查任一本**，所以悬空期间也算「够得着」

**关键规则就一条：每个请求 ID 出了队列，就必须走到 `release_claim`。**

## 三、问题是什么：既不是 bug，也不是 feature

这是我一开始最困惑的地方——issue 读了半天不知道它想干嘛。

后来明白了：**它是一条「记账规则」的登记单**，相当于把代码里本该写成注释的一条 TODO 提成了 issue。当前代码是**对的**，issue 是在给未来挖坑的人立警示牌。

那「未来的坑」是什么？清账这件事，**只靠调度循环那一个 `finally`**。要是有别的地方调了出队却不走那个 `finally`，ID 就永久卡在第二本账里，然后：

1. `is_reachable()` 永远返回真
2. 之后所有针对它的 abort 全落进 `_queued_aborts`
3. 而 `_queued_aborts` **没有上限**（不像预防性 abort 有 LRU 淘汰），且只有调度循环取到该请求时才清——请求早没了，**永远清不掉**

issue 点名的未来路径是 Python 3.13 新增的 `Queue.shutdown(immediate=True)`。而项目 [`pyproject.toml`](../../pyproject.toml) 钉了 `requires-python = ">=3.10,<3.13"`，**3.13 根本进不来，所以今天不是 bug**。

## 四、我怎么解决的

### 1. 先别信推断，去复现

issue 只是「推断」3.13 会出问题。我先翻了 CPython 3.13 的 `Queue.shutdown` 源码：

```python
if immediate:
    while self._qsize():
        self._get()          # ← 直接调 _get()，子类的副作用照样触发
```

确认它确实走子类钩子。然后用机器上现成的 `/usr/local/bin/python3.13`（3.13.5）真跑了一遍：

```
drained _claimed_counts: {'req-0': 1, 'req-1': 1, 'req-2': 1}   ← 泄漏
is_reachable('req-0')  : True                                    ← 队列空了却还是真
_queued_aborts         : {'req-0'}                               ← 墓碑永远清不掉
```

**issue 描述的每一步都对上了。** 这一步把理论 issue 变成了已证实的 issue——是这个 PR 里最有价值的部分。

### 2. 顺手发现了一个原 issue 没写的坑

最自然的修法是覆盖 `shutdown()`，先清账再调父类。但——

```
mutex 类型: <class '_thread.lock'>
是可重入锁吗: False
```

`Queue.shutdown` 全程持有 `self.mutex`，而 `release_claim` 也要拿同一把锁，且这把锁**不可重入**。所以在 `shutdown` 里调 `release_claim` **会直接死锁**。将来真要修的人，必须在「已经持锁」的状态下直接操作账本。

### 3. 交付什么

既然当前不是 bug，就**不改生产代码**：

- 在 `_CountingInbox` 的 docstring 里把规则写清楚，连带死锁那个坑
- 两条测试：
  - 正常派发后两本账都清空
  - 绕过派发的 drain 会把 ID 卡死（**用 `_get()` 直接驱动，不用 `shutdown()`**——后者要 3.13，在 CI 的 3.12 上会被 skip，等于死代码）

### 4. 验证绊线真的有效

这步很重要。一条「断言当前行为」的测试，如果将来别人修好了它还照样通过，那它就是废的。

我临时加了个会正确清账的 `shutdown` 覆盖，跑了一遍——**第二条测试确实失败了**。说明将来谁加 drain 路径，必须读到那段 docstring 才能让测试过。这才叫绊线。

### 5. 提交前的检查

- 3.12.11（CI 用的版本）和 3.13.5 上各跑一遍，都是 **15 passed**
- `pre-commit run --files` 两个文件全绿，无自动修改

## 五、过程中踩的坑

### 坑 1：和自己撞车了

我在跑命令的同时，自己在另一个终端切了分支、`git add` 了 `notes/`。结果：

- 以为在 `test/counting-inbox-claim-invariant` 分支上，实际 HEAD 已被切回 `docs/architecture-notes-zh`
- commit 落到了错误的分支，还把 `notes/` 一起卷了进去

靠 `git reflog` 才看清发生了什么，`git reset HEAD~1` 撤销，没有损失。

**教训：别在两个终端里同时操作同一个 working tree。** 后来改用 `git worktree` 把 PR 的活隔离出去，就再也不撞了：

```bash
git worktree add /tmp/omni-pr test/counting-inbox-claim-invariant
```

同样的事后来又发生了一次：写 PR 记录时切到了 `main`，`notes/` 整个消失（它在 docs 分支上）。

### 坑 2：`git add` 两个文件 ≠ commit 里只有两个文件

暂存区是全局的。别人（或另一个自己）往里加东西，`git commit` 照单全收。
**提交前一定 `git diff --cached --name-only` 看一眼，提交后 `git show --stat` 再复查一遍。**

### 坑 3：`.venv` 里没有 pytest

项目 `pyproject.toml` 把 pytest 列为核心依赖，但本地 `.venv` 装的时候没带全。
不想污染现有环境的话，在临时目录另建一个只装 pytest 的 venv，用 `PYTHONPATH` 指向仓库跑就行——
这个测试文件只依赖标准库和一个 dataclass，跑得起来。

### 坑 4：礼貌问题

写完才发现 issue 评论区里 **SuKi2cn 在 8/28 已经说要做这个了**（他在本仓库零 PR 记录，5 天没动静）。
我在 PR 和 issue 评论里都点了他的名，说明我是后开的、方案接近、他要做的话我关掉。
**下次先看评论区再动手。**

## 六、结果

### 第一轮评审（2026-09-02）

SuKi2cn 来评审了，指出一个**真实的盲区**，值得记下来。

他说：调度循环那个 `with self._lock` 块**有两个出口**——

```python
with self._lock:
    try:
        if self._consume_reachable_tombstone(request_id):
            continue          # ← 出口 1：这个请求被 abort 过，直接丢弃
        future = self._executor.submit(...)
        self._pending[request_id] = future   # ← 出口 2：正常派发
    finally:
        self.inbox.release_claim(request_id)   # ← 两个出口共用这一个 finally
```

而我的测试**只跑了出口 2**（4 个正常请求全部派发成功）。所以如果将来有人「简化」代码、
把 `release_claim` 从 `finally` 挪到 `submit` 后面，正常路径照样清账，
**我的测试依然全绿**，但走 `continue` 那条路的请求就永远清不掉账了。

他给的补丁只有两行：入队一个请求后立刻 abort 它，逼测试走一遍出口 1。

**我用变异测试验证了他的判断**，三次实验：

| 实验 | 代码 | 测试 | 结果 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | 注入回归 | 我原来的 | ✅ 通过 | **坏消息**：代码坏了却不报警，盲区确认存在 |
| 2 | 注入回归 | 加了他的建议 | ❌ 失败 | **好消息**：代码坏了就报警，盲区堵上了 |
| 3 | 干净 | 加了他的建议 | ✅ 3.12/3.13 各 15 passed | 正常代码下不误报 |

实验 2 的「失败」正是我要的证据——**一条测试的价值 = 代码坏时必须失败，代码好时必须通过**。

已按建议采纳（`804c5ca`），并在评审串里回复了这三组数据。

**这条评审教会我的**：我自己写「验证绊线有效」那一步时，只对第二条测试做了变异验证，
第一条测试想当然地以为够了。**变异测试要对每一条测试都做一遍**，不能挑着做。

### 当前状态（2026-09-02）

- PR 开着，2 个 commit，非 draft
- 第一轮评审已回应，等 SuKi2cn 确认
- 还在等 maintainer 打 `run-ci` 标签 —— GPU CI 跑在自建 runner 上，不打标签不触发；
  免费 runner 上的 Lint / Docs Check / Test Layout 会自动跑

查进度：

```bash
gh pr view 1889 --repo sgl-project/sglang-omni --comments
gh pr checks 1889 --repo sgl-project/sglang-omni
```

<!-- 后续进展往下追加 -->

## 七、我学到了什么

1. **「找一个没人做的任务」本身就是个技术活。** 在活跃项目里，`good first issue` 标签基本等于「已经有人在做」。可靠的筛法是查 timeline 交叉引用。

2. **实测 > 推断。** 这个 PR 里最有价值的不是那两条测试，是那份 3.13 上的复现输出和顺带发现的死锁陷阱。原 issue 只写了推断，我把它证实了——这是没人能抢走的贡献。

3. **一条断言当前行为的测试，必须验证它「会在被修好时失败」**，否则它只是穿着测试外衣的注释。
   而且这件事要**对每一条测试都做**——我只对其中一条做了变异验证，另一条的盲区就被评审逮到了。

5. **好的评审会给出可验证的假设。** SuKi2cn 没有只说「你少测了一条路」，
   而是直接指出「把 release 挪到 submit 之后，你的测试仍会通过」——
   这是一个能跑出来的断言，所以我能在十分钟内确认他是对的。以后我评审别人也该这么写。

4. **`git worktree` 是并行开发的正解**，不是什么高级技巧。同一个仓库要同时干两件事，就该开两个 worktree。
