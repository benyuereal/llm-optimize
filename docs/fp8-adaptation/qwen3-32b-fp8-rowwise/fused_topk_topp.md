# Fused top-k + top-p 采样 kernel

## 背景

decode 采样原用 triton `_topk_topp_kernel`（`vllm/v1/sample/ops/topk_topp_triton.py`），
是**多趟百分位/三分搜索**算法（zeroth pass 统计 → 全 vocab 收集 outlier → 三分搜索
pivot → top-p pivot 搜索 → softmax+gather）。在**尖峰分布**（真实 decode logits，
top-1 概率极高）下，`while found_pivot == 0` 循环迭代次数爆炸，退化到 **~10ms/step**
（profiler 实测 median 10.1ms，509/515 次落在 5–12ms）。

> 注意：micro-bench 用 smooth randn 分布只能复现 1.5ms，**复现不出 10ms**——
> 10ms 是分布相关的，真实 decode logits 是尖峰的。

## 语义（关键，必须和 triton 完全一致）

triton 的 top-p **不是**标准 full-vocab nucleus：

1. 取 top-k 候选；
2. 把这 top-k 个候选**归一化到和为 1**（不是全词表 softmax）；
3. 按概率降序累加，保留最短前缀使累计概率 ≥ p。

fused kernel 严格复刻该语义。

## 算法

`torch.topk(logits, POOL=1024)` 选候选（对分布不敏感，~1.7ms）+ 单 triton kernel
做 mask 填充 + top-k 归一化 nucleus（~0.1ms）。共 2 类 kernel launch，对比 triton 多趟。
POOL=1024 足够覆盖 top_k=50 的候选。

## 修改的文件

### 1. 新增 `vllm/v1/sample/ops/fused_topk_topp.py`

- `_fused_topk_topp_mask_kernel`：单 kernel，mask 填充 + top-k 归一化 nucleus。
- `apply_top_k_top_p_fused(logits, k, p, pool=1024)`：wrapper。
  - `V < pool` 时回退 `apply_top_k_top_p_fast`。
  - `p is None` 时 `TOPP_ENABLED=False`，纯 top-k。

### 2. 修改 `vllm/v1/sample/ops/topk_topp_sampler.py`

- import：
  ```python
  if HAS_TRITON:
      from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
      from vllm.v1.sample.ops.fused_topk_topp import apply_top_k_top_p_fused
  ```
- dispatch `apply_top_k_top_p`：
  ```python
  if p is None and k is None:
      return logits
  # Fused single-pass top-k + top-p (matches triton semantics).
  # ~4x faster than multi-pass triton on peaked decode logits.
  # Requires k (top-k); pure top-p falls back to triton.
  if HAS_TRITON and k is not None and logits.shape[0] >= 8:
      return apply_top_k_top_p_fused(logits, k, p)
  if HAS_TRITON and logits.shape[0] >= 8:
      return apply_top_k_top_p_triton(logits, k, p)
  return apply_top_k_top_p_pytorch(logits, k, p)
  ```

## 前置条件 / 安全性

- `sampler.py:91` 在采样前 `logits = logits.to(torch.float32)`，
  所以 fused kernel 的 `assert logits.dtype == torch.float32` 安全。
- 仅当 `k is not None`（有 top-k）且 `B >= 8` 时启用；
  纯 top-p 或极小 batch 回退原 triton / pytorch 路径，行为不变。
- server 默认 `generation_config="auto"`，从 `generation_config.json` 加载
  top_k=50 / top_p=0.7 / temperature=0.7，所以 `k` 非 None，fused 生效。

## 验证

- **正确性**：`/workspace/_bench/test_fused_vs_triton.py` → **45/45 pass**
  （B∈{1,8,64} × {smooth,peaked,very_peaked} × {(50,0.7),(50,0.9),(50,None),(1,0.7),(100,0.7)}），
  triton vs naive 参考 45/45 一致。
- **dispatch 路由 + 输出一致性**：`apply_top_k_top_p` vs `apply_top_k_top_p_triton`，
  B=64 peaked → 0/64 行差异。
- **profiler 确认**（server 实测，batch=64，512 decode 步）：
  trace 中 `_topk_topp_kernel` 消失，替换为 `_fused_topk_topp_mask_kernel`
  （15.89ms）+ `torch.topk` 的 mbtopk/radix kernel，采样合计 **0.955ms/step**
  （原 ~10ms，4.3x）。

## 性能

| 分布 | 原 triton | torch fast path | **fused** |
|---|---|---|---|
| very_peaked（真实 decode，B=256, V=151936, top_k=50） | 9.934 ms | 2.990 ms | **2.303 ms** |
| server 实测（batch=64） | ~10 ms | — | **0.955 ms** |

## 回退方法

把 `topk_topp_sampler.py` 的 dispatch 改回：
```python
if HAS_TRITON and logits.shape[0] >= 8:
    return apply_top_k_top_p_triton(logits, k, p)
return apply_top_k_top_p_pytorch(logits, k, p)
```
（删除 fused import 与 fused 分支即可；`fused_topk_topp.py` 可保留不影响。）
