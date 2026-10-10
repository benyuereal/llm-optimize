# HCU fp8 channelwise GEMM:从 ~200 到 432 TFLOPS 的优化复盘

> 目标:让 HCU(gfx938)上原生 tilelang 编译的 fp8 channelwise GEMM,尽量逼近
> lightop / hipblasLt w8a8 的 ~560 TFLOPS 峰值。
>
> 我们分三步走,把 8192³ 的算力从 **~200 TFLOPS** 提到主干水平的 **~300 TFLOPS**,
> 再到当前 **432 TFLOPS**(占参考峰值 77%),最终结果与精确 fp32 参考**逐元素相等**。
>
> 环境:Hygon DCU **BW1101**,gfx938,64 CU,warp size 64。所有运行用 `HIP_VISIBLE_DEVICES=2`。
> 脚本在本目录(三个版本 + 诊断都留了,方便回溯每一步改了什么)。

| 阶段 | 路径 | TFLOPS @ 8192³ | 脚本 |
|---|---|---|---|
| ① 起步 | 手写 async copy channelwise | ~200 | `async_copy_gemm_channelwise.py` |
| ② 主干水平 | 原生 `T.gemm` fp8 + cp.async(scalar LDS) | ~300 | `native_fp8_channelwise.py` |
| ③ 当前 | 原生 `T.gemm` fp8 + **MLS 矩阵 LDS 读** | **432** | `repro_fp8_432.py` |

一键复现当前最优:

```bash
cd /workspace/llm-optimize
HIP_VISIBLE_DEVICES=2 python fp8_channelwise_gemm/repro_fp8_432.py
# 期望:max_rel=0.00e+00, 走了 ds_read_format_tensor_a/b, TFLOPS ~= 430
```

---

## 0. 参考基线是怎么来的

- 参考峰值:lightop / hipblasLt w8a8 在 8192³ ≈ **560 TFLOPS**。
- 单看 MMAC 指令 `v_mmac_f32_16x16x32_fp8_fp8` 的理论上限 ≈ **529 TFLOPS**。
- 我们把 hipblaslt/Tensile 的 `.co` 库用 `/opt/dtk/aillvm/bin/llvm-objdump` 反汇编,
  逐条和我们编译出的内核对比,得到下面这条贯穿三步的核心结论。

## 1. 关键定位:差距不在 MMA,也不在 global 拷贝

对比我方内核与 lightop/hipblaslt 反汇编后确认:**两者用的是同一条 MMAC 指令、
同一种 16 字节异步 global→shared 拷贝**(`cp_async_gs<16>` / `buffer_load_dwordx4`)。

真正的差异只在 **shared→register(fragment)这一读的形态**:

| shared 读方式 | 指令 | 对应阶段 |
|---|---|---|
| scalar 散读 | 逐元素 `ds_read_b8` | ① / ② |
| **矩阵 LDS 读** | `ds_read_format_tensor_a/b` → HCU `ds_read_m32x32_b8` | ③ |

结论:global 搬运和 MMA 都不是瓶颈,**LDS 读是不是"矩阵形态"并和 MMA 重叠**
才是决定性能的那一环。lightop/hipblaslt 用的正是硬件矩阵 LDS 读。
三步优化就是围绕这一点。验证工具:`fp8_asm_dump.py`
(导出内核源码,grep `ds_read_m32x32_b8` / `v_mmac` / `cp_async`)。

---

## 2. ① 起步:手写 async copy,~200 TFLOPS

`async_copy_gemm_channelwise.py`。手写四段异步 pipeline,
把 global→shared 做成了 16 字节异步,但 **fragment 读仍是 scalar 散读**,
且流水线重叠没调好。8192³ 大约 **~200 TFLOPS**。

这一步的价值:搭起 fp8 channelwise 的正确骨架——
- channelwise epilogue:`C_l[i,j] *= scale_a[i] * scale_b[j]`(scale 在 fp32 累加器上);
- fp8 e4m3 的 shared/fragment 分配;
- 手写 MMAC 循环。

性能离峰值远,但骨架是对的。

## 3. ② 主干水平:原生 `T.gemm` fp8,~300 TFLOPS

对照版脚本:`native_fp8_channelwise.py`。
关键改动:**把手写 MMAC 循环换成 tilelang 的 `T.gemm` fp8 原语**,配 cp.async 生产者
+ channelwise epilogue,线程拉到 1024。这就是合并 main 后的水平:

```python
for k in T.Pipelined(K // block_K, num_stages=2):
    T.copy(A[...], A_shared)          # cp.async 生产者
    T.copy(B[...], B_shared)
    T.gemm(A_shared, B_shared, C_local, False, True, k_pack=1)   # fp8 MMAC,scalar LDS
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] *= scale_a[...] * scale_b[...]
T.copy(C_local, C[...])
```

- 最优:256×256×32, ns=2, 1024 线程 → **~314 TFLOPS**。
- 相比 ① 的收益:lowering 生成更规整的 MMAC 循环 + 更充分的异步分段。
- **但 fragment 读仍是 scalar LDS**,天花板被压在 300 出头。

### ② 的天花板:scalar 路径加线程只能到 ~330
继续加线程 256→512→1024 能爬到 ~332,但 **1536/2048 会寄存器溢出**——scalar 路径
的硬上限,跨不过去。结论:光靠 occupancy 不够,必须换 LDS 读形态 → ③。

## 4. ③ 当前:MLS 矩阵 LDS 读,432 TFLOPS

`fp8_mls_channelwise.py` / `repro_fp8_432.py`。
把 LDS 读从 scalar 切到**硬件矩阵形态**:让 A 和 B **都**走 MLS 生产者
(`prefer_instruction="matrix_load"`),`T.gemm` 就走 `_gemm_mls_mls` lowering,
生成 `ds_read_format_tensor_a/b` + MMAC:

```python
for k in T.Pipelined(K // block_K, num_stages=1):
    T.copy(A[...], A_shared, prefer_instruction="matrix_load")   # A 走矩阵 LDS
    T.copy(B[...], B_shared, prefer_instruction="matrix_load")   # B 走矩阵 LDS
    T.gemm(A_shared, B_shared, C_local, False, True, k_pack=1)   # -> ds_read_format + v_mmac
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] *= scale_a[...] * scale_b[...]
T.copy(C_local, C[...])
```

> 这条指令链正是 hipblaslt fp8 内核用的读法,是 300→400+ 的关键,现在打通了。

### 最优配置与 sweep(8192³, block_K=64)

| 配置 (bm×bn×64, ns, threads) | TFLOPS |
|---|---|
| **256×256, ns=1, 512** | **432.6** ✅ 最优 |
| 256×128, ns=1, 256 | 351 |
| 256×256, ns=2, 512 | 340 |
| ns=0 各配置 | ≈ 245 |

两个决定性因素:**矩阵 LDS 读** + **单段异步重叠(ns=1)**。
sweep 脚本:`fp8_mls_channelwise_bench.py`。

### ③ 踩过的两个约束

1. **`block_K` 必须是 64 的倍数。**
   fp8 MLS 在 `trans_B=True` 时,B 侧用 trans b8 MLS atom(见 tilelang 仓库
   `src/hcu/op/mls.cc` 的 `kMlsTileConfigsB8Trans`,tile_k=64/128),要求
   `block_K % 64 == 0`。`block_K=32` 会报 "No valid MLS tile config"。参考内核正是 64。

2. **`num_stages==0` 时手动补异步等待。**
   MLS 的 global→shared 拷贝是异步的;ns=0 时 pipeline pass 不会自动发 wait,
   必须显式 `T.ptx_wait_group(0); T.sync_warp()`,否则读到未落地的数据。ns=1 时自动。

---

## 5. 走过的弯路(负结果,别重复)

- **循环内 full-K 寄存器重叠(double-buffer fragment 读)**:gfx938 fp8 上
  **零收益**(225.7 vs 225.8)。瓶颈是每线程 256 宽 fp8 fragment 的寄存器压力,
  不是 load-vs-MMAC 的重叠,改动已回退。
- **scalar 路径无脑加线程数**:上限 ~330(寄存器溢出),跨不过去,必须走 ③ 的矩阵 LDS 读。

## 6. 复现注意事项

- 所有 GPU 运行统一 `HIP_VISIBLE_DEVICES=2`。
- 三个版本的脚本**全部在 DSL 层,无 C++ 改动**,`tilelang` 装好即可直接跑
  (本机是 dev build,`import tilelang` 全局可用)。
- **缓存陷阱**:若改过 tilelang 的 C++,`~/.tilelang/cache/` 的 key 只含 git hash、
  不含未提交的 C++ 改动,须 `rm -rf ~/.tilelang/cache/*` 再跑。纯 Python 改动同理,
  清缓存更保险。
- 正确性:③ 的结果与精确 fp32 参考逐元素相等(max_rel=0.00e+00);
  注意别用 bf16 做参考(bf16 自身再舍入一次,会看到 ~0.004 的假误差)。

## 7. 交付物索引(本目录)

| 文件 | 作用 |
|---|---|
| `async_copy_gemm_channelwise.py` | ① 起步版(~200),手写 async + channelwise |
| `native_fp8_channelwise.py` | ② 主干版(~300),原生 `T.gemm` + scalar LDS |
| `fp8_mls_channelwise.py` | ③ 当前算子(默认 432 配置),供 import |
| `repro_fp8_432.py` | ③ 单文件自包含:编译 + 正确性校验 + 打印 TFLOPS(验证入口) |
| `fp8_mls_channelwise_bench.py` | ③ 的 (bm,bn)×ns×threads sweep |
| `fp8_asm_dump.py` | 导出内核源码,grep 关键指令,用于定位/验证 |
