# 稀疏 Attention 分组并集与掩码

这个目录提供 `build_union_sparse_metadata` 的 PyTorch 参考实现，以及中文命令行可视化和正确性测试。
它将相邻 query 选择的 KV 块合并为有序并集，同时用整数位掩码保留每个 query 对每个 KV token 的访问权限。

序列长度参数使用与 GBSA 相同的语义：`cuSqlenQ` 对应 `cu_seqlens_q`，`seqUsedKv` 对应 `seqused_kv`。

## 1. 快速运行

以下命令都从仓库根目录 `union_sparse_attn_mask` 执行。使用已安装 PyTorch 的 Python：

```bash
# 精简可视化：只显示第一组，验证全部输入
python visualize_union_sparse.py

# 全部正确性测试
python -m unittest discover -s . -v

# 运行实现文件内的调用示例
python union_sparse_torch.py
```

需要 Python 3.10 或更新版本，以及 PyTorch。依赖清单见 `requirements.txt`。

目录中的文件：

| 文件 | 用途 |
| --- | --- |
| `union_sparse_torch.py` | 算子接口、参数校验、并集与位掩码实现 |
| `visualize_union_sparse.py` | 生成可复现输入，展示输入输出，并独立验证 |
| `test_union_sparse_torch.py` | 随机对照、请求边界、容量和掩码测试 |
| `test_visualization.py` | 命令行展示、参数默认值及错误检出测试 |

## 2. 调用方式

下面的例子可在仓库根目录直接运行：

```python
import torch
from union_sparse_torch import build_union_sparse_metadata

# 1 个 KV head、3 个 query，每个 query 最多选 2 个 KV 块。
sparse_idx = torch.tensor([
    [[0, -1], [0, 1], [1, -1]]
], dtype=torch.int32)
sparse_count = torch.tensor([[1, 2, 1]], dtype=torch.int32)

result = build_union_sparse_metadata(
    sparse_idx,
    sparse_count,
    cuSqlenQ=torch.tensor([0, 3], dtype=torch.int64),
    seqUsedKv=torch.tensor([10], dtype=torch.int32),
    maxUnionSparseCount=513,
)

# blockX=1、blockY=8、unionSize=8、isCausal=True 均使用函数默认值。
print(result.unionSparseIdx.shape)     # torch.Size([1, 1, 513])
print(result.unionSparseCount)         # tensor([[2]], dtype=torch.int32)
print(result.unionSparseIdx[0, 0, :3])  # tensor([0, 1, -1], dtype=torch.int32)
print(result.exactMask[0, 0, :16])      # [3]*8 + [6, 4] + [0]*6
```

输入已处理因果约束时，显式传入 `isCausal=False`。即使关闭 causal，KV 有效长度边界仍会生效。

### 输入约定

设 `H` 为 KV head 数，`T` 为拼接后的 query 总数，`K` 为单个 query 的稀疏块容量，`B` 为请求数。

| 输入参数 | 类型／形状 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `sparseIdx` | 张量 `[H, T, K]` | 必传 | 每个 query 选择的请求内逻辑 KV 块编号 |
| `sparseCount` | 张量 `[H, T]` | 必传 | 每行实际有效的稀疏块数量 |
| `cuSqlenQ` | 张量 `[B+1]` | 必传 | 从 0 开始的 query 前缀和，相邻项之差为请求 Q 长度 |
| `seqUsedKv` | 张量 `[B]` | 必传 | 各请求实际有效的 KV token 长度，不是前缀和 |
| `blockX` | 整数 | `1` | Q 块粒度，目前仅支持 1 |
| `blockY` | 整数 | `8` | 每个 KV 块包含的 token 数，必须为正整数 |
| `unionSize` | 整数 | `8` | 每组最多包含的 query 数，支持 1～32 |
| `isCausal` | 布尔值 | `True` | 是否应用右对齐的因果约束 |
| `maxUnionSparseCount` | 整数或 `None` | `None` | 输出并集容量 U；省略时取 `unionSize * K` |

四个输入张量都接受 `int32` 或 `int64`。`sparseIdx` 和 `sparseCount` 必须位于同一设备。

- 每行只有前 `sparseCount[h,q]` 项有效，编号非负且升序排列，允许重复编号；后续项必须为 `-1`。
- KV 块编号不是 token 起始位置，也不是物理 cache 地址。块 `b` 对应的 token 位置为 `b * blockY + lane`。
- query 按请求紧凑拼接，请求之间没有额外 padding。不需要 `cuSqlenKv` 或 `seqUsedQ`。
- `cuSqlenQ` 必须从 0 开始，保持非递减，末项必须等于 `T`。
- `seqUsedKv` 必须有 `B` 项，且均为非负数。

例如 `cuSqlenQ=[0,11,16]`、`seqUsedKv=[23,13]` 表示两个请求：Q 长度分别为 11、5，KV 长度分别为 23、13。

示例、可视化和主随机测试显式设置 `maxUnionSparseCount=513`，不改变函数本身的默认容量规则。
容量必须容纳所有实际并集，容量不足会报错，不会截断结果。

### 输出

设 `G` 为全部请求的 query 分组总数，`U` 为并集容量。返回值是可按字段读取或按顺序解包的 `UnionSparseMetadata`：

| 输出 | 形状 | 有效值与填充 |
| --- | --- | --- |
| `unionSparseIdx` | `[H, G, U]` | 按升序保存逻辑 KV 块编号，尾部补 `-1` |
| `unionSparseCount` | `[H, G]` | 并集有效块数量，单位是块，不是 token |
| `exactMask` | `[H, G, U * blockY]` | 每个 token 对应一个 query 位掩码，填充为 `0` |

所有输出均为 `int32`，位于 `sparseIdx` 所在设备。容量为 513、`blockY=8` 时，`exactMask` 最后一维为 4104。

## 3. 算法核心流程

```text
读取请求边界和长度
        ↓
每个请求内按 unionSize 切分 query
        ↓
各 KV head 分别合并有效块编号，排序去重
        ↓
构造 query × 并集块的成员关系
        ↓
将每个块展开成 blockY 个 token，应用长度/因果约束
        ↓
按组内 query 位置压缩为 int32 位掩码
```

### 3.1 请求内分组

每个请求独立分组，尾组只包含剩余的有效 query，分组不会跨越请求边界：

```text
请求0：11个query → q0～q7、q8～q10，共2组
请求1： 5个query → q0～q4，共1组
全局分组总数 G = 3
```

数学上，`G = sum(ceil(Q长度[b] / unionSize))`。这里的分组是 query token 分组，不是 GQA head 分组。

### 3.2 排序去重，得到并集

先根据 `sparseCount` 排除 `-1` 填充，再对同组 query 的有效块编号执行 `torch.unique(..., sorted=True)`。

```text
q0选择 {0}
q1选择 {0,1}
q2选择 {1}
       ↓
并集 {0,1}
unionSparseIdx   = [0,1,-1,...]
unionSparseCount = 2
```

每个 KV head 独立执行。并集只记录块编号，不会因为某个块最终的 token 掩码全为 0 就删除它。

### 3.3 恢复每个 query 的选择关系

并集是整组共享的，但组内不同 query 选择的块可能不同。
因此需要将每个 query 的原始有效索引与并集比较，判断它选中了哪些并集块；不能仅从并集恢复这一关系。

设本组 query 数为 `R`、每行索引容量为 `K`、实际并集块数为 `C`，对每个 KV head：

```python
# row_indices、valid：[R, K]；blocks：[C]
matches = (row_indices[:, :, None] == blocks[None, None, :])  # [R, K, C]
membership = (matches & valid[:, :, None]).any(dim=1)        # [R, C]
```

`valid` 根据 `sparseCount` 排除填充；沿索引维取 `any`，表示只要一个有效索引匹配该并集块，就视为选中。
例如原始选择为 `{0}`、`{0,1}`、`{1}`，并集为 `[0,1]`，得到：

```text
        并集块0  并集块1
q0         1        0
q1         1        1
q2         0        1
```

随后用 `repeat_interleave(blockY)` 将成员关系展开到 token 粒度。

### 3.4 应用 KV 长度和 causal

所有 token 都必须满足 `kv_position < seqUsedKv[b]`。
若启用 `isCausal=True`，还需要满足右对齐条件：

```text
kv_position <= seqUsedKv[b] - Q长度[b] + 请求内q位置
```

上面调用示例中 `Q=3`、`KV=10`，q0、q1、q2 分别对应完整 KV 序列中的位置 7、8、9。
因此 q0 最多访问 k7，q1 最多访问 k8，q2 最多访问 k9。位置是逻辑序列位置，不是内存地址。

关闭 causal 时，只保留输入选择关系和 KV 有效长度约束。选中一个块意味着访问该块内的有效 token；接口没有额外的逐 token 输入掩码。

### 3.5 将 query 维压缩为整数位掩码

对于并集 token 列 `t`：

```text
exactMask[h,g,t] = sum(允许访问的query slot i所对应的 1 << i)
```

第 i 位表示组内第 i 个 query 能否访问该 token，不对应 head，也不对应 token lane。
调用示例中的结果为：

| KV token | 可访问的组内 query | 二进制（q2 q1 q0） | 整数值 |
| --- | --- | --- | --- |
| k0～k7 | q0、q1 | `011` | 3 |
| k8 | q1、q2 | `110` | 6 |
| k9 | q2 | `100` | 4 |
| k10～k15 | 无，超出KV长度 | `000` | 0 |

输出第 `rank * blockY + lane` 列对应块 `unionSparseIdx[h,g,rank]` 的第 `lane` 个 token。
读取某个 query 的 bit 可使用：

```python
allowed = ((int(mask_value) & 0xFFFFFFFF) >> query_slot) & 1
```

尾组缺少的 query 位保持 0。`unionSize=32` 时第 31 位使用 `int32` 符号位，整数可能为负数，解码时使用上面的无符号转换。

## 4. 命令行可视化

### 默认精简模式

```bash
python visualize_union_sparse.py
```

默认只显示第一个切片，内容包含实际参数、请求长度、输入索引及数量、输出并集、实际 0/1 矩阵、对应整数掩码和验证结果。
矩阵中每段字符串对应一个并集块，从左到右是块内 token；每一行对应一个 query。
正常的 `-1/0` 填充区域折叠显示数量，验证仍覆盖全部 513 个并集位置。

### 筛选与切换

```bash
# 请求0、head0的尾组；group是请求内分组编号
python visualize_union_sparse.py \
  --request 0 --head 0 --group 1

# 展示全部切片
python visualize_union_sparse.py --all

# 输入已处理因果约束时，关闭额外的causal
python visualize_union_sparse.py --no-causal

# 自定义请求长度
python visualize_union_sparse.py \
  --q-lengths 11,5 --kv-lengths 23,13 --heads 2 --seed 7
```

`--request`、`--head`、`--group` 可独立筛选，显式筛选时展示全部匹配切片。
未指定筛选和 `--all` 时，只显示第一个切片。`--all` 不能与筛选同时使用。

### 详细模式

```bash
# 完整输入输出、二进制解码及逐列比较
python visualize_union_sparse.py --detail

# 在详细模式中额外显示屏蔽原因
python visualize_union_sparse.py --show-reasons

# 保存完整报告，避免终端刷屏
python visualize_union_sparse.py \
  --detail --color never > /tmp/union_sparse_report.txt
```

详细模式保留全部填充值，容量 513 时输出量较大。每张解码表最多 8 列，较窄终端自动减少列数，可用 `--tokens-per-panel` 调整上限。
访问条件单独显示为“可见、未选、因果、越界、填充”，实际输出矩阵仍保持 0/1。

其他选项：`--max-union-sparse-count` 调整容量，`--block-y` 和 `--union-size` 调整块/分组大小，`--color auto|always|never` 控制颜色。
默认调用仅显式设置容量，四个算子属性使用函数默认值；修改对应选项时才覆盖默认属性。
脚本生成可复现示例输入；若要展示自己的稀疏模式，可以修改 `build_demo` 的输入生成部分。

## 5. 正确性验证

```bash
python -m unittest discover -s . -v
```

主随机测试包含 120 组输入，使用容量 513 和默认属性；覆盖多请求、多 head、空序列、尾组、重复索引及填充。
其他测试覆盖填满 513 个并集位置、容量不足、请求边界不匹配、右对齐 causal、KV 尾块和第 31 位符号位等情况。

可视化验证器只从输入出发，用 Python 集合构造期望并集，再逐 query、逐 token 计算期望整数，不使用实际输出构造期望值。
它检查并集及顺序、有效块数、每个 token 的掩码和填充。显示筛选不影响验证范围。

- 成功：打印 `验证结果：通过`，退出码为 0。
- 结果不一致：报告请求、head、分组、token 列、期望值与实际值，退出码为 1。
- 参数错误：命令行报告原因并以非零状态退出。

这是参考实现：包含 Python 循环、将长度读取到 CPU 的操作，以及 query/块广播比较。
适合检查算法和对照 kernel 结果，尚未针对性能或 CUDA Graph 优化；当前测试在 CPU 上验证。

## 6. 运行问题排查

若提示找不到 `torch`，确认当前 Python 已安装 PyTorch。
在仓库根目录可以检查：

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

若只出现 `Failed to initialize NumPy` 警告，但最后显示“验证结果：通过”，计算已经成功。
本目录的实现不依赖 NumPy；需要消除该警告时，可在同一环境中安装 NumPy。

其他失败请保留完整命令和 Traceback，区分依赖导入失败、参数错误和实际计算结果不一致。
