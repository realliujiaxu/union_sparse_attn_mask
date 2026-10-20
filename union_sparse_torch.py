"""PyTorch reference for the parameter contract in union_design.xlsx, Sheet2.

Sparse indices are request-local KV block IDs (not token offsets or physical
cache addresses). Query rows are packed across requests. Groups never cross
request boundaries. This reference uses Python loops and host length reads;
it is intended for correctness checks, not CUDA graph capture or benchmarking.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class UnionSparseMetadata(NamedTuple):
    unionSparseIdx: torch.Tensor
    unionSparseCount: torch.Tensor
    exactMask: torch.Tensor


def _length_vector(name: str, value: torch.Tensor) -> list[int]:
    if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional int32/int64 tensor")
    return value.detach().cpu().tolist()


def _offsets(name: str, value: torch.Tensor) -> list[int]:
    offsets = _length_vector(name, value)
    if len(offsets) < 2 or offsets[0] != 0:
        raise ValueError(f"{name} must contain B+1 entries, starting at 0")
    if any(end < start for start, end in zip(offsets, offsets[1:])):
        raise ValueError(f"{name} must be nondecreasing")
    return offsets


def build_union_sparse_metadata(
    sparseIdx: torch.Tensor,
    sparseCount: torch.Tensor,
    cuSqlenQ: torch.Tensor,
    seqUsedKv: torch.Tensor,
    *,
    blockX: int = 1,
    blockY: int = 8,
    unionSize: int = 8,
    isCausal: bool = True,
    maxUnionSparseCount: int | None = None,
) -> UnionSparseMetadata:
    """Build sorted per-head unions and a token-wise query membership bitmask.

    Args:
        sparseIdx: Int32/int64 [kvN, totalqblocknum, maxKvBlockNum]. Each
            valid prefix is sorted, contains nonnegative request-local KV
            block IDs, and is followed by -1 padding. Duplicate IDs are allowed.
        sparseCount: Int32/int64 [kvN, totalqblocknum], valid prefix lengths.
        cuSqlenQ: Packed Q offsets [B+1], starting at 0, matching
            GBSA cu_seqlens_q. Adjacent differences are the actual Q lengths:
            there must be no padding between requests, and the final
            offset must equal totalqblocknum.
        seqUsedKv: Actual KV token lengths [B], matching GBSA seqused_kv.
            These lengths determine KV tail validity and causal alignment;
            no cumulative KV offsets or physical cache addresses are needed.
        blockX: Q block size. Only the workbook's value 1 is supported.
        blockY: KV tokens per sparse block, default 8.
        unionSize: Q rows per union, default 8, at most 32 for an int32 mask.
        isCausal: Defaults to True, using bottom-right alignment: key_position
            <= kv_length - q_length + query_position. Set False when input
            selections already handle causality.
        maxUnionSparseCount: Output capacity U. Defaults to the lossless upper
            bound unionSize * maxKvBlockNum. An explicit smaller capacity is
            allowed only if every actual union fits; otherwise raise.

    Returns:
        unionSparseIdx: Int32 [kvN, G, U], sorted block IDs, padded with -1.
        unionSparseCount: Int32 [kvN, G], number of valid union BLOCKS.
        exactMask: Int32 [kvN, G, U * blockY]. Bit i is 1 when query slot i
            selected that block AND may attend to that particular KV token.
            Zero means no query can attend. Unused query slots, union padding,
            and tokens beyond seqUsedKv have zero bits. Slot 31 uses the sign
            bit of int32. The query-bitset encoding follows the CuTe code and
            the clarification accompanying the workbook.

        G = sum(ceil(active_q_length[b] / unionSize)). Groups are concatenated
        in request order, independently for each KV head. A selected block
        remains in the union even if all its tokens are masked by causality.
    """
    if blockX != 1:
        raise ValueError("only blockX=1 is supported")
    if not isinstance(blockY, int) or blockY <= 0:
        raise ValueError("blockY must be a positive integer")
    if not isinstance(unionSize, int) or not 1 <= unionSize <= 32:
        raise ValueError("unionSize must be an integer in [1, 32]")
    if sparseIdx.ndim != 3 or sparseIdx.dtype not in (torch.int32, torch.int64):
        raise ValueError("sparseIdx must be int32/int64 [kvN, totalQ, K]")
    heads, total_q, topk = sparseIdx.shape
    if (
        sparseCount.shape != (heads, total_q)
        or sparseCount.dtype not in (torch.int32, torch.int64)
        or sparseCount.device != sparseIdx.device
    ):
        raise ValueError("sparseCount must be int32/int64 [kvN, totalQ] on the input device")
    if bool(((sparseCount < 0) | (sparseCount > topk)).any()):
        raise ValueError("sparseCount must be in [0, maxKvBlockNum]")

    q_offsets = _offsets("cuSqlenQ", cuSqlenQ)
    if q_offsets[-1] != total_q:
        raise ValueError("cuSqlenQ must cover exactly the packed sparseIdx query axis")
    batch = len(q_offsets) - 1
    used_q = [end - start for start, end in zip(q_offsets, q_offsets[1:])]
    used_kv = _length_vector("seqUsedKv", seqUsedKv)
    if len(used_kv) != batch or any(length < 0 for length in used_kv):
        raise ValueError("seqUsedKv must have B nonnegative lengths")

    capacity = unionSize * topk if maxUnionSparseCount is None else maxUnionSparseCount
    if not isinstance(capacity, int) or capacity < 0:
        raise ValueError("maxUnionSparseCount must be a nonnegative integer")
    groups = sum((length + unionSize - 1) // unionSize for length in used_q)
    device = sparseIdx.device
    union_idx = torch.full((heads, groups, capacity), -1, dtype=torch.int32, device=device)
    union_count = torch.zeros((heads, groups), dtype=torch.int32, device=device)
    exact_mask = torch.zeros(
        (heads, groups, capacity * blockY), dtype=torch.int32, device=device
    )
    input_slots = torch.arange(topk, device=device)
    token_lanes = torch.arange(blockY, device=device)
    group = 0
    for request, q_length in enumerate(used_q):
        kv_length = used_kv[request]
        for local_start in range(0, q_length, unionSize):
            rows = min(unionSize, q_length - local_start)
            start = q_offsets[request] + local_start
            selected = sparseIdx[:, start : start + rows, :].to(torch.int64)
            counts = sparseCount[:, start : start + rows]
            valid = input_slots[None, None, :] < counts[:, :, None]
            if bool(((selected < 0) & valid).any()):
                raise ValueError("valid sparseIdx entries must be nonnegative")
            if bool(((selected > torch.iinfo(torch.int32).max) & valid).any()):
                raise ValueError("sparse block IDs must fit in int32")
            if bool(((selected != -1) & ~valid).any()):
                raise ValueError("sparseIdx padding must be -1")
            if topk > 1 and bool(
                ((selected[:, :, 1:] < selected[:, :, :-1]) & valid[:, :, 1:]).any()
            ):
                raise ValueError("valid sparseIdx prefixes must be sorted")
            q_position = local_start + torch.arange(rows, device=device)
            bit_weights = torch.ones(rows, dtype=torch.int64, device=device) << torch.arange(
                rows, device=device
            )
            for head in range(heads):
                blocks = torch.unique(selected[head][valid[head]], sorted=True)
                count = blocks.numel()
                if count > capacity:
                    raise ValueError(
                        f"maxUnionSparseCount={capacity} is smaller than actual union "
                        f"{count} at KV head {head}, group {group}"
                    )
                union_idx[head, group, :count] = blocks.to(torch.int32)
                union_count[head, group] = count
                membership = (
                    (selected[head, :, :, None] == blocks[None, None, :])
                    & valid[head, :, :, None]
                ).any(dim=1)
                allowed = membership.repeat_interleave(blockY, dim=1)
                kv_positions = (blocks[:, None] * blockY + token_lanes).reshape(-1)
                allowed &= kv_positions[None, :] < kv_length
                if isCausal:
                    q_absolute = kv_length - q_length + q_position
                    allowed &= kv_positions[None, :] <= q_absolute[:, None]
                packed = (allowed.to(torch.int64) * bit_weights[:, None]).sum(dim=0)
                exact_mask[head, group, : count * blockY] = packed.to(torch.int32)
            group += 1
    return UnionSparseMetadata(union_idx, union_count, exact_mask)


if __name__ == "__main__":
    indices = torch.tensor([[[0, -1], [0, 1], [1, -1]]], dtype=torch.int32)
    counts = torch.tensor([[1, 2, 1]], dtype=torch.int32)
    result = build_union_sparse_metadata(
        indices,
        counts,
        cuSqlenQ=torch.tensor([0, 3]),
        seqUsedKv=torch.tensor([10]),
        maxUnionSparseCount=513,
    )
    for name, tensor in zip(result._fields, result):
        print(f"{name}: shape={tuple(tensor.shape)}\n{tensor}")
