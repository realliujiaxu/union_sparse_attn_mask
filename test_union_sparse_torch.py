import random
import unittest

import torch

if __package__:
    from .union_sparse_torch import build_union_sparse_metadata
else:
    from union_sparse_torch import build_union_sparse_metadata


def check(seed):
    rng = random.Random(seed)
    heads = rng.randrange(1, 4)
    batch = rng.randrange(1, 5)
    size = 8
    width = 8
    capacity = 513
    topk = rng.randrange(0, 6)
    q_used = [rng.randrange(0, 18) for _ in range(batch)]
    kv_used = [rng.randrange(0, 25) for _ in range(batch)]
    causal = True
    total = sum(q_used)
    data = torch.full((heads, total, topk), -1, dtype=torch.int32)
    counts = torch.zeros((heads, total), dtype=torch.int32)
    raw = [[[] for _ in range(total)] for _ in range(heads)]
    for h in range(heads):
        for row in range(total):
            values = sorted(rng.randrange(0, 8) for _ in range(rng.randrange(topk + 1)))
            raw[h][row] = values
            counts[h, row] = len(values)
            data[h, row, :len(values)] = torch.tensor(values, dtype=torch.int32)
    q_offsets = [0]
    for q in q_used:
        q_offsets.append(q_offsets[-1] + q)
    result = build_union_sparse_metadata(
        data, counts, torch.tensor(q_offsets), torch.tensor(kv_used),
        maxUnionSparseCount=capacity,
    )
    groups = sum((q + size - 1) // size for q in q_used)
    assert result.unionSparseIdx.shape == (heads, groups, capacity)
    assert result.exactMask.shape == (heads, groups, capacity * width)
    group = 0
    offset = 0
    for request, length in enumerate(q_used):
        for begin in range(0, length, size):
            rows = min(size, length - begin)
            for h in range(heads):
                selected = raw[h][offset + begin:offset + begin + rows]
                union = sorted(set(v for row in selected for v in row))
                assert result.unionSparseCount[h, group].item() == len(union)
                assert result.unionSparseIdx[h, group, :len(union)].tolist() == union
                assert (result.unionSparseIdx[h, group, len(union):] == -1).all()
                expected = []
                for block in union:
                    for lane in range(width):
                        token = block * width + lane
                        mask = 0
                        for slot, blocks in enumerate(selected):
                            if (block in blocks and token < kv_used[request]
                                and (not causal or token <= kv_used[request] - length + begin + slot)):
                                mask |= 1 << slot
                        if mask >= 2**31:
                            mask -= 2**32
                        expected.append(mask)
                assert result.exactMask[h, group, :len(expected)].tolist() == expected
                assert (result.exactMask[h, group, len(expected):] == 0).all()
            group += 1
        offset += q_used[request]


class UnionSparseTests(unittest.TestCase):
    def setUp(self):
        self.idx = torch.zeros((1, 32, 1), dtype=torch.int32)
        self.cnt = torch.ones((1, 32), dtype=torch.int32)

    def test_randomized_against_token_oracle(self):
        for seed in range(120):
            with self.subTest(seed=seed):
                check(seed)

    def test_capacity_513_fully_used_with_default_attributes(self):
        indices = torch.full((1, 8, 65), -1, dtype=torch.int32)
        counts = torch.tensor([[65] * 7 + [58]], dtype=torch.int32)
        for row, count in enumerate(counts[0].tolist()):
            indices[0, row, :count] = torch.arange(row * 65, row * 65 + count)
        result = build_union_sparse_metadata(
            indices, counts, torch.tensor([0, 8]), torch.tensor([513 * 8]),
            maxUnionSparseCount=513,
        )
        self.assertEqual(result.unionSparseIdx.shape, (1, 1, 513))
        self.assertEqual(result.exactMask.shape, (1, 1, 4104))
        self.assertEqual(result.unionSparseCount.item(), 513)
        self.assertEqual(result.unionSparseIdx[0, 0].tolist(), list(range(513)))
        self.assertEqual(result.exactMask[0, 0, -8:].tolist(), [128] * 8)

    def test_query_slot_31_survives_signed_int32(self):
        result = build_union_sparse_metadata(
            self.idx, self.cnt, cuSqlenQ=torch.tensor([0, 32]), seqUsedKv=torch.tensor([8]),
            unionSize=32, isCausal=False,
        )
        self.assertEqual(result.exactMask[0, 0, :8].tolist(), [-1] * 8)

    def test_empty_kv_masks_every_token(self):
        result = build_union_sparse_metadata(
            self.idx, self.cnt, cuSqlenQ=torch.tensor([0, 32]), seqUsedKv=torch.tensor([0]),
        )
        self.assertFalse(result.exactMask.any())
        self.assertEqual(result.unionSparseCount.tolist(), [[1, 1, 1, 1]])

    def test_cumulative_q_and_used_kv_describe_packed_requests(self):
        result = build_union_sparse_metadata(
            self.idx[:, :3], self.cnt[:, :3], cuSqlenQ=torch.tensor([0, 1, 3]),
            seqUsedKv=torch.tensor([8, 10]), maxUnionSparseCount=1,
        )
        self.assertEqual(result.unionSparseCount.tolist(), [[1, 1]])

    def test_undersized_capacity_raises_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "actual union"):
            build_union_sparse_metadata(
                self.idx, self.cnt, cuSqlenQ=torch.tensor([0, 32]), seqUsedKv=torch.tensor([8]),
                maxUnionSparseCount=0,
            )

    def test_kv_lengths_must_match_request_count(self):
        with self.assertRaisesRegex(ValueError, "B nonnegative lengths"):
            build_union_sparse_metadata(
                self.idx, self.cnt, torch.tensor([0, 16, 32]), torch.tensor([40]),
            )

    def test_q_offsets_must_cover_packed_query_axis(self):
        with self.assertRaisesRegex(ValueError, "packed sparseIdx query axis"):
            build_union_sparse_metadata(
                self.idx, self.cnt, torch.tensor([0, 31]), torch.tensor([40]),
            )

    def test_q_offsets_must_start_at_zero(self):
        with self.assertRaisesRegex(ValueError, "starting at 0"):
            build_union_sparse_metadata(
                self.idx, self.cnt, torch.tensor([1, 33]), torch.tensor([40]),
            )

    def test_known_trapezoid_and_partial_kv_block(self):
        result = build_union_sparse_metadata(
            torch.tensor([[[0, -1], [0, 1], [1, -1]]], dtype=torch.int32),
            torch.tensor([[1, 2, 1]], dtype=torch.int32),
            cuSqlenQ=torch.tensor([0, 3]), seqUsedKv=torch.tensor([10]),
            maxUnionSparseCount=2, isCausal=True,
        )
        self.assertEqual(result.unionSparseIdx.tolist(), [[[0, 1]]])
        self.assertEqual(result.exactMask.tolist(), [[[3] * 8 + [6, 4] + [0] * 6]])

    def test_default_applies_causal(self):
        result = build_union_sparse_metadata(
            torch.tensor([[[1], [1], [1]]], dtype=torch.int32),
            torch.tensor([[1, 1, 1]], dtype=torch.int32),
            cuSqlenQ=torch.tensor([0, 3]), seqUsedKv=torch.tensor([10]),
        )
        self.assertEqual(result.exactMask[0, 0, :8].tolist(), [6, 4, 0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
