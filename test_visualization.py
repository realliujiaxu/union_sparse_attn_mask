"""Exercise terminal output and prove the independent checker detects faults."""

import contextlib
import copy
import io
import unittest
from unittest.mock import patch

if __package__:
    from . import visualize_union_sparse as visual
else:
    import visualize_union_sparse as visual


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.data = visual.build_demo([11, 0, 5], [23, 0, 13])

    def test_partial_groups_empty_request_and_padding_pass_oracle(self):
        self.assertEqual([g["request"] for g in self.data["groups"]], [0, 0, 2])
        self.assertEqual([g["rows"] for g in self.data["groups"]], [8, 3, 5])
        self.assertEqual(visual.verify_demo(self.data), [])

    def test_demo_capacity_513_uses_operator_default_attributes(self):
        operator = visual.build_union_sparse_metadata
        with patch.object(visual, "build_union_sparse_metadata", autospec=True, side_effect=operator) as call:
            data = visual.build_demo([3], [10], heads=1)
        self.assertEqual(data["capacity"], 513)
        self.assertEqual(len(data["union"][0][0]), 513)
        self.assertEqual(len(data["mask"][0][0]), 4104)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.kwargs["maxUnionSparseCount"], 513)
        for name in ("blockX", "blockY", "unionSize", "isCausal"):
            self.assertNotIn(name, call.call_args.kwargs)

    def test_corrupt_mask_reports_token_and_marks_matrix(self):
        data = copy.deepcopy(self.data)
        data["mask"][0][0][0] ^= 1
        errors = visual.verify_demo(data)
        self.assertTrue(any("列=0，KV位置=" in e for e in errors))
        text = visual.render_group(data, 0, 0, tokens_per_panel=128)
        self.assertIn("不一致", text)
        self.assertIn("对照期望值", text)

    def test_corrupt_union_count_and_padding_are_detected(self):
        for field in ("union", "counts", "mask"):
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                if field == "counts":
                    data[field][0][0] += 1
                else:
                    data[field][0][0][-1] += 1
                self.assertTrue(visual.verify_demo(data))

    def test_cli_outputs_text_and_success_status(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = visual.main(["--q-lengths", "3", "--kv-lengths", "10", "--color", "never", "--detail", "--max-union-sparse-count", "3"])
        self.assertEqual(status, 0)
        text = output.getvalue()
        for label in ("cuSqlenQ", "seqUsedKv", "sparseIdx原始值", "sparseCount",
                      "输入选择矩阵", "unionSparseIdx[", "unionSparseCount[",
                      "exactMask[", "整数掩码", "二进制掩码", "KV位置", "本组验证"):
            self.assertIn(label, text)
        self.assertIn("验证结果：通过", text)
        self.assertNotIn("\033[", text)

    def test_report_is_compact_by_default_and_can_expand_or_filter(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(visual.main([]), 0)
        text = output.getvalue()
        self.assertIn("显示 1 个头/分组切片", text)
        self.assertIn("sparseIdx", text)
        self.assertIn("sparseCount", text)
        self.assertIn("unionSparseIdx", text)
        self.assertIn("unionSparseCount", text)
        self.assertIn("exactMask整数值", text)
        self.assertNotIn("二进制掩码", text)
        self.assertIn("maxUnionSparseCount=513", text)
        self.assertIn("折叠 510 个填充块", text)
        self.assertLess(len(text.splitlines()), 45)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(visual.main(["--all"]), 0)
        self.assertIn("显示 6 个头/分组切片", output.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(visual.main(["--request", "0", "--head", "1", "--group", "1"]), 0)
        self.assertIn("显示 1 个头/分组切片", output.getvalue())
        self.assertIn("请求0 / KV头1 / 请求内分组1", output.getvalue())

    def test_causal_switch_matches_displayed_parameters_and_oracle(self):
        causal = visual.build_demo([3], [10], heads=1, causal=True)
        noncausal = visual.build_demo([3], [10], heads=1, causal=False)
        self.assertNotEqual(causal["mask"], noncausal["mask"])
        for data in (causal, noncausal):
            self.assertEqual(visual.verify_demo(data), [])
            self.assertIn(f"isCausal={data['isCausal']}", visual.render_overview(data))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(visual.main(["--no-causal", "--show-reasons", "--max-union-sparse-count", "4"]), 0)
        self.assertIn("isCausal=False", output.getvalue())
        self.assertIn("访问条件（独立计算）", output.getvalue())

    def test_decoding_covers_every_column_without_truncation(self):
        data = visual.build_demo([3], [10], heads=1, max_union_sparse_count=3)
        text = visual.render_group(data, 0, 0, tokens_per_panel=1)
        self.assertEqual(text.count("逐列比较"), len(data["mask"][0][0]))
        self.assertIn("slot 7 … slot 0", text)
        # High unused query slots in a tail group must still appear as zero bits.
        for packed in data["mask"][0][0]:
            self.assertIn(format(packed & 0xFFFFFFFF, "08b"), text)

    def test_cli_failure_exits_nonzero(self):
        self.data["mask"][0][0][0] ^= 1
        with patch.object(visual, "build_demo", return_value=self.data):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(visual.main([]), 1)
        self.assertIn("验证结果：失败", output.getvalue())


if __name__ == "__main__":
    unittest.main()
