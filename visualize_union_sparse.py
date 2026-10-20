"""中文终端报告：输入张量 → 输入选择矩阵 → 输出张量 → 位掩码解码 → 验证。"""

from __future__ import annotations

import argparse
import inspect
import random
import shutil
import sys
import unicodedata

import torch

if __package__:
    from .union_sparse_torch import build_union_sparse_metadata
else:
    from union_sparse_torch import build_union_sparse_metadata


def build_demo(q_lengths, kv_lengths, heads=2, block_y=8, union_size=8, seed=7, causal=True,
               max_union_sparse_count=513):
    """构建可复现输入，保存实际传给实现的参数、张量及实际输出。"""
    if not q_lengths or len(q_lengths) != len(kv_lengths):
        raise ValueError("Q 和 KV 长度必须描述同一批非空请求")
    if any(q < 0 or kv < 0 for q, kv in zip(q_lengths, kv_lengths)):
        raise ValueError("序列长度不能为负数")
    if heads < 1 or block_y < 1 or not 1 <= union_size <= 32:
        raise ValueError("KV 头数、块大小必须为正数，分组大小必须在 1～32 之间")
    if not any(q_lengths):
        raise ValueError("至少一个请求必须包含 query")
    rng = random.Random(seed)
    topk = 3
    indices = torch.full((heads, sum(q_lengths), topk), -1, dtype=torch.int32)
    counts = torch.zeros((heads, sum(q_lengths)), dtype=torch.int32)
    offset = 0
    for q, kv in zip(q_lengths, kv_lengths):
        blocks = (kv + block_y - 1) // block_y
        for head in range(heads):
            for row in range(q):
                selected = sorted(rng.sample(range(blocks), min(blocks, rng.randint(1, topk))))
                indices[head, offset + row, :len(selected)] = torch.tensor(selected, dtype=torch.int32)
                counts[head, offset + row] = len(selected)
        offset += q
    cu_q = torch.tensor([0] + q_lengths, dtype=torch.int64).cumsum(0)
    used_kv = torch.tensor(kv_lengths, dtype=torch.int32)
    kwargs = dict(cuSqlenQ=cu_q, seqUsedKv=used_kv)
    # Default demo calls rely on the operator's defaults, rather than override them.
    defaults = inspect.signature(build_union_sparse_metadata).parameters
    for name, value in (("blockY", block_y), ("unionSize", union_size), ("isCausal", causal)):
        if value != defaults[name].default:
            kwargs[name] = value
    capacity = max_union_sparse_count
    result = build_union_sparse_metadata(indices, counts, **kwargs, maxUnionSparseCount=capacity)
    groups = []
    offset = 0
    for request, (q, kv) in enumerate(zip(q_lengths, kv_lengths)):
        for local_group, begin in enumerate(range(0, q, union_size)):
            rows = min(union_size, q - begin)
            groups.append(dict(
                request=request, localGroup=local_group, begin=begin, rows=rows,
                inputStart=offset + begin, qLength=q, kvLength=kv,
                input=indices[:, offset + begin:offset + begin + rows].tolist(),
                counts=counts[:, offset + begin:offset + begin + rows].tolist(),
            ))
        offset += q
    return dict(
        heads=heads, blockX=1, blockY=block_y, unionSize=union_size, isCausal=causal,
        capacity=capacity, topk=topk, seed=seed, qLengths=q_lengths, kvLengths=kv_lengths,
        cuSqlenQ=cu_q.tolist(), seqUsedKv=used_kv.tolist(),
        sparseIdx=indices.tolist(), sparseCount=counts.tolist(), groups=groups,
        union=result.unionSparseIdx.tolist(), counts=result.unionSparseCount.tolist(),
        mask=result.exactMask.tolist(),
    )


def _expected(data, head, group):
    """只读取输入，用集合和标量循环构造期望输出。"""
    g = data["groups"][group]
    selections = [row[:count] for row, count in zip(g["input"][head], g["counts"][head])]
    blocks = sorted({block for row in selections for block in row})
    packed = []
    for block in blocks:
        for lane in range(data["blockY"]):
            position = block * data["blockY"] + lane
            bits = 0
            for slot, selected in enumerate(selections):
                limit = g["kvLength"] - g["qLength"] + g["begin"] + slot
                if (block in selected and position < g["kvLength"]
                        and (not data["isCausal"] or position <= limit)):
                    bits |= 1 << slot
            packed.append(bits if bits < 2**31 else bits - 2**32)
    return dict(union=blocks + [-1] * (data["capacity"] - len(blocks)),
                count=len(blocks), mask=packed + [0] * ((data["capacity"] - len(blocks)) * data["blockY"]))


def _checks(data, head, group):
    expected = _expected(data, head, group)
    count = expected["count"]
    padding = count * data["blockY"]
    actual_union, actual_mask = data["union"][head][group], data["mask"][head][group]
    return [
        ("并集索引及顺序", actual_union == expected["union"]),
        ("有效块数量", data["counts"][head][group] == count),
        ("逐token整数掩码", actual_mask == expected["mask"]),
        ("填充区域", actual_union[count:] == [-1] * (data["capacity"] - count)
         and actual_mask[padding:] == [0] * ((data["capacity"] - count) * data["blockY"])),
    ]


def verify_demo(data):
    """验证全部 head 和分组，失败时保留输入、输出坐标与实际值。"""
    errors = []
    for head in range(data["heads"]):
        for group, g in enumerate(data["groups"]):
            expected = _expected(data, head, group)
            label = f"KV头={head}，请求={g['request']}，分组={group}"
            for name, want, actual in [
                ("并集索引", expected["union"], data["union"][head][group]),
                ("有效块数量", expected["count"], data["counts"][head][group]),
            ]:
                if want != actual:
                    errors.append(f"{label}：{name}期望={want}，实际={actual}")
            actual = data["mask"][head][group]
            if len(actual) != len(expected["mask"]):
                errors.append(f"{label}：掩码长度期望={len(expected['mask'])}，实际={len(actual)}")
            for column, (want, got) in enumerate(zip(expected["mask"], actual)):
                if want != got:
                    rank, lane = divmod(column, data["blockY"])
                    block = expected["union"][rank]
                    position = -1 if block < 0 else block * data["blockY"] + lane
                    errors.append(f"{label}：列={column}，KV位置={position}，期望={want}，实际={got}")
    return errors


def _width(text):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(text))


def _table(rows, color=False):
    widths = [max(_width(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        cells = []
        for value, width in zip(row, widths):
            text = str(value) + " " * (width - _width(value))
            if color and str(value) in ("不一致", "一致"):
                text = f"\033[{'31' if value == '不一致' else '32'}m{text}\033[0m"
            cells.append(text)
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def render_overview(data):
    total_q, groups = sum(data["qLengths"]), len(data["groups"])
    lines = [
        "输入参数与张量总览",
        f"  blockX={data['blockX']}  blockY={data['blockY']}  unionSize={data['unionSize']}  isCausal={data['isCausal']}",
        f"  maxUnionSparseCount={data['capacity']}",
        f"  sparseIdx    shape={[data['heads'], total_q, data['topk']]} dtype=int32",
        f"  sparseCount  shape={[data['heads'], total_q]} dtype=int32",
        f"  cuSqlenQ     shape={[len(data['cuSqlenQ'])]} dtype=int64 值={data['cuSqlenQ']}",
        f"  seqUsedKv    shape={[len(data['seqUsedKv'])]} dtype=int32 值={data['seqUsedKv']}",
        "  sparseIdx、sparseCount 的全部实际值在后续各组的输入表中展开。",
        f"输出形状（dtype 均为 int32）：unionSparseIdx={[data['heads'], groups, data['capacity']]}，"
        f"unionSparseCount={[data['heads'], groups]}，exactMask={[data['heads'], groups, data['capacity'] * data['blockY']]}",
    ]
    rows = [["请求", "输入Q行范围", "Q长度", "KV长度", "全局分组序号"]]
    for b, (q, kv) in enumerate(zip(data["qLengths"], data["kvLengths"])):
        ids = [i for i, g in enumerate(data["groups"]) if g["request"] == b]
        rows.append([b, f"[{data['cuSqlenQ'][b]},{data['cuSqlenQ'][b+1]})", q, kv, str(ids)])
    lines.append(_table(rows))
    return "\n".join(lines)


def render_group(data, head, group, color=False, tokens_per_panel=8, show_reasons=False):
    g, width = data["groups"][group], data["blockY"]
    union, packed = data["union"][head][group], data["mask"][head][group]
    expected = _expected(data, head, group)
    selections = [row[:n] for row, n in zip(g["input"][head], g["counts"][head])]
    lines = [f"\n{'=' * 64}", f"请求 {g['request']} → KV头 {head} → 请求内分组 {g['localGroup']}（全局分组 {group}）",
             f"① 输入张量切片：sparseIdx[{head},{g['inputStart']}:{g['inputStart']+g['rows']},:]；组内有效query={g['rows']}/{data['unionSize']}"]
    rows = [["组内位置", "请求内q", "输入Q行", "sparseIdx原始值", "sparseCount"]]
    for slot, (indices, n) in enumerate(zip(g["input"][head], g["counts"][head])):
        rows.append([slot, g["begin"] + slot, g["inputStart"] + slot, str(indices), n])
    lines.append(_table(rows))
    lines.append("② 输入选择矩阵：■=选中，·=未选；编号为请求内逻辑KV块")
    input_blocks = sorted(set(range((g["kvLength"] + width - 1) // width))
                          | set(expected["union"][:expected["count"]]))
    for start in range(0, max(1, len(input_blocks)), 8):
        blocks = input_blocks[start:start+8]
        rows = [["query", *[f"块{b} k{b*width}～{(b+1)*width-1}" for b in blocks]]]
        rows += [[f"q{g['begin']+slot}", *["■" if b in selected else "·" for b in blocks]] for slot, selected in enumerate(selections)]
        lines.append(_table(rows))
    lines += ["③ 实际输出张量（未过滤、未截断，包含填充）：",
              f"  unionSparseIdx[{head},{group},:] = {union}",
              f"  unionSparseCount[{head},{group}] = {data['counts'][head][group]}",
              f"  exactMask[{head},{group},:]："]
    for rank, block in enumerate(union):
        begin = rank * width
        lines.append(f"    列[{begin}:{begin+width}) 并集序号={rank} 块编号={block}：{packed[begin:begin+width]}")
    lines += ["④ 输出解码：每列是一个真实输出token位置；每个query行仅显示实际bit值0/1。",
              f"  二进制从高位到低位排列：slot {data['unionSize']-1} … slot 0；尾组缺少的query位保持0。"]
    if data["isCausal"]:
        lines.append("  本次启用因果约束：k <= KV长度 - Q长度 + 请求内q位置。")
    else:
        lines.append("  本次 isCausal=False，按输入块选择关系直接展开。")
    for start in range(0, len(packed), tokens_per_panel):
        columns = list(range(start, min(start + tokens_per_panel, len(packed))))
        blocks = [union[c // width] if c // width < len(union) else -1 for c in columns]
        positions = [-1 if b < 0 else b * width + c % width for b, c in zip(blocks, columns)]
        rows = [["输出列", *columns], ["并集序号/块", *[f"{c//width}/{b}" for c, b in zip(columns, blocks)]],
                ["KV位置", *["填充" if p < 0 else p for p in positions]],
                ["整数掩码", *[packed[c] for c in columns]],
                ["二进制掩码", *[format(packed[c] & 0xFFFFFFFF, f"0{data['unionSize']}b") for c in columns]]]
        for slot in range(g["rows"]):
            rows.append([f"组内位置 {slot} / q{g['begin']+slot}", *[(packed[c] & 0xFFFFFFFF) >> slot & 1 for c in columns]])
        rows.append(["对照期望值", *[expected["mask"][c] if c < len(expected["mask"]) else "多余" for c in columns]])
        rows.append(["逐列比较", *["一致" if c < len(expected["mask"]) and packed[c] == expected["mask"][c] else "不一致" for c in columns]])
        lines.append(_table(rows, color))
        if show_reasons:
            reasons = [["访问条件（独立计算）", *[f"列{c}" for c in columns]]]
            for slot, selected in enumerate(selections):
                limit = g["kvLength"] - g["qLength"] + g["begin"] + slot
                values = ["填充" if b < 0 else "越界" if p >= g["kvLength"] else "未选" if b not in selected
                          else "因果" if data["isCausal"] and p > limit else "可见" for b, p in zip(blocks, positions)]
                reasons.append([f"q{g['begin']+slot}", *values])
            lines.append(_table(reasons))
    lines += ["⑤ 本组验证", _table([["检查项", "结果"], *[[name, "一致" if ok else "不一致"] for name, ok in _checks(data, head, group)]], color)]
    return "\n".join(lines)


def render_compact(data, head, group, color=False):
    """一组输入与输出并排对应，token 按块拼成字符串，不重复展开二进制表。"""
    g = data["groups"][group]
    width = data["blockY"]
    union, packed = data["union"][head][group], data["mask"][head][group]
    visible_ranks = [r for r, b in enumerate(union) if b >= 0 or any(packed[r*width:(r+1)*width])]
    lines = [
        f"\n请求{g['request']} / KV头{head} / 请求内分组{g['localGroup']}（全局{group}）  Q={g['qLength']} KV={g['kvLength']}",
        "输入：sparseIdx 为KV块编号，-1为填充；sparseCount 为有效块数。",
    ]
    rows = [["query", "sparseIdx", "sparseCount"]]
    for slot, (indices, count) in enumerate(zip(g["input"][head], g["counts"][head])):
        rows.append([f"q{g['begin']+slot}", str(indices), count])
    lines += [_table(rows),
              f"输出 unionSparseIdx有效项={union[:data['counts'][head][group]]}，unionSparseCount={data['counts'][head][group]}",
              "输出 exactMask 解码：每段对应一个并集块，每个字符为一个token的实际bit。"]
    # Split wide block lists; normal padding is summarized below.
    columns = max(1, (shutil.get_terminal_size((120, 30)).columns - 12) // (max(width, 10) + 3))
    for begin in range(0, len(visible_ranks), columns):
        ranks = visible_ranks[begin:begin+columns]
        rows = [["并集块", *[str(union[r]) for r in ranks]],
                ["KV位置", *["填充" if union[r] < 0 else f"{union[r]*width}～{(union[r]+1)*width-1}" for r in ranks]]]
        for slot in range(g["rows"]):
            rows.append([f"q{g['begin']+slot}", *[
                "".join(str((v & 0xFFFFFFFF) >> slot & 1) for v in packed[r*width:(r+1)*width]) for r in ranks]])
        lines.append(_table(rows))
    lines.append("exactMask整数值（与上表各块对应）：")
    for rank in visible_ranks:
        lines.append(f"  并集序号{rank}/块{union[rank]}: {packed[rank*width:(rank+1)*width]}")
    hidden = len(union) - len(visible_ranks)
    if hidden:
        lines.append(f"折叠 {hidden} 个填充块：unionSparseIdx=-1，exactMask共 {hidden*width} 个0（全部参与验证）。")
    passed = all(ok for _, ok in _checks(data, head, group))
    lines.append(f"本组验证：{'一致' if passed else '不一致（详见末尾错误）'}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-lengths", default="11,5", help="各请求Q长度，逗号分隔")
    parser.add_argument("--kv-lengths", default="23,13", help="各请求KV长度，逗号分隔")
    parser.add_argument("--heads", type=int, default=2, help="KV头数")
    parser.add_argument("--block-y", type=int, default=8, help="每个KV块的token数")
    parser.add_argument("--union-size", type=int, default=8, help="每组query数量")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--max-union-sparse-count", type=int, default=513, help="并集容量，默认513")
    parser.add_argument("--head", type=int, help="仅显示指定KV头")
    parser.add_argument("--request", type=int, help="仅显示指定请求")
    parser.add_argument("--group", type=int, help="仅显示指定请求内分组")
    parser.add_argument("--all", action="store_true", help="显示全部分组，默认只显示第一组；不可与筛选同时使用")
    parser.add_argument("--detail", action="store_true", help="展示完整输入矩阵、整数/二进制对照和逐列验证")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True, help="启用因果约束（默认）；--no-causal 关闭")
    parser.add_argument("--show-reasons", action="store_true", help="使用详细模式并展示屏蔽原因矩阵")
    parser.add_argument("--tokens-per-panel", type=int, default=8, help="每张解码表最多展示多少token列")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="终端颜色")
    args = parser.parse_args(argv)
    if args.tokens_per_panel < 1:
        parser.error("每张表的token列数必须为正数")
    if args.all and any(x is not None for x in (args.head, args.request, args.group)):
        parser.error("--all 不可与 --head、--request、--group 同时使用")
    try:
        data = build_demo([int(x) for x in args.q_lengths.split(",")], [int(x) for x in args.kv_lengths.split(",")],
                          args.heads, args.block_y, args.union_size, args.seed, causal=args.causal,
                          max_union_sparse_count=args.max_union_sparse_count)
    except ValueError as exc:
        parser.error(str(exc))
    if args.request is not None and not 0 <= args.request < len(data["qLengths"]):
        parser.error("请求编号超出范围")
    if args.head is not None and not 0 <= args.head < data["heads"]:
        parser.error("KV头编号超出范围")
    if args.group is not None and args.group < 0:
        parser.error("分组编号不能为负数")
    targets = [(h, i) for i, g in enumerate(data["groups"])
               for h in range(data["heads"])
               if (args.head is None or h == args.head)
               and (args.request is None or g["request"] == args.request)
               and (args.group is None or g["localGroup"] == args.group)]
    targets.sort(key=lambda target: (data["groups"][target[1]]["request"], target[0], target[1]))
    if not args.all and all(x is None for x in (args.head, args.request, args.group)):
        targets = targets[:1]
    if not targets and any(x is not None for x in (args.head, args.request, args.group)):
        parser.error("筛选范围内没有有效query分组")
    errors = verify_demo(data)
    use_color = args.color == "always" or args.color == "auto" and sys.stdout.isatty()
    # Leave room for labels and the binary representation of every query bit.
    available = max(1, (shutil.get_terminal_size((120, 30)).columns - 26) // (max(data["unionSize"], 8) + 3))
    columns = min(args.tokens_per_panel, available)
    detail = args.detail or args.show_reasons
    if detail:
        print(render_overview(data))
    else:
        print(f"blockX={data['blockX']} blockY={data['blockY']} unionSize={data['unionSize']} isCausal={data['isCausal']}")
        print(f"maxUnionSparseCount={data['capacity']}")
        print(f"cuSqlenQ={data['cuSqlenQ']}  seqUsedKv={data['seqUsedKv']}")
    print(f"显示 {len(targets)} 个头/分组切片（共 {data['heads']*len(data['groups'])} 个）；验证覆盖全部输入。")
    for head, group in targets:
        if detail:
            print(render_group(data, head, group, color=use_color, tokens_per_panel=columns, show_reasons=args.show_reasons))
        else:
            print(render_compact(data, head, group, color=use_color))
    print(f"\n验证结果：{'失败' if errors else '通过'}（全部 {data['heads']} 个KV头、{len(data['groups'])} 个分组；随机种子={args.seed}）")
    for error in errors:
        print("  " + error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
