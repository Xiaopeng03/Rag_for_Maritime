# -*- coding: utf-8 -*-
"""
classify_questions.py
调用 Qwen3-8B 对 maritimebench_test.json 中的题目进行分类，
输出带 category 字段的 JSON 文件，并打印统计结果。

用法：
  python classify_questions.py                          # 分类全部题目
  python classify_questions.py --limit 100              # 只分类前 100 题（测试用）
  python classify_questions.py --input other.json       # 指定输入文件
  python classify_questions.py --batch-size 20          # 每批 20 题（节省 API 调用次数）
"""
import os
import sys
import json
import time
import argparse
from openai import OpenAI

API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8790af9e6e8b42ac9621dd9578741970")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ── 预定义类别 ──────────────────────────────────────────────────
CATEGORIES = [
    "船舶电气与电子",       # 电力系统、整流、电机、电路
    "轮机与动力装置",       # 柴油机、蒸汽、制冷、泵、轴系
    "航行与操纵",           # 避碰、操舵、航行规则、导航
    "船舶结构与稳性",       # 船体结构、稳性、载重线
    "消防与安全",           # 消防、救生、安全设备
    "船舶法规与证书",       # 海事法规、公约、证书、检验
    "船舶保安",             # ISPS、保安审核
    "货物运输与装卸",       # 货物积载、装卸、危险品
    "船舶保险与海商法",     # 保险、海商法、赔偿
    "通信与GMDSS",          # 无线电、GMDSS、气象
    "医疗急救",             # 急救、烧伤、止血、医疗
    "船舶管理与营运",       # 船员管理、营运、ISM
    "其他",                 # 不属于以上类别
]

CATEGORY_LIST_STR = "\n".join(f"{i+1}. {c}" for i, c in enumerate(CATEGORIES))

SYSTEM_PROMPT = f"""你是航运领域专家，负责对题目进行分类。
请将题目归入以下类别之一，只输出类别名称，不要输出任何解释：

{CATEGORY_LIST_STR}

如果题目属于多个类别，选择最主要的一个。"""


def classify_single(client: OpenAI, question: str) -> str:
    """单题分类，返回类别名称字符串"""
    resp = client.chat.completions.create(
        model="qwen3-8b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"题目：{question}"}
        ],
        extra_body={"enable_thinking": False},
        max_tokens=20,
    )
    raw = resp.choices[0].message.content.strip()
    # 匹配最接近的类别
    for cat in CATEGORIES:
        if cat in raw:
            return cat
    # 如果没有精确匹配，返回原始输出（截断）
    return raw[:20] if raw else "其他"


def classify_batch(client: OpenAI, questions: list[str]) -> list[str]:
    """
    批量分类：一次 API 调用处理多道题，节省请求次数。
    返回与 questions 等长的类别列表。
    """
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    prompt = (
        f"请对以下 {len(questions)} 道题目逐一分类，"
        f"每行输出格式为「序号. 类别名称」，不要输出其他内容。\n\n{numbered}"
    )
    resp = client.chat.completions.create(
        model="qwen3-8b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        extra_body={"enable_thinking": False},
        max_tokens=len(questions) * 15,
    )
    raw = resp.choices[0].message.content.strip()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    results = []
    for i, q in enumerate(questions):
        # 找对应行（序号 i+1）
        matched_cat = "其他"
        for line in lines:
            if line.startswith(f"{i+1}.") or line.startswith(f"{i+1}、"):
                for cat in CATEGORIES:
                    if cat in line:
                        matched_cat = cat
                        break
                break
        results.append(matched_cat)
    return results


def main():
    parser = argparse.ArgumentParser(description="用 Qwen3-8B 对题目进行分类")
    parser.add_argument("--input", default="maritimebench_test.json",
                        help="输入 JSON 文件（默认 maritimebench_test.json）")
    parser.add_argument("--output", default="maritimebench_classified.json",
                        help="输出 JSON 文件（默认 maritimebench_classified.json）")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 题（0=全部）")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="每批题目数量（默认 10，设为 1 则逐题调用）")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="每批之间的间隔秒数（默认 0.2）")
    parser.add_argument("--resume", action="store_true",
                        help="从已有输出文件断点续跑")
    args = parser.parse_args()

    # 读取数据
    with open(args.input, "rb") as f:
        data = json.loads(f.read().decode("utf-8"))

    if args.limit > 0:
        data = data[:args.limit]

    total = len(data)
    print(f"📋 共 {total} 道题，批大小={args.batch_size}")

    # 断点续跑：读取已有结果
    done_indices = set()
    results_map = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, "rb") as f:
            existing = json.loads(f.read().decode("utf-8"))
        for item in existing:
            if "category" in item:
                done_indices.add(item["index"])
                results_map[item["index"]] = item
        print(f"⏩ 已有 {len(done_indices)} 题，跳过")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 过滤未完成的题目
    todo = [item for item in data if item["index"] not in done_indices]
    print(f"🔄 待分类: {len(todo)} 题")

    # 分批处理
    batch_size = max(1, args.batch_size)
    processed = 0

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start: batch_start + batch_size]
        questions = [item["question"] for item in batch]

        try:
            if batch_size == 1:
                categories = [classify_single(client, questions[0])]
            else:
                categories = classify_batch(client, questions)
        except Exception as e:
            print(f"  ⚠️  批次失败，回退到逐题模式: {e}")
            categories = []
            for q in questions:
                try:
                    categories.append(classify_single(client, q))
                except Exception as e2:
                    print(f"    ❌ 单题失败: {e2}")
                    categories.append("其他")

        for item, cat in zip(batch, categories):
            item["category"] = cat
            results_map[item["index"]] = item

        processed += len(batch)
        pct = processed / len(todo) * 100
        print(f"  [{processed}/{len(todo)}] {pct:.0f}%  最新: {categories[-1]}")

        if args.delay > 0 and batch_start + batch_size < len(todo):
            time.sleep(args.delay)

    # 合并并按 index 排序输出
    all_results = sorted(results_map.values(), key=lambda x: x["index"])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 结果已保存到: {args.output}")

    # 统计
    from collections import Counter
    counter = Counter(item.get("category", "其他") for item in all_results)
    print("\n📊 分类统计：")
    print(f"{'类别':<20} {'数量':>6}  {'占比':>6}")
    print("-" * 38)
    for cat, count in sorted(counter.items(), key=lambda x: -x[1]):
        pct = count / len(all_results) * 100
        print(f"{cat:<20} {count:>6}  {pct:>5.1f}%")
    print(f"\n合计: {len(all_results)} 题")


if __name__ == "__main__":
    main()
