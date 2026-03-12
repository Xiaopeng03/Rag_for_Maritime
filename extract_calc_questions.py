# -*- coding: utf-8 -*-
"""
extract_calc_questions.py
从 maritimebench_test.json 中提取计算题。

策略：
  1. 先用正则快速筛选候选题（含具体数值 + 计算相关关键词）
  2. 再用 Qwen3-8B 对候选题做二次判断，确认是否为计算题
  3. 输出提取结果到 JSON 文件，并打印统计

用法：
  python extract_calc_questions.py                        # 默认输入输出
  python extract_calc_questions.py --no-llm               # 只用正则，不调用 LLM
  python extract_calc_questions.py --input other.json     # 指定输入文件
"""
import os
import re
import json
import time
import argparse
from openai import OpenAI

API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8790af9e6e8b42ac9621dd9578741970")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ── 正则预筛选规则 ──────────────────────────────────────────────
# 必须同时满足：含具体数值 AND 含计算相关词
NUM_PATTERN = re.compile(
    r'\d+[\.,。]\d+'           # 小数
    r'|\d{3,}'                 # 3位以上整数
    r'|[＝=×÷±√∑∫]'           # 数学符号
)

CALC_KEYWORDS = re.compile(
    r'排水量|吃水|GM|KM|KG|BM|GZ|GG|浮心|重心|漂心|稳性|力矩|横倾|纵倾'
    r'|航速|转速|功率|电压|电流|电阻|频率|波长|电容|电感|阻抗|效率|扭矩'
    r'|压力|温度|流量|热量|燃油|耗油|航程|航时'
    r'|方位|偏差|误差|经纬度|潮高|潮差|水深|船速|风速'
    r'|kN|kW|rpm|knot|海里|节|吨|kPa|MPa|kJ|kΩ|mA|Hz'
    r'|displacement|draft|speed|power|voltage|current|resistance|frequency'
    r'|calculate|compute|determine|find the|how many|how much'
)


def regex_filter(data: list) -> list:
    """用正则快速筛选候选计算题"""
    candidates = []
    for item in data:
        q = item['question']
        if NUM_PATTERN.search(q) and CALC_KEYWORDS.search(q):
            candidates.append(item)
    return candidates


# ── LLM 二次判断 ────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一位航运考试专家。
判断以下题目是否为"计算题"——即需要代入具体数值进行数学运算才能得出答案的题目。
纯概念题、判断题、选择知识点的题不算计算题。

只回答 YES 或 NO，不要输出其他内容。"""


def llm_is_calc(client: OpenAI, question: str) -> bool:
    """调用 LLM 判断是否为计算题，返回 True/False"""
    try:
        resp = client.chat.completions.create(
            model="qwen3-8b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"题目：{question}"}
            ],
            extra_body={"enable_thinking": False},
            max_tokens=5,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer.startswith("Y")
    except Exception as e:
        print(f"  ⚠️  LLM 调用失败: {e}")
        return True  # 失败时保留候选题


def llm_batch_filter(client: OpenAI, candidates: list,
                     batch_size: int = 15, delay: float = 0.2) -> list:
    """
    批量 LLM 判断：一次调用处理多道题，节省 API 次数。
    返回确认为计算题的列表。
    """
    confirmed = []
    total = len(candidates)

    for start in range(0, total, batch_size):
        batch = candidates[start: start + batch_size]
        numbered = "\n".join(
            f"{i+1}. {item['question'][:200]}"
            for i, item in enumerate(batch)
        )
        prompt = (
            f"以下 {len(batch)} 道题，哪些是计算题（需要代入数值运算）？\n"
            f"请只输出计算题的序号，用逗号分隔，例如：1,3,5\n"
            f"如果都不是，输出 NONE。\n\n{numbered}"
        )
        try:
            resp = client.chat.completions.create(
                model="qwen3-8b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                extra_body={"enable_thinking": False},
                max_tokens=50,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.upper() == "NONE":
                indices = []
            else:
                indices = [int(x.strip()) - 1
                           for x in re.findall(r'\d+', raw)
                           if 0 < int(x.strip()) <= len(batch)]
        except Exception as e:
            print(f"  ⚠️  批次失败，保留全部候选: {e}")
            indices = list(range(len(batch)))

        for idx in indices:
            confirmed.append(batch[idx])

        done = min(start + batch_size, total)
        print(f"  LLM 判断 [{done}/{total}]  本批确认 {len(indices)} 道计算题")

        if delay > 0 and start + batch_size < total:
            time.sleep(delay)

    return confirmed


def main():
    parser = argparse.ArgumentParser(description="从题库中提取计算题")
    parser.add_argument("--input", default="maritimebench_test.json",
                        help="输入 JSON 文件（默认 maritimebench_test.json）")
    parser.add_argument("--output", default="calc_questions.json",
                        help="输出 JSON 文件（默认 calc_questions.json）")
    parser.add_argument("--no-llm", action="store_true",
                        help="只用正则筛选，不调用 LLM 二次判断")
    parser.add_argument("--batch-size", type=int, default=15,
                        help="LLM 每批题目数（默认 15）")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="每批间隔秒数（默认 0.2）")
    args = parser.parse_args()

    # 读取数据
    print(f"📂 读取: {args.input}")
    with open(args.input, "rb") as f:
        data = json.loads(f.read().decode("utf-8"))
    print(f"   共 {len(data)} 道题")

    # 第一步：正则预筛选
    print("\n🔍 第一步：正则预筛选...")
    candidates = regex_filter(data)
    print(f"   候选计算题: {len(candidates)} 道")

    # 第二步：LLM 二次判断
    if args.no_llm:
        confirmed = candidates
        print("   跳过 LLM 判断（--no-llm）")
    else:
        print(f"\n🤖 第二步：LLM 二次判断（批大小={args.batch_size}）...")
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        confirmed = llm_batch_filter(client, candidates,
                                     batch_size=args.batch_size,
                                     delay=args.delay)

    print(f"\n✅ 最终确认计算题: {len(confirmed)} 道")

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(confirmed, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存到: {args.output}")

    # 打印预览
    print(f"\n📋 计算题预览（前 10 道）：")
    print("-" * 70)
    for i, item in enumerate(confirmed[:10]):
        q = item['question']
        display = q[:100] + ("..." if len(q) > 100 else "")
        print(f"[{i+1:3d}] #{item['index']:4d}  {display}")
        opts = "  ".join(f"{k}.{item.get(k,'')}" for k in 'ABCD' if item.get(k))
        print(f"       {opts}")
        print(f"       答案: {item.get('answer', '?')}")
        print()


if __name__ == "__main__":
    main()
