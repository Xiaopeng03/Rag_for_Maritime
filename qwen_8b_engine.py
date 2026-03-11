# -*- coding: utf-8 -*-
"""
qwen_8b_engine.py
批量推理引擎：读取题目 JSONL 文件，调用 Qwen3-8B（RAG 增强），生成符合评测格式的 JSONL 输出。

输入格式（每行一个 JSON）：
  {"index": 0, "input": "**User**: \n请回答单选题...\n 当前题目\n 题目文本\n选项：\nA. ...\nB. ...", "target": "A"}

输出格式（与 Q2-sample-submit.jsonl 一致）：
  {"index": 0, "input": "...", "target": "A",
   "sample_score": {"score": {"value": {"acc": 1.0}, "extracted_prediction": "A",
                               "prediction": "[A]", "explanation": null,
                               "metadata": {}, "main_score_name": null},
                    "sample_id": 0, "group_id": 0, "sample_metadata": {}}}
"""
import os
import sys
import re
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RAG'))

from openai import OpenAI

# ── 可选：RAG 增强 ──────────────────────────────────────────────
try:
    from RAG.rag_engine import RAGMaritimeQA
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8790af9e6e8b42ac9621dd9578741970")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VECTORSTORE_PATH = "./RAG/vectorstore"

# ── 系统提示词 ──────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是航运领域专家。请回答单选题。\n"
    "要求：\n"
    "1. 只输出选项字母（A/B/C/D）\n"
    "2. 必须用方括号包裹，如 [A]\n"
    "3. 不要输出任何解释或额外内容"
)

SYSTEM_PROMPT_RAG = (
    "你是航运领域专家。请根据提供的参考资料回答单选题。\n"
    "要求：\n"
    "1. 只输出选项字母（A/B/C/D）\n"
    "2. 必须用方括号包裹，如 [A]\n"
    "3. 不要输出任何解释或额外内容"
)


# ── 解析 input 字段 ─────────────────────────────────────────────
def parse_input_field(input_text: str):
    """
    从 JSONL input 字段中提取题目文本和选项字典。
    input 字段格式：
      **User**: \n请回答单选题...\n 当前题目\n 题目内容\n选项：\nA. ...\nB. ...
    """
    # 取"当前题目"之后的内容
    marker = "当前题目"
    idx = input_text.find(marker)
    if idx != -1:
        body = input_text[idx + len(marker):].strip()
    else:
        body = input_text.strip()

    lines = [l.strip() for l in body.split('\n') if l.strip()]

    # 分离题目行和选项行
    question_lines = []
    option_lines = []
    in_options = False
    for line in lines:
        if re.match(r'^选项[：:]\s*$', line):
            in_options = True
            continue
        if re.match(r'^[A-D][.、\s]', line):
            in_options = True
        if in_options:
            option_lines.append(line)
        else:
            question_lines.append(line)

    question = ' '.join(question_lines).strip()
    options = {}
    for line in option_lines:
        m = re.match(r'^([A-D])[.、\s]+(.+)$', line)
        if m:
            options[m.group(1)] = m.group(2).strip()

    return question, options


# ── 提取答案 ────────────────────────────────────────────────────
def extract_answer(text: str) -> str:
    m = re.search(r'\[([A-D])\]', text)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    return 'A'


# ── 构建评测格式的 sample_score ─────────────────────────────────
def build_sample_score(prediction: str, target: str, sample_id: int):
    extracted = extract_answer(prediction)
    acc = 1.0 if extracted == target else 0.0
    return {
        "score": {
            "value": {"acc": acc},
            "extracted_prediction": extracted,
            "prediction": prediction,
            "explanation": None,
            "metadata": {},
            "main_score_name": None
        },
        "sample_id": sample_id,
        "group_id": sample_id,
        "sample_metadata": {}
    }


# ── 主推理类 ────────────────────────────────────────────────────
class Qwen8BEngine:
    def __init__(self, use_rag: bool = True):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.use_rag = False
        self.rag_engine = None

        if use_rag and RAG_AVAILABLE:
            try:
                self.rag_engine = RAGMaritimeQA(
                    api_key=API_KEY,
                    base_url=BASE_URL,
                    vectorstore_path=VECTORSTORE_PATH
                )
                self.use_rag = self.rag_engine.use_rag
            except Exception as e:
                print(f"⚠️  RAG 初始化失败，使用无 RAG 模式: {e}")

    def answer_one(self, question: str, options: dict) -> str:
        """返回格式化答案字符串，如 [A]"""
        context = ""
        if self.use_rag and self.rag_engine:
            context = self.rag_engine.retrieve_knowledge(question)

        if context:
            prompt = (
                f"参考资料：\n{context}\n\n"
                f"请根据以上参考资料回答单选题：\n\n"
                f"题目：{question}\n选项：\n"
                + "\n".join(f"{k}. {v}" for k, v in options.items())
                + "\n\n答："
            )
            sys_prompt = SYSTEM_PROMPT_RAG
        else:
            prompt = (
                f"题目：{question}\n选项：\n"
                + "\n".join(f"{k}. {v}" for k, v in options.items())
                + "\n\n答："
            )
            sys_prompt = SYSTEM_PROMPT

        resp = self.client.chat.completions.create(
            model="qwen3-8b",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            extra_body={"enable_thinking": False}
        )
        raw = resp.choices[0].message.content.strip()
        letter = extract_answer(raw)
        return f"[{letter}]"

    def run_batch(self, input_file: str, output_file: str,
                  delay: float = 0.3, progress_cb=None):
        """
        批量推理。
        input_file: 每行一个 JSON，必须含 index / input 字段，target 可选。
        output_file: 输出 JSONL，格式与 Q2-sample-submit.jsonl 一致。
        delay: 每题之间的间隔秒数，避免限流。
        progress_cb: 可选回调 fn(done, total, index, prediction, correct)
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            records = [json.loads(l) for l in f if l.strip()]

        total = len(records)
        print(f"📋 共 {total} 道题，开始推理...")

        correct = 0
        results = []

        with open(output_file, 'w', encoding='utf-8') as out:
            for i, rec in enumerate(records):
                idx = rec.get('index', i)
                input_text = rec.get('input', '')
                target = rec.get('target', '')

                question, options = parse_input_field(input_text)

                try:
                    prediction = self.answer_one(question, options)
                except Exception as e:
                    print(f"  ❌ 题目 {idx} 失败: {e}")
                    prediction = "[A]"

                extracted = extract_answer(prediction)
                is_correct = (extracted == target) if target else None
                if is_correct:
                    correct += 1

                result = {
                    "index": idx,
                    "input": input_text,
                    "target": target,
                    "sample_score": build_sample_score(prediction, target, idx)
                }
                results.append(result)
                out.write(json.dumps(result, ensure_ascii=False) + '\n')
                out.flush()

                status = "✅" if is_correct else ("❓" if is_correct is None else "❌")
                print(f"  [{i+1}/{total}] 题目 {idx}: {prediction} {status}")

                if progress_cb:
                    progress_cb(i + 1, total, idx, prediction, is_correct)

                if delay > 0 and i < total - 1:
                    time.sleep(delay)

        if any(r.get('target') for r in records):
            acc = correct / total * 100
            print(f"\n🎯 准确率: {correct}/{total} = {acc:.1f}%")
        print(f"✅ 结果已保存到: {output_file}")
        return results


# ── CLI 入口 ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Qwen3-8B 批量推理，生成评测 JSONL")
    parser.add_argument('input', help='输入 JSONL 文件路径')
    parser.add_argument('-o', '--output', default='predictions.jsonl',
                        help='输出 JSONL 文件路径（默认 predictions.jsonl）')
    parser.add_argument('--no-rag', action='store_true', help='禁用 RAG 检索')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='每题间隔秒数（默认 0.3）')
    args = parser.parse_args()

    engine = Qwen8BEngine(use_rag=not args.no_rag)
    engine.run_batch(args.input, args.output, delay=args.delay)


if __name__ == '__main__':
    main()