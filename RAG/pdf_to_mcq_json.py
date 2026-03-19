# -*- coding: utf-8 -*-
"""
选择题题库 PDF → 结构化 JSON
每道题解析为：
  {
    "chunk_id": "避碰选择题题库.pdf_001",
    "source": "避碰选择题题库.pdf",
    "question_number": 1,
    "chapter": "第1章 习题集",        # 当前所在章节（若PDF有章节标题）
    "question": "题干文本",
    "options": {
      "A": "选项A内容",
      "B": "选项B内容",
      "C": "选项C内容",
      "D": "选项D内容"
    },
    "answer": null,                   # 初级阶段暂无答案，留空
    "raw_text": "原始完整文本",        # 保留原始文本，信息最大化
    "page": 1                         # 所在页码
  }
"""

import os
import re
import json
import argparse
from collections import defaultdict

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("请安装 PyMuPDF: pip install pymupdf")


# ── 正则模式 ──────────────────────────────────────────────────────────────────

# 题号行：以数字开头，后跟 ". " 或 "．" 或 "、"，数字可能有空格
# 例：1. xxx  /  12．xxx  /  123. xxx
RE_QUESTION_START = re.compile(r'^(\d{1,4})[．.\s、]\s*(.+)')

# 选项行：A/B/C/D 后跟 ．或 . 或 、或空格
RE_OPTION = re.compile(r'^([A-Da-d])[．.\s、]\s*(.+)')

# 章节标题（字体大/加粗行，或符合常见中文标题格式）
RE_CHAPTER = re.compile(
    r'^(?:第[一二三四五六七八九十百\d]+[章节篇部]|'
    r'\d+(?:\.\d+)*\s+[^\d]|'
    r'[一二三四五六七八九十]+[、．]\s*\S)'
)


# ── PDF 提取（保留页码信息）────────────────────────────────────────────────────

def extract_lines_with_pages(pdf_path: str):
    """
    返回 [(page_num, text, font_size, is_bold), ...]
    page_num 从 1 开始
    """
    doc = fitz.open(pdf_path)
    lines = []
    for page_idx, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                max_size = max(s["size"] for s in spans)
                is_bold = any(
                    "Bold" in s.get("font", "") or "bold" in s.get("font", "")
                    for s in spans
                )
                lines.append((page_idx + 1, text, max_size, is_bold))
    doc.close()
    return lines


# ── 推断正文字体大小 ──────────────────────────────────────────────────────────

def get_body_size(lines):
    from collections import Counter
    c = Counter(size for _, _, size, _ in lines)
    return c.most_common(1)[0][0] if c else 12.0


# ── 核心解析：将行列表解析为题目列表 ─────────────────────────────────────────

def parse_questions(lines_with_pages, body_size):
    """
    状态机解析：
      IDLE -> 遇到题号行 -> IN_QUESTION
      IN_QUESTION -> 遇到选项行 -> IN_OPTIONS
      IN_OPTIONS -> 遇到下一题号 -> 保存当前题，开始新题
    """
    questions = []
    current = None
    current_page = 1
    current_chapter = ""

    def flush():
        if current is None:
            return
        # 补全缺失选项为 null
        for opt in "ABCD":
            if opt not in current["options"]:
                current["options"][opt] = None
        # 构建 raw_text
        opts_text = "\n".join(
            f"{k}．{v}" for k, v in current["options"].items() if v
        )
        current["raw_text"] = current["question"] + "\n" + opts_text
        questions.append(current)

    for page_num, text, size, is_bold in lines_with_pages:
        # 检测章节标题（字体比正文大，或符合章节正则）
        if size > body_size * 1.1 or (is_bold and size >= body_size and len(text) < 40):
            if RE_CHAPTER.match(text) or (size > body_size * 1.1 and len(text) < 60):
                current_chapter = text
                continue

        # 尝试匹配题号
        m_q = RE_QUESTION_START.match(text)
        if m_q:
            flush()
            q_num = int(m_q.group(1))
            q_text = m_q.group(2).strip()
            current = {
                "question_number": q_num,
                "chapter": current_chapter,
                "question": q_text,
                "options": {},
                "answer": None,
                "page": page_num,
                "source": "",
                "chunk_id": "",
                "raw_text": "",
            }
            current_page = page_num
            continue

        # 尝试匹配选项
        m_opt = RE_OPTION.match(text)
        if m_opt and current is not None:
            key = m_opt.group(1).upper()
            val = m_opt.group(2).strip()
            if key in current["options"] and current["options"][key]:
                # 同一选项续行，拼接
                current["options"][key] += val
            else:
                current["options"][key] = val
            continue

        # 其他行：续接到当前题干或最后一个选项
        if current is not None:
            if current["options"]:
                # 已有选项，续接到最后一个选项
                last_key = list(current["options"].keys())[-1]
                current["options"][last_key] += text
            else:
                # 还没有选项，续接到题干
                current["question"] += text

    flush()
    return questions


# ── 主函数 ────────────────────────────────────────────────────────────────────

def pdf_to_mcq_json(pdf_path: str, output_path: str):
    source = os.path.basename(pdf_path)
    print(f"读取: {source}")

    lines = extract_lines_with_pages(pdf_path)
    print(f"  提取行数: {len(lines)}")

    body_size = get_body_size(lines)
    print(f"  正文字体大小: {body_size:.1f}pt")

    questions = parse_questions(lines, body_size)
    print(f"  解析题目数: {len(questions)}")

    # 填充 source 和 chunk_id
    for q in questions:
        q["source"] = source
        q["chunk_id"] = f"{source}_{q['question_number']:04d}"

    # 统计章节分布
    chapter_counts = defaultdict(int)
    for q in questions:
        chapter_counts[q["chapter"] or "(无章节)"] += 1
    print("  章节分布:")
    for ch, cnt in sorted(chapter_counts.items()):
        print(f"    {ch}: {cnt} 题")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\n完成，共 {len(questions)} 道题，已保存到: {output_path}")
    return questions


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将选择题题库 PDF 解析为结构化 JSON")
    parser.add_argument(
        "--input",
        default="../knowledge_docs/避碰选择题题库.pdf",
        help="输入 PDF 路径"
    )
    parser.add_argument(
        "--output",
        default="cleaned_data/mcq_chunks.json",
        help="输出 JSON 路径"
    )
    args = parser.parse_args()
    pdf_to_mcq_json(args.input, args.output)
