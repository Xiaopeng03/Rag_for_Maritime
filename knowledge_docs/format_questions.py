# -*- coding: utf-8 -*-
import json, os, re
from openai import OpenAI

API_KEY  = os.getenv("DASHSCOPE_API_KEY", "sk-8790af9e6e8b42ac9621dd9578741970")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
client   = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content.strip()


def split_by_article(content: str) -> list[tuple[str, str]]:
    """按条款切分，返回 [(条款名, 文本块), ...]"""
    parts = re.split(r'(第[一二三四五六七八九十]+条\s*\S+)', content)
    chunks = []
    i = 1
    while i < len(parts) - 1:
        title = parts[i].strip()
        body  = parts[i + 1]
        chunks.append((title, body))
        i += 2
    return chunks


def parse_chunk(article_title: str, chunk: str) -> list[dict]:
    prompt = f"""你是一个数据整理助手。下面是一段航海法规选择题原文，包含题目和答案键，格式比较混乱。

请提取其中所有选择题，严格按照以下 JSON 数组格式输出，不要输出任何其他内容：

[
  {{
    "type": "条款名称（如：适用范围、船舶、机动船等）",
    "question": "完整题干文字",
    "options": "A．... B．... C．... D．...",
    "answer": "单个字母，如 A"
  }},
  ...
]

规则：
- type 填当前条款或子分类名称，本段条款为：{article_title}
- question 只含题干，不含选项
- options 包含所有四个选项，保持原文
- answer 只填一个大写字母（A/B/C/D），从答案键中匹配；找不到填 "?"
- 答案键通常在文段末尾，格式如 "1-5、CCDDD" 或 "（1）DBD"

原文：
{chunk}
"""
    raw = call_llm(prompt)
    # 提取 JSON 数组
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if not m:
        print(f"  [警告] {article_title} 未解析到 JSON，跳过")
        return []
    try:
        items = json.loads(m.group())
        return items
    except json.JSONDecodeError as e:
        print(f"  [警告] {article_title} JSON 解析失败: {e}")
        return []


def main():
    input_file  = '选择题题库/第一章.txt'
    output_file = '整理后的题库_第一章.json'

    with open(input_file, 'r', encoding='gb18030', errors='ignore') as f:
        content = f.read()

    chunks = split_by_article(content)
    if not chunks:
        # 没有条款分隔，整体处理
        chunks = [('第一章', content)]

    print(f"共 {len(chunks)} 个条款块，开始调用 LLM...")

    all_results = []
    for idx, (title, body) in enumerate(chunks, 1):
        print(f"[{idx}/{len(chunks)}] 处理：{title} ...", flush=True)
        items = parse_chunk(title, body)
        print(f"  → 提取到 {len(items)} 道题", flush=True)
        all_results.extend(items)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共 {len(all_results)} 道题 → {output_file}")
    for item in all_results[:2]:
        print(item)


if __name__ == '__main__':
    main()