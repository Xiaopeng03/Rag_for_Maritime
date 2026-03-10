# -*- coding: utf-8 -*-
"""
简化版答题系统（无 RAG）
直接使用 Qwen3-8B 回答问题
"""
import os
import re
from openai import OpenAI


class SimpleQA:
    """简单问答系统（无 RAG）"""

    def __init__(self, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt = """你是航运领域专家。请回答单选题。

要求：
1. 只输出选项字母（A/B/C/D）
2. 必须用方括号包裹，如 [A]
3. 不要输出任何解释或额外内容"""

    def extract_answer(self, response):
        """提取答案"""
        match = re.search(r'\[([A-D])\]', response)
        if match:
            return match.group(1)
        match = re.search(r'\b([A-D])\b', response)
        if match:
            return match.group(1)
        return 'A'

    def answer_question(self, question, options=None):
        """回答问题"""
        if options is None:
            options = {'A': '', 'B': '', 'C': '', 'D': ''}

        prompt = f"""题目：{question}
选项：
A. {options.get('A', '')}
B. {options.get('B', '')}
C. {options.get('C', '')}
D. {options.get('D', '')}

答："""

        try:
            response = self.client.chat.completions.create(
                model="qwen3-8b",
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                extra_body={"enable_thinking": False}
            )

            answer_text = response.choices[0].message.content
            answer = self.extract_answer(answer_text)
            return f'[{answer}]'

        except Exception as e:
            print(f"❌ API 调用失败: {e}")
            return '[A]'


def main():
    """主函数"""
    print("=" * 60)
    print("🚢 MaritimeBench 航运知识问答系统（简化版）")
    print("=" * 60)

    API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8790af9e6e8b42ac9621dd9578741970")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    engine = SimpleQA(api_key=API_KEY, base_url=BASE_URL)

    print("\n💡 使用说明:")
    print("  1. 输入题目和选项")
    print("  2. 系统会直接调用 Qwen3-8B 回答")
    print("  3. 输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        print("\n" + "-" * 60)
        question = input("📝 请输入题目（或输入 quit 退出）: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break

        if not question:
            print("⚠️  题目不能为空")
            continue

        print("\n请输入选项（直接回车跳过）:")
        options = {}
        for letter in ['A', 'B', 'C', 'D']:
            opt = input(f"  {letter}. ").strip()
            options[letter] = opt

        print("\n🤔 正在思考...")
        answer = engine.answer_question(question, options)

        print("\n" + "=" * 60)
        print(f"✅ 答案: {answer}")
        print("=" * 60)


if __name__ == '__main__':
    main()