# -*- coding: utf-8 -*-
"""
MaritimeBench 答题系统
使用 RAG 增强的 Qwen3-8B 回答航运知识题目
"""
import os
import sys

# 添加 RAG 模块到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RAG'))

from RAG.rag_engine import RAGMaritimeQA
import re


def clean_context(text):
    """
    清理参考资料文本，使其更易读

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    if not text:
        return ""

    # 移除多余的空行
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # 移除行首的特殊字符和数字标记（如 "1→", "2→" 等）
    text = re.sub(r'^\s*\d+→\s*', '', text, flags=re.MULTILINE)

    # 移除一些常见的格式问题字符
    text = text.replace('\u3000', ' ')  # 全角空格
    text = text.replace('\xa0', ' ')    # 不间断空格

    # 移除连续的感叹号或特殊符号
    text = re.sub(r'!+', '', text)
    text = re.sub(r'#+', '', text)
    text = re.sub(r'\$+', '', text)
    text = re.sub(r'%+', '', text)

    # 规范化空格
    text = re.sub(r' +', ' ', text)

    return text.strip()


def parse_question_with_options(text):
    """
    解析包含选项的题目文本

    Args:
        text: 完整的题目文本，包括选项

    Returns:
        tuple: (question, options_dict)
    """
    lines = text.strip().split('\n')

    # 查找选项开始的位置
    option_start_idx = -1
    for i, line in enumerate(lines):
        # 匹配 "选项：" 或直接以 "A." "A、" "A " 开头的行
        if re.match(r'^选项[：:]\s*$', line.strip()):
            option_start_idx = i + 1
            break
        elif re.match(r'^[A-D][.、\s]', line.strip()):
            option_start_idx = i
            break

    # 分离题目和选项
    if option_start_idx > 0:
        question_lines = lines[:option_start_idx]
        option_lines = lines[option_start_idx:]
    else:
        # 如果没有找到选项标记，尝试从最后几行提取选项
        question_lines = []
        option_lines = []
        for i, line in enumerate(lines):
            if re.match(r'^[A-D][.、\s]', line.strip()):
                question_lines = lines[:i]
                option_lines = lines[i:]
                break

        if not option_lines:
            # 没有找到选项，整个文本作为题目
            return text.strip(), {}

    # 清理题目（移除"选项："标记）
    question = '\n'.join(question_lines).strip()
    question = re.sub(r'选项[：:]\s*$', '', question).strip()

    # 解析选项
    options = {}
    for line in option_lines:
        line = line.strip()
        if not line:
            continue

        # 匹配 A. B. C. D. 或 A、B、C、D、格式
        match = re.match(r'^([A-D])[.、\s]+(.+)$', line)
        if match:
            letter = match.group(1)
            content = match.group(2).strip()
            options[letter] = content

    return question, options


def main():
    """主函数"""
    print("=" * 60)
    print("🚢 MaritimeBench 航运知识问答系统")
    print("=" * 60)

    # API 配置
    API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8790af9e6e8b42ac9621dd9578741970")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    VECTORSTORE_PATH = "./RAG/vectorstore"

    # 初始化 RAG 引擎
    engine = RAGMaritimeQA(
        api_key=API_KEY,
        base_url=BASE_URL,
        vectorstore_path=VECTORSTORE_PATH
    )

    print("\n" + "=" * 60)
    print("💡 使用说明:")
    print("  1. 一次性输入完整题目（包括选项）")
    print("  2. 输入完成后按两次回车")
    print("  3. 系统会自动进行RAG检索并回答")
    print("  4. 输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    # 交互式问答
    while True:
        print("\n" + "-" * 60)
        print("📝 请输入完整题目（包括选项），输入 'quit' 退出")
        print("   格式示例：")
        print("   题目内容")
        print("   选项：")
        print("   A. 选项A")
        print("   B. 选项B")
        print("   C. 选项C")
        print("   D. 选项D")
        print("\n请输入（输入完成后按两次回车）:")

        # 读取多行输入
        lines = []
        empty_count = 0
        while True:
            line = input()
            if line.strip().lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                return

            if not line.strip():
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
                lines.append(line)

        if not lines:
            print("⚠️  题目不能为空")
            continue

        # 解析输入内容
        full_text = "\n".join(lines)
        question, options = parse_question_with_options(full_text)

        if not question:
            print("⚠️  无法解析题目")
            continue

        # 回答问题
        print("\n🤔 正在思考...")
        answer, context = engine.answer_question(question, options)

        print("\n" + "=" * 60)
        print(f"✅ 答案: {answer}")

        if context and engine.use_rag:
            print("\n📚 参考资料:")
            print("-" * 60)
            # 清理并显示参考资料
            cleaned_context = clean_context(context)
            # 显示前 500 字符
            display_text = cleaned_context[:500]
            print(display_text)
            if len(cleaned_context) > 500:
                print("\n... (更多内容已省略)")
        print("=" * 60)


def batch_mode():
    """批量处理模式"""
    print("=" * 60)
    print("🚢 MaritimeBench 批量推理模式")
    print("=" * 60)

    # API 配置
    API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8790af9e6e8b42ac9621dd9578741970")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    VECTORSTORE_PATH = "./RAG/vectorstore"

    # 初始化 RAG 引擎
    engine = RAGMaritimeQA(
        api_key=API_KEY,
        base_url=BASE_URL,
        vectorstore_path=VECTORSTORE_PATH
    )

    # 示例：批量处理题目
    sample_questions = [
        {
            'question': '船舶电力系统供电网络中，放射形网络的特点是______。①发散形传输②环形传输③缺乏冗余④冗余性能好',
            'options': {'A': '②③', 'B': '①③', 'C': '②④', 'D': '①④'}
        },
        {
            'question': '在单相桥式整流电路中，如果一只整流二极管被击穿，则______。',
            'options': {
                'A': '输出电压降低一半',
                'B': '输出电压降低至四分之一',
                'C': '输入交流电源短路',
                'D': '电路工作正常'
            }
        }
    ]

    # 批量推理
    output_file = "predictions.jsonl"
    engine.batch_inference(sample_questions, output_file)


if __name__ == '__main__':
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        batch_mode()
    else:
        main()