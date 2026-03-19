# -*- coding: utf-8 -*-
"""
使用 Qwen3-8B + RAG 批量处理 MaritimeBench 测试集
生成答案并计算准确率
"""
import json
import os
import sys
from tqdm import tqdm

# 添加 RAG 模块到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RAG'))
from RAG.rag_engine import RAGMaritimeQA


def load_test_data(file_path):
    """加载测试数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_answer(response_text):
    """从模型回复中提取答案"""
    import re

    # 匹配 [A] [B] [C] [D] 格式
    match = re.search(r'\[([A-D])\]', response_text)
    if match:
        return match.group(1)

    # 匹配单独的字母
    match = re.search(r'\b([A-D])\b', response_text)
    if match:
        return match.group(1)

    # 默认返回 A
    return 'A'


def generate_answers_with_rag(data, rag_engine):
    """使用RAG批量生成答案"""
    results = []
    correct_count = 0

    print(f"开始处理 {len(data)} 道题目（使用RAG增强）...")

    for item in tqdm(data, desc="生成答案"):
        try:
            # 构建选项字典
            options = {
                'A': item['A'],
                'B': item['B'],
                'C': item['C'],
                'D': item['D']
            }

            # 调用RAG引擎回答问题
            answer_with_bracket, context = rag_engine.answer_question(
                item['question'],
                options
            )

            # 提取答案字母（去掉方括号）
            predicted_answer = extract_answer(answer_with_bracket)

            # 判断正确性
            is_correct = (predicted_answer == item['answer'])
            if is_correct:
                correct_count += 1

            # 保存结果
            result = {
                'index': item['index'],
                'question': item['question'],
                'A': item['A'],
                'B': item['B'],
                'C': item['C'],
                'D': item['D'],
                'ground_truth': item['answer'],
                'prediction': predicted_answer,
                'is_correct': is_correct,
                'raw_response': answer_with_bracket,
                'rag_context': context[:500] if context else ""  # 保存前500字符的RAG上下文
            }
            results.append(result)

        except Exception as e:
            print(f"\n错误：题目 {item['index']} 处理失败: {e}")
            # 记录失败的题目
            result = {
                'index': item['index'],
                'question': item['question'],
                'A': item['A'],
                'B': item['B'],
                'C': item['C'],
                'D': item['D'],
                'ground_truth': item['answer'],
                'prediction': 'ERROR',
                'is_correct': False,
                'raw_response': str(e),
                'rag_context': ""
            }
            results.append(result)

    # 计算准确率
    accuracy = correct_count / len(data) * 100 if len(data) > 0 else 0

    return results, correct_count, accuracy


def save_results(results, output_file):
    """保存结果为JSONL格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def save_summary(total, correct, accuracy, output_file):
    """保存统计摘要"""
    summary = {
        'model': 'Qwen3-8B + RAG',
        'total_questions': total,
        'correct_answers': correct,
        'wrong_answers': total - correct,
        'accuracy': f"{accuracy:.2f}%"
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    print("=" * 60)
    print("🚢 MaritimeBench 批量评测系统 (RAG增强)")
    print("=" * 60)

    # 配置
    API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8790af9e6e8b42ac9621dd9578741970")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    VECTORSTORE_PATH = "./RAG/vectorstore"
    INPUT_FILE = "maritimebench_test.json"
    OUTPUT_FILE = "predictions_rag.jsonl"
    SUMMARY_FILE = "evaluation_summary_rag.json"

    # 初始化 RAG 引擎
    print(f"\n🚀 初始化 RAG 引擎...")
    rag_engine = RAGMaritimeQA(
        api_key=API_KEY,
        base_url=BASE_URL,
        vectorstore_path=VECTORSTORE_PATH
    )

    # 加载数据
    print(f"\n📂 加载测试数据: {INPUT_FILE}")
    data = load_test_data(INPUT_FILE)
    print(f"✅ 加载完成，共 {len(data)} 道题目")

    # 生成答案
    print(f"\n🤖 使用 Qwen3-8B + RAG 生成答案...")
    results, correct_count, accuracy = generate_answers_with_rag(data, rag_engine)

    # 保存结果
    print(f"\n💾 保存结果到: {OUTPUT_FILE}")
    save_results(results, OUTPUT_FILE)

    print(f"💾 保存统计摘要到: {SUMMARY_FILE}")
    save_summary(len(data), correct_count, accuracy, SUMMARY_FILE)

    # 显示结果
    print("\n" + "=" * 60)
    print("📊 评测结果 (RAG增强)")
    print("=" * 60)
    print(f"模型: Qwen3-8B + RAG")
    print(f"总题目数: {len(data)}")
    print(f"正确数量: {correct_count}")
    print(f"错误数量: {len(data) - correct_count}")
    print(f"准确率: {accuracy:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()