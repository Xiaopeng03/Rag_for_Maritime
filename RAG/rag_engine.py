# -*- coding: utf-8 -*-
"""
RAG 增强推理引擎
结合向量检索和 Qwen3-8B 进行航运知识问答
"""
import re
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAGMaritimeQA:
    """RAG 增强的航运知识问答系统"""

    def __init__(self, api_key, base_url, vectorstore_path):
        """
        初始化 RAG 引擎

        Args:
            api_key: 阿里云 API Key
            base_url: API 基础 URL
            vectorstore_path: 向量库路径
        """
        print("🚀 初始化 RAG 引擎...")

        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print("✅ API 客户端初始化完成")

        # 加载向量知识库
        if os.path.exists(vectorstore_path):
            print(f"📚 加载向量知识库: {vectorstore_path}")
            embeddings = HuggingFaceEmbeddings(
                model_name='BAAI/bge-small-zh-v1.5',
                model_kwargs={'device': 'cpu'}
            )
            self.vectorstore = FAISS.load_local(
                vectorstore_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ 知识库加载完成，共 {self.vectorstore.index.ntotal} 个向量")
            self.use_rag = True
        else:
            print(f"⚠️  向量库不存在: {vectorstore_path}")
            print("⚠️  将使用无 RAG 模式")
            self.vectorstore = None
            self.use_rag = False

        # 系统提示词
        self.system_prompt = """你是航运领域专家。请根据提供的参考资料回答单选题。

要求：
1. 只输出选项字母（A/B/C/D）
2. 必须用方括号包裹，如 [A]
3. 不要输出任何解释或额外内容"""

        self.system_prompt_no_rag = """你是航运领域专家。请回答单选题。

要求：
1. 只输出选项字母（A/B/C/D）
2. 必须用方括号包裹，如 [A]
3. 不要输出任何解释或额外内容"""

    def retrieve_knowledge(self, question, k=5):
        """
        检索相关知识

        Args:
            question: 问题文本
            k: 返回前 k 个最相关的文档

        Returns:
            相关知识的文本内容
        """
        if not self.use_rag:
            return ""

        # 使用相似度搜索，返回文档和分数
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)

        # 过滤低相关性文档（分数越低越相关，FAISS使用L2距离）
        # 只保留分数较低的文档（更相关的）
        filtered_docs = [(doc, score) for doc, score in docs_with_scores if score < 1.5]

        if not filtered_docs:
            # 如果过滤后没有文档，至少保留最相关的一个
            filtered_docs = [docs_with_scores[0]] if docs_with_scores else []

        # 拼接文档内容
        context_parts = []
        for doc, score in filtered_docs:
            context_parts.append(doc.page_content)

        context = "\n\n---\n\n".join(context_parts)
        return context

    def format_prompt(self, question, options, context=""):
        """
        格式化 prompt

        Args:
            question: 题目
            options: 选项字典 {'A': '...', 'B': '...', ...}
            context: 参考资料（可选）

        Returns:
            格式化后的 prompt
        """
        if context:
            prompt = f"""参考资料：
{context}

请根据以上参考资料回答单选题：

题目：{question}
选项：
A. {options.get('A', '')}
B. {options.get('B', '')}
C. {options.get('C', '')}
D. {options.get('D', '')}

答："""
        else:
            prompt = f"""题目：{question}
选项：
A. {options.get('A', '')}
B. {options.get('B', '')}
C. {options.get('C', '')}
D. {options.get('D', '')}

答："""

        return prompt

    def extract_answer(self, response):
        """
        提取答案（符合评分规则）

        Args:
            response: 模型返回的文本

        Returns:
            提取的答案字母（A/B/C/D）
        """
        match = re.search(r'\[([A-D])\]', response)
        if match:
            return match.group(1)

        # 如果没有方括号，尝试直接匹配字母
        match = re.search(r'\b([A-D])\b', response)
        if match:
            return match.group(1)

        # 默认返回 A
        return 'A'

    def answer_question(self, question, options=None):
        """
        回答单个问题

        Args:
            question: 题目文本
            options: 选项字典（可选）

        Returns:
            答案，格式为 [A]/[B]/[C]/[D]
        """
        if options is None:
            options = {'A': '', 'B': '', 'C': '', 'D': ''}

        # 1. 检索相关知识
        context = ""
        if self.use_rag:
            context = self.retrieve_knowledge(question)

        # 2. 构建 prompt
        prompt = self.format_prompt(question, options, context)

        # 3. 调用 Qwen3-8B
        try:
            response = self.client.chat.completions.create(
                model="qwen3-8b",
                messages=[
                    {
                        'role': 'system',
                        'content': self.system_prompt if context else self.system_prompt_no_rag
                    },
                    {'role': 'user', 'content': prompt}
                ],
                extra_body={"enable_thinking": False}
            )

            # 4. 提取答案
            answer_text = response.choices[0].message.content
            answer = self.extract_answer(answer_text)

            return f'[{answer}]', context

        except Exception as e:
            print(f"❌ API 调用失败: {e}")
            return '[A]', ""

    def batch_inference(self, questions, output_file):
        """
        批量推理

        Args:
            questions: 题目列表
            output_file: 输出文件路径
        """
        import json

        print(f"\n🔄 开始批量推理，共 {len(questions)} 道题...")
        results = []

        for idx, q in enumerate(questions):
            print(f"处理题目 {idx + 1}/{len(questions)}")

            prediction, context = self.answer_question(
                q.get('question', ''),
                q.get('options', {})
            )

            results.append({
                'index': idx,
                'input': q.get('question', ''),
                'prediction': prediction,
                'context_used': context[:200] if context else ""  # 保存前200字符
            })

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        print(f"\n✅ 推理完成！结果已保存到: {output_file}")


if __name__ == '__main__':
    # 测试代码
    API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8790af9e6e8b42ac9621dd9578741970")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 初始化引擎
    engine = RAGMaritimeQA(
        api_key=API_KEY,
        base_url=BASE_URL,
        vectorstore_path="./vectorstore"
    )

    # 测试问答
    test_question = "船舶电力系统供电网络中，放射形网络的特点是什么？"
    test_options = {
        'A': '②③',
        'B': '①③',
        'C': '②④',
        'D': '①④'
    }

    print("\n" + "=" * 50)
    print("🧪 测试问答")
    print("=" * 50)
    answer, context = engine.answer_question(test_question, test_options)
    print(f"题目: {test_question}")
    print(f"答案: {answer}")
    if context:
        print(f"参考资料: {context[:200]}...")