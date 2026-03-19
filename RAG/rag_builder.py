# -*- coding: utf-8 -*-
"""
RAG 知识库构建脚本
用于将航运知识文档转换为向量数据库
支持 PDF、TXT、JSON 格式
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


def load_json_documents(file_path):
    """
    加载 JSON 格式的知识文档

    支持的格式：
    1. 列表格式：[{"question": "...", "answer": "...", ...}, ...]
    2. 字典格式：{"key1": "value1", "key2": "value2", ...}
    """
    documents = []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        # 列表格式：每个元素作为一个文档
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # 将字典转换为文本
                if '题目' in item and '答案' in item:
                    # 新格式：题目、选项、答案
                    options = item.get('选项', {})
                    correct = item.get('答案', '')
                    correct_text = options.get(correct, '')
                    text = f"题目：{item['题目']}\n"
                    for k, v in options.items():
                        text += f"{k}. {v}\n"
                    text += f"正确答案：{correct}. {correct_text}"
                    if '类型' in item:
                        text = f"类型：{item['类型']}\n" + text
                elif 'question' in item and 'answer' in item:
                    # 问答格式
                    text = f"问题：{item['question']}\n答案：{item['answer']}"
                    if 'A' in item:
                        text += f"\n选项A：{item.get('A', '')}"
                        text += f"\n选项B：{item.get('B', '')}"
                        text += f"\n选项C：{item.get('C', '')}"
                        text += f"\n选项D：{item.get('D', '')}"
                else:
                    # 通用字典格式
                    text = "\n".join([f"{k}：{v}" for k, v in item.items()])

                doc = Document(
                    page_content=text,
                    metadata={"source": file_path, "index": i}
                )
                documents.append(doc)
            elif isinstance(item, str):
                # 纯文本列表
                doc = Document(
                    page_content=item,
                    metadata={"source": file_path, "index": i}
                )
                documents.append(doc)

    elif isinstance(data, dict):
        # 字典格式：每个键值对作为一个文档
        for i, (key, value) in enumerate(data.items()):
            text = f"{key}：{value}"
            doc = Document(
                page_content=text,
                metadata={"source": file_path, "key": key}
            )
            documents.append(doc)

    return documents


def build_knowledge_base(docs_folder, output_path):
    """构建航运知识向量库"""
    print(f"📚 开始构建知识库...")
    print(f"📂 文档目录: {docs_folder}")

    # 1. 加载文档
    documents = []
    if not os.path.exists(docs_folder):
        print(f"❌ 文档目录不存在: {docs_folder}")
        return None

    files = os.listdir(docs_folder)
    if not files:
        print(f"⚠️  文档目录为空，请添加 PDF 或 TXT 文件")
        return None

    print(f"📄 找到 {len(files)} 个文件")

    for file in files:
        file_path = os.path.join(docs_folder, file)
        try:
            if file.endswith('.pdf'):
                print(f"  - 加载 PDF: {file}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                print(f"  - 加载 TXT: {file}")
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            elif file.endswith('.docx'):
                print(f"  - 加载 DOCX: {file}")
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.json'):
                print(f"  - 加载 JSON: {file}")
                json_docs = load_json_documents(file_path)
                documents.extend(json_docs)
                print(f"    ✅ 加载了 {len(json_docs)} 条记录")
        except Exception as e:
            print(f"  ⚠️  加载失败 {file}: {e}")

    if not documents:
        print("❌ 没有成功加载任何文档")
        return None

    print(f"✅ 成功加载 {len(documents)} 个文档片段")

    # 2. 文本分块
    print("✂️  正在分块文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 分块完成，共 {len(chunks)} 个文本块")

    # 3. 生成向量并保存
    print("🔢 正在生成向量（首次运行会下载模型，请耐心等待）...")
    embeddings = HuggingFaceEmbeddings(
        model_name='BAAI/bge-small-zh-v1.5',
        model_kwargs={'device': 'cpu'}
    )

    print("💾 正在构建 FAISS 向量库...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    vectorstore.save_local(output_path)

    print(f"✅ 知识库构建完成！")
    print(f"📍 保存位置: {output_path}")
    print(f"📊 向量数量: {vectorstore.index.ntotal}")

    return vectorstore


if __name__ == '__main__':
    # 配置路径
    docs_folder = '../knowledge_docs'
    output_path = './vectorstore'

    # 构建知识库
    vectorstore = build_knowledge_base(docs_folder, output_path)

    if vectorstore:
        # 测试检索
        print("\n🔍 测试检索功能...")
        test_query = "船舶电力系统"
        results = vectorstore.similarity_search(test_query, k=2)
        print(f"查询: {test_query}")
        print(f"结果数量: {len(results)}")
        for i, doc in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(doc.page_content[:200] + "...")