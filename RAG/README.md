# RAG 知识增强模块

这个文件夹包含了 RAG（检索增强生成）相关的代码。

## 文件说明

- `rag_builder.py` - 知识库构建脚本，将文档转换为向量数据库
- `rag_engine.py` - RAG 推理引擎，结合检索和 Qwen3-8B 进行问答
- `requirements.txt` - 依赖包列表
- `vectorstore/` - 向量数据库存储目录（运行后自动生成）

## 使用步骤

### 1. 安装依赖
```bash
cd RAG
pip install -r requirements.txt
```

### 2. 准备知识文档
在项目根目录创建 `knowledge_docs` 文件夹，放入航运相关的 PDF 或 TXT 文档：
```bash
mkdir ../knowledge_docs
# 将你的航运知识文档（PDF/TXT）放入该文件夹
```

### 3. 构建知识库
```bash
python rag_builder.py
```

### 4. 测试 RAG 引擎
```bash
python rag_engine.py
```

## 注意事项

- 首次运行会自动下载 BGE 嵌入模型（约 400MB）
- 知识库构建时间取决于文档数量和大小
- 如果没有知识库，系统会自动切换到无 RAG 模式