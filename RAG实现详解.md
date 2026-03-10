# MaritimeBench RAG 实现详解

## 概述

本系统使用 RAG (Retrieval-Augmented Generation) 技术，结合向量检索和大语言模型，实现航运知识问答。

## 系统架构

```
知识文档 (PDF/TXT)
    ↓
文档加载与分块
    ↓
向量化 (Embedding)
    ↓
向量数据库 (FAISS)
    ↓
用户提问 → 向量检索 → 相关文档
    ↓
构建 Prompt (问题 + 检索到的文档)
    ↓
大语言模型 (Qwen3-8B)
    ↓
答案输出
```

---

## 第一阶段：知识库构建

### 文件：`RAG/rag_builder.py`

#### 1. 文档加载
**函数：** `build_knowledge_base()` (第13-79行)

**步骤：**
- 扫描 `knowledge_docs/` 目录下的所有文件
- 使用 `PyPDFLoader` 加载 PDF 文件 (第36行)
- 使用 `TextLoader` 加载 TXT 文件 (第40行)
- 返回文档列表

**关键代码：**
```python
# 第34-37行：加载PDF
if file.endswith('.pdf'):
    loader = PyPDFLoader(file_path)
    documents.extend(loader.load())

# 第38-41行：加载TXT
elif file.endswith('.txt'):
    loader = TextLoader(file_path, encoding='utf-8')
    documents.extend(loader.load())
```

#### 2. 文本分块
**函数：** `build_knowledge_base()` 中的分块部分 (第51-59行)

**步骤：**
- 使用 `RecursiveCharacterTextSplitter` 将长文档切分成小块
- 每块大小：500字符
- 块之间重叠：50字符（保证上下文连贯）
- 分隔符优先级：段落 → 句子 → 标点 → 空格

**关键代码：**
```python
# 第53-57行：配置分块器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块500字符
    chunk_overlap=50,      # 重叠50字符
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
)
chunks = text_splitter.split_documents(documents)
```

#### 3. 向量化与存储
**函数：** `build_knowledge_base()` 中的向量化部分 (第61-73行)

**步骤：**
- 使用 `BAAI/bge-small-zh-v1.5` 中文向量模型
- 将每个文本块转换为768维向量
- 使用 FAISS 构建向量索引
- 保存到 `RAG/vectorstore/` 目录

**关键代码：**
```python
# 第63-66行：初始化向量模型
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-zh-v1.5',
    model_kwargs={'device': 'cpu'}
)

# 第69行：构建FAISS向量库
vectorstore = FAISS.from_documents(chunks, embeddings)

# 第73行：保存到本地
vectorstore.save_local(output_path)
```

---

## 第二阶段：问答推理

### 文件：`RAG/rag_engine.py`

#### 1. 初始化 RAG 引擎
**类：** `RAGMaritimeQA`
**函数：** `__init__()` (第16-49行)

**步骤：**
- 初始化 OpenAI 客户端（连接阿里云 DashScope API）
- 加载已构建的向量数据库
- 设置系统提示词

**关键代码：**
```python
# 第28行：初始化API客户端
self.client = OpenAI(api_key=api_key, base_url=base_url)

# 第34-42行：加载向量库
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-zh-v1.5',
    model_kwargs={'device': 'cpu'}
)
self.vectorstore = FAISS.load_local(
    vectorstore_path,
    embeddings,
    allow_dangerous_deserialization=True
)
```

#### 2. 向量检索
**函数：** `retrieve_knowledge()` (第66-97行)

**步骤：**
- 将用户问题转换为向量
- 在向量库中搜索最相似的 k 个文档（k=5）
- 使用 L2 距离计算相似度（分数越低越相关）
- 过滤低相关性文档（分数 < 1.5）
- 拼接多个文档内容

**关键代码：**
```python
# 第81行：相似度搜索（返回文档和分数）
docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)

# 第85行：过滤低相关性文档
filtered_docs = [(doc, score) for doc, score in docs_with_scores if score < 1.5]

# 第96行：拼接文档内容
context = "\n\n---\n\n".join(context_parts)
```

#### 3. 构建 Prompt
**函数：** `format_prompt()` (第99-120行)

**步骤：**
- 将检索到的参考资料放在前面
- 添加题目和选项
- 构建完整的提示词

**关键代码：**
```python
# 第97-109行：带参考资料的prompt
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
```

#### 4. 调用大模型
**函数：** `answer_question()` (第144-188行)

**步骤：**
- 检索相关知识（调用 `retrieve_knowledge()`）
- 构建 prompt（调用 `format_prompt()`）
- 调用 Qwen3-8B 模型生成答案
- 提取答案字母（调用 `extract_answer()`）

**关键代码：**
```python
# 第161行：检索知识
context = self.retrieve_knowledge(question)

# 第164行：构建prompt
prompt = self.format_prompt(question, options, context)

# 第168-178行：调用模型
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

# 第181-182行：提取答案
answer_text = response.choices[0].message.content
answer = self.extract_answer(answer_text)
```

#### 5. 答案提取
**函数：** `extract_answer()` (第122-142行)

**步骤：**
- 使用正则表达式匹配 `[A]` `[B]` `[C]` `[D]` 格式
- 如果没有方括号，尝试匹配单独的字母
- 默认返回 A

**关键代码：**
```python
# 第132-134行：匹配方括号格式
match = re.search(r'\[([A-D])\]', response)
if match:
    return match.group(1)

# 第136-139行：匹配单独字母
match = re.search(r'\b([A-D])\b', response)
if match:
    return match.group(1)
```

---

## 第三阶段：用户交互

### 文件：`answer.py`

#### 1. 解析用户输入
**函数：** `parse_question_with_options()` (第51-110行)

**步骤：**
- 识别题目和选项的分界（"选项："标记或 A. B. C. D. 开头）
- 分离题目文本和选项文本
- 解析选项（支持 `A.` `A、` 等格式）
- 返回题目和选项字典

**关键代码：**
```python
# 第65-72行：查找选项开始位置
for i, line in enumerate(lines):
    if re.match(r'^选项[：:]\s*$', line.strip()):
        option_start_idx = i + 1
        break
    elif re.match(r'^[A-D][.、\s]', line.strip()):
        option_start_idx = i
        break

# 第97-108行：解析选项
for line in option_lines:
    match = re.match(r'^([A-D])[.、\s]+(.+)$', line)
    if match:
        letter = match.group(1)
        content = match.group(2).strip()
        options[letter] = content
```

#### 2. 清理参考资料
**函数：** `clean_context()` (第16-48行)

**步骤：**
- 移除多余空行
- 移除行首数字标记（如 "1→"）
- 移除特殊字符（全角空格、不间断空格）
- 移除连续符号（!、#、$、%）
- 规范化空格

**关键代码：**
```python
# 第30行：移除多余空行
text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

# 第33行：移除行首标记
text = re.sub(r'^\s*\d+→\s*', '', text, flags=re.MULTILINE)

# 第36-37行：移除特殊空格
text = text.replace('\u3000', ' ')  # 全角空格
text = text.replace('\xa0', ' ')    # 不间断空格

# 第40-43行：移除连续符号
text = re.sub(r'!+', '', text)
text = re.sub(r'#+', '', text)
```

#### 3. 主交互循环
**函数：** `main()` (第113-163行)

**步骤：**
- 初始化 RAG 引擎
- 循环读取用户输入（多行，两次回车结束）
- 解析题目和选项
- 调用 RAG 引擎回答
- 显示答案和参考资料

**关键代码：**
```python
# 第127-131行：初始化引擎
engine = RAGMaritimeQA(
    api_key=API_KEY,
    base_url=BASE_URL,
    vectorstore_path=VECTORSTORE_PATH
)

# 第142-145行：解析输入
full_text = "\n".join(lines)
question, options = parse_question_with_options(full_text)

# 第148-149行：调用RAG回答
answer, context = engine.answer_question(question, options)

# 第153-163行：显示结果
print(f"✅ 答案: {answer}")
if context and engine.use_rag:
    cleaned_context = clean_context(context)
    print(cleaned_context[:500])
```

---

## 关键技术点

### 1. 向量检索原理
- **Embedding 模型：** BAAI/bge-small-zh-v1.5（中文优化）
- **向量维度：** 768维
- **相似度计算：** L2 欧氏距离（FAISS 默认）
- **检索策略：** Top-K 检索 + 分数过滤

### 2. 文本分块策略
- **块大小：** 500字符（平衡上下文和检索精度）
- **重叠：** 50字符（避免关键信息被截断）
- **分隔符：** 优先按段落、句子分割（保持语义完整）

### 3. Prompt 工程
- **系统提示词：** 明确角色（航运专家）和输出格式（[A]/[B]/[C]/[D]）
- **用户提示词：** 参考资料 + 题目 + 选项
- **答案提取：** 正则表达式匹配方括号格式

### 4. 相关性过滤
- **阈值：** L2 距离 < 1.5（经验值）
- **保底策略：** 至少保留最相关的1个文档
- **文档数量：** 检索5个，过滤后通常保留2-3个

---

## 使用流程

### 构建知识库
```bash
cd RAG
python rag_builder.py
```

### 运行问答系统
```bash
python answer.py
```

### 批量推理
```bash
python answer.py --batch
```

---

## 依赖库

- **langchain-community：** 文档加载、文本分块、向量存储
- **faiss-cpu：** 向量检索引擎
- **sentence-transformers：** Embedding 模型
- **openai：** API 客户端（兼容阿里云 DashScope）
- **pypdf：** PDF 解析

---

## 性能优化建议

1. **向量库优化：**
   - 使用 GPU 加速 Embedding（修改 `device='cuda'`）
   - 使用 FAISS 的 IVF 索引（适合大规模数据）

2. **检索优化：**
   - 调整 k 值（检索数量）
   - 调整相似度阈值
   - 使用混合检索（向量 + 关键词）

3. **Prompt 优化：**
   - 精简参考资料长度
   - 添加思维链提示（Chain-of-Thought）
   - 使用少样本学习（Few-shot）

---

## 常见问题

**Q: 为什么检索结果不准确？**
A: 可能原因：
- 知识库文档质量不高
- 文本分块不合理
- Embedding 模型不适合领域
- 相似度阈值设置不当

**Q: 如何添加新的知识文档？**
A:
1. 将 PDF/TXT 文件放入 `knowledge_docs/` 目录
2. 重新运行 `python RAG/rag_builder.py`

**Q: 如何切换其他大模型？**
A: 修改 `rag_engine.py` 第169行的 `model` 参数

---

## 总结

本系统通过以下步骤实现 RAG：

1. **离线构建：** 文档加载 → 分块 → 向量化 → 存储（`rag_builder.py`）
2. **在线检索：** 问题向量化 → 相似度搜索 → 返回相关文档（`rag_engine.py`）
3. **生成答案：** 构建 Prompt → 调用大模型 → 提取答案（`rag_engine.py`）
4. **用户交互：** 解析输入 → 调用 RAG → 显示结果（`answer.py`）

核心优势：
- ✅ 结合知识库和大模型，提高答案准确性
- ✅ 支持动态更新知识库
- ✅ 可追溯答案来源（显示参考资料）
