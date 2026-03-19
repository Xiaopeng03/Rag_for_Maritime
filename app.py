# -*- coding: utf-8 -*-
import os
import sys
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'RAG'))

import json
import threading
from flask import Flask, request, jsonify, render_template, send_from_directory
from openai import OpenAI
from RAG.rag_engine import RAGMaritimeQA
from qwen_8b_engine import Qwen8BEngine, parse_input_field, extract_answer, build_sample_score

app = Flask(__name__)

# 🔐 API 配置（建议打包时通过环境变量注入）
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8d06a454088f42569c26079b97421737")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VECTORSTORE_PATH = "./RAG/vectorstore"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
chat_histories = {}

# 批量任务状态存储
batch_jobs = {}   # job_id -> {status, total, done, results, error, output_file}
_batch_engine = None

def get_batch_engine():
    global _batch_engine
    if _batch_engine is None:
        _batch_engine = Qwen8BEngine(use_rag=True)
    return _batch_engine

# 初始化 RAG 引擎（懒加载）
_rag_engine = None

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGMaritimeQA(
            api_key=API_KEY,
            base_url=BASE_URL,
            vectorstore_path=VECTORSTORE_PATH
        )
    return _rag_engine


def parse_question_with_options(text):
    """解析题目和选项"""
    lines = text.strip().split('\n')
    option_start_idx = -1
    for i, line in enumerate(lines):
        if re.match(r'^选项[：:]\s*$', line.strip()):
            option_start_idx = i + 1
            break
        elif re.match(r'^[A-D][.、\s]', line.strip()):
            option_start_idx = i
            break

    if option_start_idx > 0:
        question_lines = lines[:option_start_idx]
        option_lines = lines[option_start_idx:]
    else:
        question_lines, option_lines = [], []
        for i, line in enumerate(lines):
            if re.match(r'^[A-D][.、\s]', line.strip()):
                question_lines = lines[:i]
                option_lines = lines[i:]
                break
        if not option_lines:
            return text.strip(), {}

    question = '\n'.join(question_lines).strip()
    question = re.sub(r'选项[：:]\s*$', '', question).strip()

    options = {}
    for line in option_lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^([A-D])[.、\s]+(.+)$', line)
        if match:
            options[match.group(1)] = match.group(2).strip()

    return question, options

# 🌐 Flask 路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id', 'default')
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': '消息不能为空'}), 400
    
    if session_id not in chat_histories:
        chat_histories[session_id] = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}
        ]
    
    chat_histories[session_id].append({'role': 'user', 'content': user_message})
    
    try:
        completion = client.chat.completions.create(
            model="qwen3-8b",
            messages=chat_histories[session_id],
            extra_body={"enable_thinking": False}
        )
        
        assistant_reply = completion.choices[0].message.content
        chat_histories[session_id].append({'role': 'assistant', 'content': assistant_reply})
        
        # 限制历史长度
        if len(chat_histories[session_id]) > 22:
            chat_histories[session_id] = [chat_histories[session_id][0]] + chat_histories[session_id][-20:]
        
        return jsonify({'reply': assistant_reply, 'session_id': session_id})
        
    except Exception as e:
        print(f"❌ API 错误: {e}")
        return jsonify({'error': f'请求失败: {str(e)}'}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    session_id = request.json.get('session_id', 'default')
    if session_id in chat_histories:
        chat_histories[session_id] = [chat_histories[session_id][0]]
    return jsonify({'success': True})


@app.route('/rag', methods=['POST'])
def rag_answer():
    data = request.json
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': '题目不能为空'}), 400

    question, options = parse_question_with_options(text)
    if not question:
        return jsonify({'error': '无法解析题目'}), 400

    try:
        engine = get_rag_engine()
        answer, context = engine.answer_question(question, options)

        # 清理并截取参考资料
        if context:
            context = re.sub(r'\n\s*\n\s*\n+', '\n\n', context)
            context = context.replace('\u3000', ' ').replace('\xa0', ' ')
            context = re.sub(r' +', ' ', context).strip()

        return jsonify({
            'answer': answer,
            'context': context[:1000] if context else '',
            'has_more': len(context) > 1000 if context else False
        })
    except Exception as e:
        print(f"❌ RAG 错误: {e}")
        return jsonify({'error': f'请求失败: {str(e)}'}), 500


@app.route('/batch/start', methods=['POST'])
def batch_start():
    data = request.json
    jsonl_text = data.get('jsonl', '').strip()
    output_name = data.get('output', 'predictions.jsonl').strip()

    if not jsonl_text:
        return jsonify({'error': '内容不能为空'}), 400

    records = []
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            return jsonify({'error': f'JSON 解析失败: {e}'}), 400

    if not records:
        return jsonify({'error': '没有有效的题目'}), 400

    import uuid
    job_id = str(uuid.uuid4())[:8]
    output_file = output_name if output_name.endswith('.jsonl') else output_name + '.jsonl'

    batch_jobs[job_id] = {
        'status': 'running', 'total': len(records),
        'done': 0, 'correct': 0, 'results': [],
        'error': None, 'output_file': output_file
    }

    def run():
        try:
            engine = get_batch_engine()
            out_path = os.path.join(os.path.dirname(__file__), output_file)
            with open(out_path, 'w', encoding='utf-8') as f:
                for i, rec in enumerate(records):
                    idx = rec.get('index', i)
                    input_text = rec.get('input', '')
                    target = rec.get('target', '')
                    question, options = parse_input_field(input_text)
                    try:
                        prediction = engine.answer_one(question, options)
                    except Exception:
                        prediction = '[A]'
                    result = {
                        'index': idx, 'input': input_text, 'target': target,
                        'sample_score': build_sample_score(prediction, target, idx)
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()
                    extracted = extract_answer(prediction)
                    is_correct = (extracted == target) if target else False
                    batch_jobs[job_id]['done'] = i + 1
                    if is_correct:
                        batch_jobs[job_id]['correct'] += 1
                    batch_jobs[job_id]['results'].append({
                        'index': idx, 'prediction': prediction,
                        'target': target, 'correct': is_correct
                    })
            batch_jobs[job_id]['status'] = 'done'
        except Exception as e:
            batch_jobs[job_id]['status'] = 'error'
            batch_jobs[job_id]['error'] = str(e)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({'job_id': job_id, 'total': len(records)})


@app.route('/batch/status/<job_id>', methods=['GET'])
def batch_status(job_id):
    job = batch_jobs.get(job_id)
    if not job:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(job)


@app.route('/batch/download/<job_id>', methods=['GET'])
def batch_download(job_id):
    from flask import send_file
    job = batch_jobs.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': '任务未完成'}), 400
    path = os.path.join(os.path.dirname(__file__), job['output_file'])
    return send_file(path, as_attachment=True, download_name=job['output_file'])


# 🎬 主入口
if __name__ == '__main__':
    print("=" * 50)
    print("🚀 MaritimeBench RAG 系统启动中...")
    print("=" * 50)
    print("📍 访问地址: http://127.0.0.1:5000")
    print("💡 提示: 在浏览器中打开上述地址")
    print("=" * 50)
    app.run(host='127.0.0.1', port=5000, debug=True)