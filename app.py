# -*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from openai import OpenAI

app = Flask(__name__)

# 🔐 API 配置（建议打包时通过环境变量注入）
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-8790af9e6e8b42ac9621dd9578741970")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
chat_histories = {}

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

# 🎬 主入口
if __name__ == '__main__':
    print("=" * 50)
    print("🚀 MaritimeBench RAG 系统启动中...")
    print("=" * 50)
    print("📍 访问地址: http://127.0.0.1:5000")
    print("💡 提示: 在浏览器中打开上述地址")
    print("=" * 50)
    app.run(host='127.0.0.1', port=5000, debug=True)