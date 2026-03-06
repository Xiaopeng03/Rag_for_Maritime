document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chatBox');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    // 生成唯一 session ID（简单实现，刷新会重置）
    let sessionId = localStorage.getItem('qwen_session_id');
    if (!sessionId) {
        sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('qwen_session_id', sessionId);
    }

    // 发送消息
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        
        // 添加用户消息到界面
        addMessage(message, 'user');
        userInput.value = '';
        sendBtn.disabled = true;
        
        // 显示打字指示器
        const typingId = addTypingIndicator();
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, session_id: sessionId })
            });
            
            const data = await response.json();
            
            // 移除打字指示器
            removeTypingIndicator(typingId);
            
            if (data.error) {
                addMessage(`❌ 错误: ${data.error}`, 'assistant');
            } else {
                addMessage(data.reply, 'assistant');
            }
        } catch (error) {
            removeTypingIndicator(typingId);
            addMessage(`❌ 网络错误: ${error.message}`, 'assistant');
        } finally {
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    // 添加消息到聊天框
    function addMessage(content, role) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        msgDiv.innerHTML = `
            <div class="avatar">${role === 'user' ? '👤' : '🤖'}</div>
            <div class="bubble">${escapeHtml(content)}</div>
        `;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // 添加打字指示器
    function addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.id = id;
        typingDiv.innerHTML = `
            <div class="avatar">🤖</div>
            <div class="bubble"><span class="typing">正在思考</span></div>
        `;
        chatBox.appendChild(typingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return id;
    }

    // 移除打字指示器
    function removeTypingIndicator(id) {
        const element = document.getElementById(id);
        if (element) element.remove();
    }

    // HTML 转义，防止 XSS
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }

    // 清空对话
    async function clearChat() {
        if (!confirm('确定要清空当前对话吗？')) return;
        
        try {
            await fetch('/clear', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });
            // 清空界面，只保留欢迎语
            chatBox.innerHTML = `
                <div class="message assistant">
                    <div class="avatar">🤖</div>
                    <div class="bubble">对话已清空，有什么可以帮你的吗？</div>
                </div>
            `;
        } catch (error) {
            alert('清空失败: ' + error.message);
        }
    }

    // 事件绑定
    sendBtn.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    clearBtn.addEventListener('click', clearChat);
    
    // 自动聚焦输入框
    userInput.focus();
});