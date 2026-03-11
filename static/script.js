document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chatBox');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    const ragResultBox = document.getElementById('ragResultBox');
    const ragInput = document.getElementById('ragInput');
    const ragBtn = document.getElementById('ragBtn');

    // Tab 切换
    const tabBtns = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.panel');
    let currentTab = 'chat';

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            currentTab = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(currentTab + 'Panel').classList.add('active');
        });
    });

    // 清空按钮
    clearBtn.addEventListener('click', () => {
        if (currentTab === 'chat') clearChat();
        else clearRag();
    });

    // session ID
    let sessionId = localStorage.getItem('qwen_session_id');
    if (!sessionId) {
        sessionId = 'session_' + Math.random().toString(36).slice(2, 11);
        localStorage.setItem('qwen_session_id', sessionId);
    }

    // HTML 转义
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }

    function addMessage(box, content, role, isHtml = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        msgDiv.innerHTML = `
            <div class="avatar">${role === 'user' ? '👤' : '🤖'}</div>
            <div class="bubble">${isHtml ? content : escapeHtml(content)}</div>
        `;
        box.appendChild(msgDiv);
        box.scrollTop = box.scrollHeight;
    }

    function addTypingIndicator(box) {
        const id = 'typing-' + Date.now();
        const div = document.createElement('div');
        div.className = 'message assistant';
        div.id = id;
        div.innerHTML = `<div class="avatar">🤖</div><div class="bubble"><span class="typing">正在思考</span></div>`;
        box.appendChild(div);
        box.scrollTop = box.scrollHeight;
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    // ── 聊天模式 ──
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        addMessage(chatBox, message, 'user');
        userInput.value = '';
        sendBtn.disabled = true;

        const typingId = addTypingIndicator(chatBox);

        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, session_id: sessionId })
            });
            const data = await res.json();
            removeTypingIndicator(typingId);
            addMessage(chatBox, data.error ? `❌ 错误: ${data.error}` : data.reply, 'assistant');
        } catch (e) {
            removeTypingIndicator(typingId);
            addMessage(chatBox, `❌ 网络错误: ${e.message}`, 'assistant');
        } finally {
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    async function clearChat() {
        if (!confirm('确定要清空当前对话吗？')) return;
        try {
            await fetch('/clear', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            });
            chatBox.innerHTML = `
                <div class="message assistant">
                    <div class="avatar">🤖</div>
                    <div class="bubble">对话已清空，有什么可以帮你的吗？</div>
                </div>`;
        } catch (e) {
            alert('清空失败: ' + e.message);
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    // ── RAG 答题模式 ──
    async function submitRag() {
        const text = ragInput.value.trim();
        if (!text) return;

        addMessage(ragResultBox, text, 'user');
        ragInput.value = '';
        ragBtn.disabled = true;

        const typingId = addTypingIndicator(ragResultBox);

        try {
            const res = await fetch('/rag', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await res.json();
            removeTypingIndicator(typingId);

            if (data.error) {
                addMessage(ragResultBox, `❌ 错误: ${data.error}`, 'assistant');
            } else {
                let html = `<div class="rag-answer">✅ 答案：<strong>${escapeHtml(data.answer)}</strong></div>`;
                if (data.context) {
                    html += `<div class="rag-context"><div class="rag-context-title">📚 参考资料</div><div class="rag-context-body">${escapeHtml(data.context)}`;
                    if (data.has_more) html += `<span class="rag-more">…（更多内容已省略）</span>`;
                    html += `</div></div>`;
                }
                addMessage(ragResultBox, html, 'assistant', true);
            }
        } catch (e) {
            removeTypingIndicator(typingId);
            addMessage(ragResultBox, `❌ 网络错误: ${e.message}`, 'assistant');
        } finally {
            ragBtn.disabled = false;
            ragInput.focus();
        }
    }

    function clearRag() {
        ragResultBox.innerHTML = `
            <div class="message assistant">
                <div class="avatar">🔍</div>
                <div class="bubble">请在下方输入完整题目（包括选项），系统将通过 RAG 检索知识库后作答。</div>
            </div>`;
    }

    ragBtn.addEventListener('click', submitRag);
    ragInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitRag(); }
    });

    // ── 批量推理模式 ──
    const batchInput = document.getElementById('batchInput');
    const batchOutput = document.getElementById('batchOutput');
    const batchStartBtn = document.getElementById('batchStartBtn');
    const batchProgress = document.getElementById('batchProgress');
    const progressText = document.getElementById('progressText');
    const progressAcc = document.getElementById('progressAcc');
    const progressBar = document.getElementById('progressBar');
    const batchLog = document.getElementById('batchLog');
    const batchDownloadBtn = document.getElementById('batchDownloadBtn');

    let currentJobId = null;
    let pollTimer = null;

    async function startBatch() {
        const jsonl = batchInput.value.trim();
        if (!jsonl) return;

        batchStartBtn.disabled = true;
        batchProgress.style.display = 'block';
        batchLog.innerHTML = '';
        batchDownloadBtn.style.display = 'none';
        progressBar.style.width = '0%';
        progressText.textContent = '0 / ?';
        progressAcc.textContent = '';

        try {
            const res = await fetch('/batch/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ jsonl, output: batchOutput.value.trim() || 'predictions.jsonl' })
            });
            const data = await res.json();
            if (data.error) { alert('错误: ' + data.error); batchStartBtn.disabled = false; return; }
            currentJobId = data.job_id;
            pollTimer = setInterval(pollBatch, 1000);
        } catch (e) {
            alert('网络错误: ' + e.message);
            batchStartBtn.disabled = false;
        }
    }

    async function pollBatch() {
        if (!currentJobId) return;
        try {
            const res = await fetch(`/batch/status/${currentJobId}`);
            const job = await res.json();

            const pct = job.total > 0 ? (job.done / job.total * 100).toFixed(0) : 0;
            progressBar.style.width = pct + '%';
            progressText.textContent = `${job.done} / ${job.total}`;

            if (job.done > 0 && job.total > 0) {
                const acc = (job.correct / job.done * 100).toFixed(1);
                progressAcc.textContent = `准确率: ${job.correct}/${job.done} = ${acc}%`;
            }

            // 追加新日志行
            const rendered = batchLog.querySelectorAll('.log-row').length;
            const newRows = job.results.slice(rendered);
            newRows.forEach(r => {
                const icon = r.correct ? '✅' : (r.target ? '❌' : '➡️');
                const row = document.createElement('div');
                row.className = 'log-row';
                row.textContent = `${icon} #${r.index}  ${r.prediction}  (target: ${r.target || '?'})`;
                batchLog.appendChild(row);
            });
            batchLog.scrollTop = batchLog.scrollHeight;

            if (job.status === 'done') {
                clearInterval(pollTimer);
                batchStartBtn.disabled = false;
                batchDownloadBtn.style.display = 'inline-block';
            } else if (job.status === 'error') {
                clearInterval(pollTimer);
                batchStartBtn.disabled = false;
                const errRow = document.createElement('div');
                errRow.className = 'log-row error';
                errRow.textContent = '❌ 任务失败: ' + job.error;
                batchLog.appendChild(errRow);
            }
        } catch (e) { /* ignore poll errors */ }
    }

    batchStartBtn.addEventListener('click', startBatch);
    batchDownloadBtn.addEventListener('click', () => {
        if (currentJobId) window.location.href = `/batch/download/${currentJobId}`;
    });

    userInput.focus();
});
