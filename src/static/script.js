const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sidebar = document.getElementById("sidebar");
const overlay = document.getElementById("mobile-overlay");
const sendBtn = document.getElementById("send-btn");

let attachedFile = null; 
const clientHistory = [];
const MAX_CLIENT_HISTORY = 200; 

// --- Configure Marked.js for Syntax Highlighting ---
if (typeof marked !== 'undefined') {
    marked.setOptions({
        gfm: true,
        breaks: true,
        highlight: function(code, lang) {
            if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return code;
        }
    });
}

function pushClientHistory(role, text) {
    clientHistory.push({ role, text: text.replace(/[\*#`_~>]/g, '').slice(0, 400) });
    if (clientHistory.length > MAX_CLIENT_HISTORY) {
        clientHistory.splice(0, 20); 
    }
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

function getFileIcon(name) {
    const ext = name.split('.').pop().toLowerCase();
    const icons = {
        py: '#3b82f6', js: '#f59e0b', ts: '#3b82f6', html: '#f97316',
        css: '#8b5cf6', json: '#10b981', pdf: '#ef4444', md: '#6b7280',
        csv: '#10b981', txt: '#9ca3af', jpg: '#ec4899', jpeg: '#ec4899',
        png: '#ec4899', gif: '#ec4899', webp: '#ec4899'
    };
    const color = icons[ext] || '#6b7280';
    const isImage = ['jpg','jpeg','png','gif','webp','bmp','svg'].includes(ext);
    if (isImage) {
        return `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`;
    }
    return `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><text x="6" y="18" font-size="5" fill="${color}" stroke="none">${ext.toUpperCase()}</text></svg>`;
}

function showFilePreview(name, size, file) {
    removeFilePreview();
    const bar = document.querySelector('.modern-search-bar');
    if (!bar) return;
    const ext = name.split('.').pop().toLowerCase();
    const isImage = ['jpg','jpeg','png','gif','webp','bmp'].includes(ext);
    let thumbHtml = '';
    if (isImage && file) {
        const url = URL.createObjectURL(file);
        thumbHtml = `<img src="${url}" class="file-thumb" alt="preview" onload="URL.revokeObjectURL(this.src)">`;
    }
    const preview = document.createElement('div');
    preview.id = 'file-preview';
    preview.innerHTML = `
        ${thumbHtml}
        <div class="file-icon">${getFileIcon(name)}</div>
        <div class="file-meta">
            <span class="file-name">${name}</span>
            <span class="file-size">${formatBytes(size)}</span>
        </div>
        <span class="file-tag">${ext.toUpperCase()}</span>
        <button class="file-remove-btn" onclick="removeFilePreview()" aria-label="Remove file">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
    `;
    bar.parentNode.insertBefore(preview, bar);
}

function removeFilePreview() {
    const prev = document.getElementById('file-preview');
    if (prev) prev.remove();
    attachedFile = null;
}

function handleFileSelect(file) {
    if (!file) return;
    const MAX_SIZE = 20 * 1024 * 1024; 
    if (file.size > MAX_SIZE) {
        alert('File too large. Maximum size is 20 MB.');
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        const base64 = e.target.result.split(',')[1];
        attachedFile = { name: file.name, type: file.type, data_b64: base64, size: file.size };
        showFilePreview(file.name, file.size, file);
        sendBtn.classList.add('active');
        const input = document.getElementById('user-input');
        if (input && !input.value) {
            input.placeholder = 'Ask a question about this file, or press send to summarise...';
        }
    };
    reader.readAsDataURL(file);
}

// --- Auto-resize Textarea & Dynamic Send Button ---
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    
    if(this.value.trim().length > 0) {
        sendBtn.classList.add('active');
    } else {
        sendBtn.classList.remove('active');
        this.style.height = 'auto'; 
    }
});

userInput.addEventListener('keydown', function(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        if(this.value.trim().length > 0 || attachedFile) {
            sendMessage();
        }
    }
});

// --- Layout Toggles ---
function toggleSidebar() {
    sidebar.classList.toggle("collapsed");
    if(window.innerWidth <= 768) {
        overlay.classList.toggle("active");
    }
}

if (window.innerWidth <= 768) sidebar.classList.add("collapsed");

window.addEventListener('resize', () => {
    if (window.innerWidth > 768) overlay.classList.remove("active");
});

function updateModePill(value) {
    const badge = document.getElementById('mode-display');
    if (badge) badge.textContent = value.charAt(0).toUpperCase() + value.slice(1);
}

function createFog() {
    const wrapper = document.getElementById('fog-wrapper');
    if (!wrapper) return;
    for (let i = 0; i < 12; i++) {
        let cloud = document.createElement('div');
        cloud.className = 'fog-cloud';
        cloud.style.top = (Math.random() * 80) + '%';
        cloud.style.width = (Math.random() * 500 + 300) + 'px';
        cloud.style.height = (Math.random() * 200 + 100) + 'px';
        cloud.style.animationDuration = (Math.random() * 20 + 15) + 's';
        cloud.style.animationDelay = '-' + (Math.random() * 20) + 's';
        wrapper.appendChild(cloud);
    }
}

async function fetchDynamicQuote() {
    try {
        const response = await fetch("/quote");
        if (!response.ok) throw new Error("Backend quote endpoint not ready");
        const data = await response.json();
        return {
            text: data.text || "We did not build minds of silicon to answer everything, but to give our wildest questions a place to echo.",
            author: data.author || "Hema AI"
        };
    } catch (error) {
        const fallbacks = [
            "We did not build minds of silicon to answer everything, but to give our wildest questions a place to echo.",
            "The true sign of intelligence is not knowledge but imagination.",
            "Any sufficiently advanced technology is indistinguishable from magic.",
            "Intelligence is the ability to adapt to change.",
            "Somewhere, something incredible is waiting to be known.",
        ];
        return { text: fallbacks[Math.floor(Math.random() * fallbacks.length)], author: "" };
    }
}

async function initWelcomeScreen() {
    const savedName = localStorage.getItem("hemaUserName");
    const titleEl = document.getElementById("welcome-title");
    const nameInputEl = document.getElementById("name-input");
    const quoteEl = document.getElementById('quote-container');
    
    if (savedName && titleEl && nameInputEl) {
        titleEl.innerHTML = `Hello, <span>${savedName}</span>`;
        nameInputEl.style.display = "none";
        const sidebarName = document.getElementById("sidebar-name");
        const sidebarAvatar = document.getElementById("sidebar-avatar");
        if (sidebarName) sidebarName.textContent = savedName;
        if (sidebarAvatar) sidebarAvatar.textContent = savedName.charAt(0).toUpperCase();
    }
    
    if (quoteEl) {
        quoteEl.style.opacity = 0; 
        const quoteData = await fetchDynamicQuote();
        quoteEl.innerHTML = quoteData.text ? `"${quoteData.text}"` : '';
        quoteEl.style.opacity = 1; 
    }
}

function saveName(event) {
    if (event.key === "Enter") {
        const name = event.target.value.trim();
        if (name) {
            localStorage.setItem("hemaUserName", name);
            const titleEl = document.getElementById("welcome-title");
            titleEl.innerHTML = `Hello, <span>${name}</span>`;
            event.target.style.display = "none";
            const sidebarName = document.getElementById("sidebar-name");
            const sidebarAvatar = document.getElementById("sidebar-avatar");
            if (sidebarName) sidebarName.textContent = name;
            if (sidebarAvatar) sidebarAvatar.textContent = name.charAt(0).toUpperCase();
        }
    }
}

function editName() {
    const titleEl = document.getElementById("welcome-title");
    const nameInputEl = document.getElementById("name-input");
    if (titleEl && nameInputEl) {
        titleEl.innerHTML = `Hello, I am <span>Hema</span>`;
        nameInputEl.style.display = "inline-block";
        nameInputEl.value = localStorage.getItem("hemaUserName") || "";
        nameInputEl.focus();
    }
}

document.addEventListener("DOMContentLoaded", () => {
    createFog();
    initWelcomeScreen();
    initAttachButton();
    setupScrollFAB();
});

function initAttachButton() {
    const attachBtn = document.querySelector('.search-icon-btn');
    if (!attachBtn) return;

    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.id = 'hidden-file-input';
    fileInput.accept = '.txt,.py,.js,.ts,.html,.css,.json,.csv,.md,.xml,.yaml,.yml,.pdf,.jpg,.jpeg,.png,.gif,.java,.cpp,.c,.h,.rs,.go,.sh';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
        fileInput.value = ''; 
    });

    attachBtn.addEventListener('click', () => fileInput.click());

    const bar = document.querySelector('.modern-search-bar');
    if (bar) {
        bar.addEventListener('dragover', (e) => { e.preventDefault(); bar.style.borderColor = 'rgba(255,255,255,0.3)'; });
        bar.addEventListener('dragleave', () => { bar.style.borderColor = ''; });
        bar.addEventListener('drop', (e) => {
            e.preventDefault();
            bar.style.borderColor = '';
            if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
        });
    }
}

function removeWelcomeScreen() {
    const welcomeScreen = document.getElementById("welcome-screen");
    if (welcomeScreen) welcomeScreen.remove();
}

function formatAIResponse(text) {
    if (!text) return text;
    let formatted = text;
    const isHybridSearch = /🌐\s*<b>/i.test(formatted);
    if (isHybridSearch) {
        formatted = formatted.replace(/🌐\s*<b>(.*?)<\/b>(?:<br\s*\/?>\s*){0,2}/gi, '### 🌐 $1\n\n');
        formatted = formatted.replace(/<b>([^<]*?)(?::\s*)?<\/b>\s*(.*?)(?=(?:<br\s*\/?>\s*){2,}|<b>|$)/gis, 
            '> <span class="source-title">$1</span>\n> $2\n\n');
    }
    formatted = formatted.replace(/<br\s*\/?>/gi, '\n');
    return formatted;
}

function enhanceRenderedContent(el) {
    if (!el) return;

    el.querySelectorAll("pre:not(.enhanced)").forEach(pre => {
        pre.classList.add("enhanced");
        const codeEl = pre.querySelector("code");
        let lang = "";
        if (codeEl) {
            const cls = codeEl.className || "";
            const m = cls.match(/language-(\w+)/);
            lang = m ? m[1] : "code";
        }
        const wrapper = document.createElement("div");
        wrapper.className = "code-block-wrapper";
        const header = document.createElement("div");
        header.className = "code-block-header";
        header.innerHTML = `
            <span class="code-lang-label">${lang.toUpperCase()}</span>
            <button class="code-copy-btn" title="Copy code">Copy</button>`;
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(header);
        wrapper.appendChild(pre);
        
        header.querySelector(".code-copy-btn").onclick = function() {
            const code = codeEl ? codeEl.innerText : pre.innerText;
            navigator.clipboard.writeText(code).then(() => {
                this.textContent = "Copied!";
                setTimeout(() => { this.textContent = "Copy"; }, 2000);
            }).catch(() => {
                // Fallback
                const textArea = document.createElement("textarea");
                textArea.value = code; document.body.appendChild(textArea);
                textArea.select(); document.execCommand("copy"); document.body.removeChild(textArea);
                this.textContent = "Copied!"; setTimeout(() => { this.textContent = "Copy"; }, 2000);
            });
        };
    });

    el.querySelectorAll("table:not(.enhanced)").forEach(tbl => {
        tbl.classList.add("enhanced");
        const wrap = document.createElement("div");
        wrap.className = "table-wrap";
        tbl.parentNode.insertBefore(wrap, tbl);
        wrap.appendChild(tbl);
    });

    el.querySelectorAll("li:not(.enhanced)").forEach((li, i) => {
        li.classList.add("enhanced");
        li.style.animationDelay = Math.min(i * 0.04, 0.5) + "s";
    });

    el.querySelectorAll("h1:not(.enhanced),h2:not(.enhanced),h3:not(.enhanced)").forEach((h, i) => {
        h.classList.add("enhanced");
        h.style.animationDelay = i * 0.05 + "s";
    });
}

// ── Smart Auto-Scroll ──
function safeScrollToBottom() {
    const isUserScrollingUp = chatBox.scrollHeight - chatBox.scrollTop - chatBox.clientHeight > 200;
    if (!isUserScrollingUp) {
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

function setupScrollFAB() {
    let fab = document.getElementById("scroll-fab");
    if (!fab) {
        fab = document.createElement("button");
        fab.id = "scroll-fab";
        fab.className = "scroll-fab";
        fab.title = "Scroll to bottom";
        fab.setAttribute("aria-label", "Scroll to bottom");
        fab.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" width="16" height="16"><polyline points="6 9 12 15 18 9"/></svg>`;
        fab.style.display = "none";
        document.querySelector(".chat-container").appendChild(fab);
        fab.onclick = () => chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: "smooth" });
    }
    chatBox.addEventListener("scroll", () => {
        const isNearBottom = chatBox.scrollHeight - chatBox.scrollTop - chatBox.clientHeight < 120;
        fab.style.display = isNearBottom ? "none" : "flex";
    });
}

// ── OPTIMIZED TYPEWRITER EFFECT ──
function typeWriterEffect(contentElement, containerElement, text, actionsElement) {
    if (text.length > 600) {
        contentElement.style.opacity = "0";
        contentElement.innerHTML = DOMPurify.sanitize(marked.parse(text));
        enhanceRenderedContent(contentElement);
        contentElement.style.transition = "opacity 0.45s ease";
        requestAnimationFrame(() => { contentElement.style.opacity = "1"; });
        if (actionsElement) actionsElement.style.display = "flex";
        safeScrollToBottom();
        return;
    }

    const words = text.split(' ');
    let i = 0;
    contentElement.innerHTML = "";
    safeScrollToBottom();

    // CPU Optimization: Render multiple words per tick to avoid locking the main thread
    function typeChunk() {
        if (i < words.length) {
            // Grab up to 3 words at a time for smoother appearance
            const chunkEnd = Math.min(i + 3, words.length);
            const partial = words.slice(0, chunkEnd).join(' ') + (chunkEnd < words.length ? ' ' : '');
            i = chunkEnd;
            
            contentElement.innerHTML = DOMPurify.sanitize(marked.parse(partial));
            safeScrollToBottom();
            
            setTimeout(typeChunk, 40); // Slightly higher delay per tick, but more words per tick
        } else {
            contentElement.innerHTML = DOMPurify.sanitize(marked.parse(text));
            enhanceRenderedContent(contentElement);
            if (actionsElement) actionsElement.style.display = "flex";
            safeScrollToBottom();
        }
    }
    typeChunk();
}

function appendMessage(sender, text, isTyping = false, animate = false, fileContext = null) {
    removeWelcomeScreen();

    if (sender === "AI" && text && !isTyping) {
        text = formatAIResponse(text);
    }

    const row = document.createElement("div");
    row.className = "message-row " + (sender === "You" ? "user-row" : "ai-row");

    const msgDiv = document.createElement("div");
    msgDiv.className = sender === "You" ? "message user-bubble" : "message ai-message-wrap";
    
    if (isTyping) {
        msgDiv.id = "typing-indicator";
        msgDiv.innerHTML = `
            <div class="ai-avatar">
                <svg width="14" height="14" viewBox="0 0 200 200" fill="none">
                    <circle cx="100" cy="100" r="70" stroke="rgba(200,169,110,0.6)" stroke-width="16" stroke-dasharray="25 15" class="q-spin-slow"/>
                    <path d="M100 74 L114 100 L100 126 L86 100 Z" fill="#c8a96e" class="q-heartbeat"/>
                </svg>
            </div>
            <div class="thinking-logo-container">
                <div class="thinking-dots">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
                <span class="thinking-text" id="thinking-label">Thinking...</span>
            </div>`;
        row.appendChild(msgDiv);
        chatBox.appendChild(row);
        chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: 'smooth' });
        return;
    } 

    const contentDiv = document.createElement("div");
    if (sender === "AI") {
        const avatar = document.createElement("div");
        avatar.className = "ai-avatar";
        avatar.innerHTML = `<svg width="14" height="14" viewBox="0 0 200 200" fill="none">
            <circle cx="100" cy="100" r="70" stroke="rgba(200,169,110,0.5)" stroke-width="16" stroke-dasharray="25 15" class="q-spin-slow"/>
            <path d="M100 74 L114 100 L100 126 L86 100 Z" fill="#c8a96e" class="q-heartbeat"/>
        </svg>`;
        msgDiv.appendChild(avatar);
        const proseWrap = document.createElement("div");
        proseWrap.className = "ai-prose-wrap";
        proseWrap.style.flex = "1";
        proseWrap.style.minWidth = "0";
        contentDiv.className = "ai-prose";
        proseWrap.appendChild(contentDiv);
        msgDiv.appendChild(proseWrap);
    } else {
        contentDiv.className = "message-content";
        msgDiv.appendChild(contentDiv);
    }

    let actionsDiv = null;
    if (sender === "AI") {
        actionsDiv = document.createElement("div");
        actionsDiv.className = "message-actions";
        actionsDiv.style.display = animate ? "none" : "flex"; 
        const capturedText = text;
        actionsDiv.innerHTML = `
            <button class="action-btn copy-btn" aria-label="Copy" title="Copy to clipboard">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13">
                    <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
                <span class="copy-label">Copy</span>
            </button>
            <button class="action-btn read-btn" aria-label="Read aloud" title="Read aloud">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" width="13" height="13">
                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                </svg>
            </button>
            <div class="feedback-divider"></div>
            <button class="action-btn feedback-btn thumbs-up" title="Good answer — Hema will remember this">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" width="13" height="13">
                    <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/>
                    <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
                </svg>
            </button>
            <button class="action-btn feedback-btn thumbs-down" title="Wrong answer — correct Hema">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" width="13" height="13">
                    <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/>
                    <path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
                </svg>
            </button>
            <span class="feedback-status"></span>
        `;
        
        const readBtn = actionsDiv.querySelector('.read-btn');
        if (readBtn && 'speechSynthesis' in window) {
            readBtn.onclick = () => {
                window.speechSynthesis.cancel();
                const plain = text.replace(/[#*`_~>]/g, '').replace(/\n/g, ' ');
                const utter = new SpeechSynthesisUtterance(plain);
                utter.rate = 0.95; utter.pitch = 1;
                readBtn.style.color = '#4ade80';
                utter.onend = () => { readBtn.style.color = ''; };
                window.speechSynthesis.speak(utter);
            };
        } else if (readBtn) { readBtn.style.display = 'none'; }

        const feedbackStatus = actionsDiv.querySelector('.feedback-status');
        const thumbsUp   = actionsDiv.querySelector('.thumbs-up');
        const thumbsDown = actionsDiv.querySelector('.thumbs-down');

        const lastQ = document.getElementById("user-input")?.dataset?.lastQ || "";

        async function sendFeedback(rating, correction = "", confirmed = false) {
            try {
                const res = await fetch("/feedback", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ rating, correction, confirmed,
                        question: lastQ,
                        answer: capturedText })
                });
                const data = await res.json();
                feedbackStatus.textContent = data.msg || "Thanks!";
                feedbackStatus.style.color = rating === "up" ? "#4ade80" : "#f87171";
                thumbsUp.disabled = true; thumbsDown.disabled = true;
                setTimeout(() => { feedbackStatus.textContent = ""; }, 3000);
                return data;
            } catch(e) {
                feedbackStatus.textContent = "Could not send feedback.";
                return null;
            }
        }

        if (thumbsUp) thumbsUp.onclick = () => sendFeedback("up");

        if (thumbsDown) thumbsDown.onclick = async () => {
            thumbsDown.disabled = true;

            // Show searching spinner
            const corrBox = document.createElement("div");
            corrBox.className = "correction-box correction-searching";
            corrBox.innerHTML = `<span class="searching-spinner"></span> Searching the web for a better answer...`;
            actionsDiv.after(corrBox);

            // Call backend — it will auto-search Google + GFG + Wikipedia
            let data = null;
            try {
                const res = await fetch("/feedback", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ rating: "down", correction: "", confirmed: false,
                        question: lastQ, answer: capturedText })
                });
                data = await res.json();
            } catch(e) {
                data = { status: "not_found", msg: "Search failed." };
            }

            corrBox.remove();

            if (data && data.status === "found" && data.searched_answer) {
                // Show found answer with confirm / edit / cancel options
                const foundBox = document.createElement("div");
                foundBox.className = "correction-box correction-found";
                foundBox.innerHTML = `
                    <div class="found-answer-label">Found on the web:</div>
                    <div class="found-answer-text">${data.searched_answer}</div>
                    <div class="found-actions">
                        <button class="confirm-btn">✓ Save this</button>
                        <button class="edit-btn">✏ Edit</button>
                        <button class="action-btn correction-cancel">Cancel</button>
                    </div>
                    <textarea class="correction-input edit-area" style="display:none"
                        placeholder="Edit the answer...">${data.searched_answer}</textarea>
                `;
                actionsDiv.after(foundBox);

                foundBox.querySelector(".confirm-btn").onclick = async () => {
                    await sendFeedback("down", data.searched_answer, true);
                    foundBox.innerHTML = `<span style="color:#4ade80">✓ Saved! I'll use this next time.</span>`;
                    setTimeout(() => foundBox.remove(), 2500);
                };

                foundBox.querySelector(".edit-btn").onclick = () => {
                    const ta = foundBox.querySelector(".edit-area");
                    const actions = foundBox.querySelector(".found-actions");
                    ta.style.display = "block";
                    actions.innerHTML = `
                        <button class="confirm-btn save-edit-btn">✓ Save edited</button>
                        <button class="action-btn correction-cancel">Cancel</button>
                    `;
                    actions.querySelector(".save-edit-btn").onclick = async () => {
                        const edited = ta.value.trim();
                        if (!edited) return;
                        await sendFeedback("down", edited, false);
                        foundBox.innerHTML = `<span style="color:#4ade80">✓ Correction saved!</span>`;
                        setTimeout(() => foundBox.remove(), 2500);
                    };
                };

                foundBox.querySelector(".correction-cancel").onclick = () => {
                    foundBox.remove();
                    thumbsDown.disabled = false;
                };

            } else {
                // Not found — show manual input fallback
                const manualBox = document.createElement("div");
                manualBox.className = "correction-box";
                manualBox.innerHTML = `
                    <div style="font-size:0.78rem;color:var(--text-soft);margin-bottom:6px">
                        Couldn't find a better answer online. Type it below:
                    </div>
                    <textarea class="correction-input" rows="3"
                        placeholder="Type the correct answer..."></textarea>
                    <div style="display:flex;gap:8px;margin-top:6px">
                        <button class="confirm-btn correction-send">Save</button>
                        <button class="action-btn correction-cancel">Cancel</button>
                    </div>
                `;
                actionsDiv.after(manualBox);
                manualBox.querySelector(".correction-send").onclick = async () => {
                    const val = manualBox.querySelector(".correction-input").value.trim();
                    if (!val) return;
                    await sendFeedback("down", val, false);
                    manualBox.innerHTML = `<span style="color:#4ade80">✓ Correction saved!</span>`;
                    setTimeout(() => manualBox.remove(), 2500);
                };
                manualBox.querySelector(".correction-cancel").onclick = () => {
                    manualBox.remove();
                    thumbsDown.disabled = false;
                };
            }
        };

        const copyBtn = actionsDiv.querySelector('.copy-btn');
        copyBtn.onclick = () => {
            navigator.clipboard.writeText(text).then(() => {
                const span = copyBtn.querySelector('.copy-label');
                span.textContent = "Copied!"; copyBtn.querySelector("svg").style.display = "none";
                copyBtn.style.color = "#4ade80"; 
                copyBtn.style.borderColor = "rgba(74, 222, 128, 0.3)";
                setTimeout(() => {
                    span.textContent = "Copy"; copyBtn.querySelector("svg").style.display = "";
                    copyBtn.style.color = ""; copyBtn.style.borderColor = "";
                }, 2000);
            });
        };
        msgDiv.appendChild(actionsDiv);
    }

    if (sender === "AI" && typeof marked !== 'undefined') {
        if (animate) {
            typeWriterEffect(contentDiv, msgDiv, text, actionsDiv);
        } else {
            contentDiv.innerHTML = DOMPurify.sanitize(marked.parse(text));
            enhanceRenderedContent(contentDiv);
        }
    } else {
        contentDiv.textContent = text; 
    }
    
    row.appendChild(msgDiv);
    chatBox.appendChild(row);
    
    if (!animate) safeScrollToBottom();
}

async function sendMessage() {
    const text = userInput.value.trim();
    const activeModule = document.getElementById("module-select").value;
    
    if (!text && !attachedFile) return;

    userInput.value = "";
    userInput.style.height = 'auto'; // Reset text box height
    sendBtn.classList.remove('active');

    const fileToDisplay = attachedFile;
    if (fileToDisplay) {
        const ext = fileToDisplay.name.split('.').pop().toLowerCase();
        const isImg = ['jpg','jpeg','png','gif','webp'].includes(ext);
        const label = isImg ? "Analysing image" : "Reading file";
        const inputEl = document.getElementById("user-input");
        if (inputEl) inputEl.dataset.lastQ = text || "";
        const userText = text || `[${label}: ${fileToDisplay.name}]`;
        pushClientHistory("user", userText);
        appendMessage("You", userText);
    } else {
        appendMessage("You", text);
    }
    appendMessage("AI", "", true, fileToDisplay ? fileToDisplay.name : null);

    const fileToSend = attachedFile ? { ...attachedFile } : null;
    if (attachedFile) removeFilePreview();

    try {
        let endpoint = "/chat";
        let payload = {
            message: text,
            module: activeModule,
            history: clientHistory.slice(0, -1) 
        };

        if (fileToSend) {
            endpoint = "/upload";
            payload = {
                message: text,
                module: activeModule,
                history: clientHistory.slice(0, -1),
                file_data: fileToSend.data_b64,
                filename: fileToSend.name,
                mimetype: fileToSend.type
            };
        }

        const response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Server disconnected");

        const data = await response.json();
        const indicator = document.getElementById("typing-indicator");
        if (indicator) indicator.closest('.message-row').remove();

        pushClientHistory("ai", data.reply);
        appendMessage("AI", data.reply, false, true);
        const inp = document.getElementById('user-input');
        if (inp) inp.placeholder = 'Message Hema...';
        
    } catch (error) {
        console.warn("Backend unavailable. Using simulated response.");
        setTimeout(() => {
            const indicator = document.getElementById("typing-indicator");
            if (indicator) indicator.closest('.message-row').remove();
            
            const mockReply = `I am currently running in **Preview Mode** because my backend isn't reachable.\n\nYou asked about: \`${text}\`\n\n*If you deploy this alongside your python/node backend, I will connect automatically!*`;
            appendMessage("AI", mockReply, false, true);
            const inp2 = document.getElementById('user-input');
            if (inp2) inp2.placeholder = 'Message Hema...';
        }, 1200);
    } finally {
        if(window.innerWidth > 768) userInput.focus();
    }
}

async function clearMemory() {
    try {
        await fetch("/clear", { method: "POST" }).catch(() => {});
        chatBox.innerHTML = `
            <div class="welcome-container" id="welcome-screen">
                <div class="welcome-logo">
                    <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="qGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="#ffffff" /><stop offset="100%" stop-color="#737373" />
                            </linearGradient>
                            <linearGradient id="qGrad2" x1="100%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stop-color="#a3a3a3" /><stop offset="100%" stop-color="#404040" />
                            </linearGradient>
                            <radialGradient id="qGlow" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stop-color="rgba(255, 255, 255, 0.15)" /><stop offset="100%" stop-color="rgba(255, 255, 255, 0)" />
                            </radialGradient>
                        </defs>
                        <circle cx="100" cy="100" r="90" fill="url(#qGlow)" class="q-breathe" />
                        <circle cx="100" cy="100" r="85" stroke="url(#qGrad1)" stroke-width="2" stroke-dasharray="10 30 60 20" stroke-linecap="round" class="q-spin-slow" />
                        <circle cx="100" cy="100" r="72" stroke="url(#qGrad2)" stroke-width="4" stroke-dasharray="1 15 40 15" stroke-linecap="round" class="q-spin-fast-rev" />
                        <g class="q-breathe">
                            <path d="M 100 35 C 135 65, 135 135, 100 165 C 65 135, 65 65, 100 35 Z" stroke="rgba(255, 255, 255, 0.6)" stroke-width="1.5" />
                            <path d="M 100 35 C 135 65, 135 135, 100 165 C 65 135, 65 65, 100 35 Z" stroke="rgba(200, 200, 200, 0.5)" stroke-width="1.5" transform="rotate(60 100 100)" />
                            <path d="M 100 35 C 135 65, 135 135, 100 165 C 65 135, 65 65, 100 35 Z" stroke="rgba(150, 150, 150, 0.4)" stroke-width="1.5" transform="rotate(120 100 100)" />
                        </g>
                        <path d="M 100 75 L 107 93 L 125 100 L 107 107 L 100 125 L 93 107 L 75 100 L 93 93 Z" fill="#ffffff" class="q-heartbeat" />
                    </svg>
                </div>
                <div class="welcome-title clickable-title" id="welcome-title" onclick="editName()" title="Click to change your name">Hello, I am <span>Hema</span></div>
                <input type="text" id="name-input" class="name-input" placeholder="What should I call you?" onkeypress="saveName(event)">
                <div class="welcome-subtitle" id="quote-container" style="font-style: italic;">
                </div>
            </div>
        `;
        
        initWelcomeScreen();
        
        if (window.innerWidth <= 768) {
            sidebar.classList.add("collapsed");
            overlay.classList.remove("active");
        }
    } catch (error) {
        console.error("Failed to clear memory", error);
    }
}

function loadHistory(topic) {
    chatBox.innerHTML = '';
    if (window.innerWidth <= 768) {
        sidebar.classList.add("collapsed");
        overlay.classList.remove("active");
    }
    
    appendMessage("You", `Can you tell me about ${topic}?`);
    setTimeout(() => {
        appendMessage("AI", `Here is a summary of our past conversation about **${topic}**.\n\n*(This is a simulated history load demonstrating the button functionality.)*`, false, true); 
    }, 500);
}