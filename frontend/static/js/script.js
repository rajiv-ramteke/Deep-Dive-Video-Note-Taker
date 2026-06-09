/**
 * Deep-Dive Video Note Taker — script.js
 * Frontend logic: upload, polling, results rendering, RAG search
 */

// ── State ─────────────────────────────────────────────────────────────────────
let currentJobId   = null;
let pollingTimer   = null;
let selectedFile   = null;
let currentNotes   = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const fileInput        = document.getElementById('file-input');
const dropZone         = document.getElementById('drop-zone');
const filePreview      = document.getElementById('file-preview');
const previewName      = document.getElementById('preview-name');
const previewSize      = document.getElementById('preview-size');
const processBtn       = document.getElementById('process-btn');
const progressSection  = document.getElementById('progress-section');
const progressBar      = document.getElementById('progress-bar');
const progressLabel    = document.getElementById('progress-label');
const progressPct      = document.getElementById('progress-pct');
const resultsSection   = document.getElementById('results-section');
const notesOutput      = document.getElementById('notes-output');
const actionsOutput    = document.getElementById('actions-output');
const transcriptOutput = document.getElementById('transcript-output');
const qaOutput         = document.getElementById('qa-output');

// ── File Input ────────────────────────────────────────────────────────────────

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// Drag & drop
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  const ALLOWED = ['video/mp4','video/avi','video/quicktime','video/x-matroska',
                   'video/webm','audio/mpeg','audio/wav','audio/mp4','audio/x-m4a'];
  const ext = file.name.split('.').pop().toLowerCase();
  const ALLOWED_EXT = ['mp4','avi','mov','mkv','webm','mp3','wav','m4a'];

  if (!ALLOWED_EXT.includes(ext)) {
    showToast('⚠️ Unsupported file type. Please upload a video or audio file.', 'error');
    return;
  }
  if (file.size > 500 * 1024 * 1024) {
    showToast('⚠️ File too large. Maximum 500 MB.', 'error');
    return;
  }

  selectedFile = file;
  previewName.textContent = file.name;
  previewSize.textContent = formatBytes(file.size);
  filePreview.classList.remove('hidden');
  processBtn.disabled = false;
  showToast('✅ File selected: ' + file.name);
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  filePreview.classList.add('hidden');
  processBtn.disabled = true;
}

// ── Processing ────────────────────────────────────────────────────────────────

async function startProcessing() {
  if (!selectedFile) return;

  const openaiKey = document.getElementById('openai-key').value.trim();

  const formData = new FormData();
  formData.append('file', selectedFile);
  if (openaiKey) formData.append('openai_key', openaiKey);

  // UI: show progress
  progressSection.classList.remove('hidden');
  processBtn.disabled = true;
  processBtn.classList.add('loading');
  updateProgress(0, 'Uploading video…');
  resetProgressSteps();

  try {
    const res = await fetch('/api/v1/process/upload', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Upload failed');
    }

    const data = await res.json();
    currentJobId = data.data.job_id;
    showToast('🚀 Processing started! Job: ' + currentJobId.slice(0, 8) + '…');

    // Start polling
    startPolling();

  } catch (err) {
    showToast('❌ Error: ' + err.message, 'error');
    processBtn.disabled = false;
    processBtn.classList.remove('loading');
    updateProgress(0, 'Upload failed.');
  }
}

function startPolling() {
  if (pollingTimer) clearInterval(pollingTimer);
  pollingTimer = setInterval(pollStatus, 2500);
}

async function pollStatus() {
  if (!currentJobId) return;

  try {
    const res  = await fetch(`/api/v1/status/${currentJobId}`);
    const data = await res.json();
    const job  = data.data;

    updateProgress(job.progress, job.message);
    syncProgressSteps(job.progress);

    if (job.status === 'complete') {
      clearInterval(pollingTimer);
      await loadResults(currentJobId);
      processBtn.classList.remove('loading');
    } else if (job.status === 'error') {
      clearInterval(pollingTimer);
      showToast('❌ Processing error: ' + job.message, 'error');
      processBtn.disabled = false;
      processBtn.classList.remove('loading');
    }
  } catch (err) {
    console.error('Polling error:', err);
  }
}

function updateProgress(pct, label) {
  progressBar.style.width = pct + '%';
  progressPct.textContent = pct + '%';
  progressLabel.textContent = label;
}

function resetProgressSteps() {
  document.querySelectorAll('.progress-step').forEach(s => {
    s.classList.remove('active', 'done');
  });
}

function syncProgressSteps(pct) {
  const steps = [
    { el: document.querySelector('[data-step="audio"]'),  threshold: 5  },
    { el: document.querySelector('[data-step="asr"]'),    threshold: 20 },
    { el: document.querySelector('[data-step="llm"]'),    threshold: 50 },
    { el: document.querySelector('[data-step="rag"]'),    threshold: 75 },
    { el: document.querySelector('[data-step="notes"]'),  threshold: 90 },
  ];

  steps.forEach(({ el, threshold }, i) => {
    if (!el) return;
    el.classList.remove('active', 'done');
    if (pct >= threshold && (i === steps.length - 1 || pct < steps[i + 1]?.threshold)) {
      el.classList.add('active');
    } else if (pct >= threshold) {
      el.classList.add('done');
    }
  });
}

// ── Load & Render Results ─────────────────────────────────────────────────────

async function loadResults(jobId) {
  try {
    const res  = await fetch(`/api/v1/notes/${jobId}`);
    const data = await res.json();
    currentNotes = data.data;

    renderNotes(currentNotes.notes, currentNotes.markdown);
    renderQuiz(currentNotes.notes.quiz || []);
    renderTopics(currentNotes.notes.topics || []);
    renderQA(currentNotes.notes.qa_pairs || []);
    renderTranscript(currentNotes.notes.transcript_segments || []);

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    showToast('✅ Notes ready! Scroll down to view.', 'success');

    updateProgress(100, '✅ Complete!');
    syncProgressSteps(100);

  } catch (err) {
    showToast('❌ Failed to load results: ' + err.message, 'error');
  }
}

// ── Render: Notes (Markdown → HTML) ──────────────────────────────────────────

function renderNotes(notes, markdown) {
  if (!markdown) {
    notesOutput.innerHTML = '<p style="color:var(--clr-text-muted)">No notes generated.</p>';
    return;
  }
  notesOutput.innerHTML = markdownToHtml(markdown);
}

function markdownToHtml(md) {
  // Headings
  md = md.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  md = md.replace(/^## (.+)$/gm,  '<h2>$1</h2>');
  md = md.replace(/^# (.+)$/gm,   '<h1>$1</h1>');
  // Bold / Italic
  md = md.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  md = md.replace(/\*(.+?)\*/g,     '<em>$1</em>');
  // Inline code
  md = md.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Code blocks
  md = md.replace(/```[\w]*\n([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
  // Horizontal rule
  md = md.replace(/^---$/gm, '<hr/>');
  // Blockquote
  md = md.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
  // Unordered list items
  md = md.replace(/^\s*[-•*] (.+)$/gm, '<li>$1</li>');
  md = md.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
  // Details / summary
  md = md.replace(/<details>/g,  '<details style="margin:16px 0">');
  md = md.replace(/<summary>(.+?)<\/summary>/g, '<summary style="cursor:pointer;font-weight:600;color:var(--clr-accent)">$1</summary>');
  // Paragraphs: wrap lines that are not already HTML
  md = md.replace(/^(?!<[a-z])(.+)$/gm, (line) => {
    if (line.trim() === '' || line.startsWith('<')) return line;
    return `<p>${line}</p>`;
  });
  return md;
}

// ── Translate Notes ───────────────────────────────────────────────────────────

async function translateNotes() {
  if (!currentJobId) {
    showToast('⚠️ Process a video first to translate notes.', 'error');
    return;
  }
  const lang = document.getElementById('translate-lang').value;
  const btn = document.getElementById('translate-btn');
  const originalText = btn.innerHTML;
  
  btn.innerHTML = '⏳ Translating...';
  btn.disabled = true;
  notesOutput.innerHTML = `<div style="text-align:center;padding:40px;color:var(--clr-primary-l);">Translating notes to ${lang}...</div>`;

  try {
    const res = await fetch('/api/v1/notes/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: currentJobId, language: lang })
    });
    
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Translation failed');
    }
    
    const data = await res.json();
    
    // Update global state
    currentNotes.markdown = data.data.translated_markdown;
    if (data.data.translated_quiz) currentNotes.notes.quiz = data.data.translated_quiz;
    if (data.data.translated_topics) currentNotes.notes.topics = data.data.translated_topics;
    if (data.data.translated_qa_pairs) currentNotes.notes.qa_pairs = data.data.translated_qa_pairs;
    if (data.data.translated_transcript) currentNotes.notes.transcript_segments = data.data.translated_transcript;

    // Re-render everything
    renderNotes(currentNotes.notes, currentNotes.markdown);
    renderQuiz(currentNotes.notes.quiz || []);
    renderTopics(currentNotes.notes.topics || []);
    renderQA(currentNotes.notes.qa_pairs || []);
    renderTranscript(currentNotes.notes.transcript_segments || []);
    
    showToast(`✅ Everything translated to ${lang}`, 'success');
  } catch (err) {
    showToast('❌ Translation error: ' + err.message, 'error');
    // Restore original on error
    renderNotes(currentNotes.notes, currentNotes.markdown);
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
  }
}

// ── Render: Quiz ──────────────────────────────────────────────────────────────

function renderQuiz(quiz) {
  const container = document.getElementById('quiz-output');
  if (!quiz || !quiz.length) {
    container.innerHTML = '<p style="color:var(--clr-text-muted);text-align:center;padding:40px">No quiz available.</p>';
    return;
  }

  window.checkQuizAnswer = function(qIndex, optIndex, correctIndex, el) {
    const parent = el.closest('.action-card');
    const buttons = parent.querySelectorAll('button');
    
    // Disable all buttons in this question
    buttons.forEach(btn => btn.disabled = true);
    
    // Highlight correct answer
    buttons[correctIndex].style.backgroundColor = 'rgba(16, 185, 129, 0.2)'; // Green tint
    buttons[correctIndex].style.borderColor = '#10b981';
    buttons[correctIndex].style.color = '#10b981';
    buttons[correctIndex].innerHTML += ' ✅';

    // Highlight selected if incorrect
    if (optIndex !== correctIndex) {
      el.style.backgroundColor = 'rgba(239, 68, 68, 0.2)'; // Red tint
      el.style.borderColor = '#ef4444';
      el.style.color = '#ef4444';
      el.innerHTML += ' ❌';
    }
  };

  container.innerHTML = quiz.map((q, qIndex) => `
    <div class="action-card" style="flex-direction:column; gap:16px; padding:24px;">
      <div style="font-size:16px; font-weight:600; color:var(--clr-text-main);">Q${qIndex + 1}: ${escHtml(q.question)}</div>
      <div style="display:flex; flex-direction:column; gap:8px;">
        ${(q.options || []).map((opt, optIndex) => `
          <button 
            onclick="checkQuizAnswer(${qIndex}, ${optIndex}, ${q.correct_index}, this)"
            style="text-align:left; padding:12px; border:1px solid var(--clr-border); border-radius:6px; background:var(--clr-surface); color:var(--clr-text-main); font-family:inherit; font-size:14px; cursor:pointer; transition:all 0.2s;"
            onmouseover="this.style.borderColor='var(--clr-primary)'"
            onmouseout="if(!this.disabled) this.style.borderColor='var(--clr-border)'"
          >
            ${escHtml(opt)}
          </button>
        `).join('')}
      </div>
    </div>
  `).join('');
}

// ── Render: Topic Summaries ───────────────────────────────────────────────────

function renderTopics(topics) {
  if (!topics || !topics.length) {
    actionsOutput.innerHTML = '<p style="color:var(--clr-text-muted);text-align:center;padding:40px">No topics extracted yet. Process a video to see structured topic summaries.</p>';
    return;
  }

  actionsOutput.innerHTML = topics.map(item => `
    <div class="action-card" style="flex-direction:column; gap:12px; padding:20px;">
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-size:20px;">🔹</span>
        <div style="font-size:17px;font-weight:700;color:var(--clr-text-main);">${escHtml(item.topic || '')}</div>
      </div>
      <div style="color:var(--clr-primary-l);font-style:italic;font-size:14px;padding-left:30px;">${escHtml(item.summary || '')}</div>
      <ul style="margin:0;padding-left:46px;color:var(--clr-text-muted);">
        ${(item.key_points || []).map(kp => `<li style="margin-bottom:4px;">${escHtml(kp)}</li>`).join('')}
      </ul>
    </div>
  `).join('');
}

// ── Render: Q&A ───────────────────────────────────────────────────────────────

function renderQA(qaPairs) {
  if (!qaPairs.length) {
    qaOutput.innerHTML = '<p style="color:var(--clr-text-muted);text-align:center;padding:40px">No Q&A generated.</p>';
    return;
  }

  qaOutput.innerHTML = qaPairs.map(item => `
    <div class="action-card" style="flex-direction:column; gap:8px;">
      <div style="font-weight:600; color:var(--clr-accent);">Q: ${escHtml(item.question)}</div>
      <div style="color:var(--clr-text-main);">A: ${escHtml(item.answer)}</div>
    </div>
  `).join('');
}

// ── Render: Transcript ────────────────────────────────────────────────────────

function renderTranscript(segments) {
  if (!segments.length) {
    transcriptOutput.innerHTML = '<p style="color:var(--clr-text-muted)">No transcript available.</p>';
    return;
  }

  transcriptOutput.innerHTML = segments.map(seg => `
    <div class="transcript-seg">
      <div class="transcript-ts">▶ ${seg.start_ts || formatSec(seg.start)} — ${seg.end_ts || formatSec(seg.end)}</div>
      <div class="transcript-text">${escHtml(seg.text)}</div>
    </div>
  `).join('');
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

function switchTab(name, btn) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}

// ── RAG Search ────────────────────────────────────────────────────────────────

async function doSearch() {
  const query = document.getElementById('search-input').value.trim();
  if (!query) { showToast('⚠️ Enter a search query.', 'error'); return; }
  if (!currentJobId) { showToast('⚠️ Process a video first.', 'error'); return; }

  const container = document.getElementById('search-results');
  container.innerHTML = '<div style="text-align:center;padding:24px;color:var(--clr-primary-l);">🧠 AI is thinking...</div>';

  try {
    const res = await fetch('/api/v1/query/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: currentJobId, query, top_k: 5 }),
    });

    const data = await res.json();
    const answer = data.data?.answer || '';

    if (!answer) {
      container.innerHTML = '<div style="text-align:center;padding:24px;color:var(--clr-text-muted)">No answer could be generated. Try rephrasing your question.</div>';
      return;
    }

    container.innerHTML = `
      <div class="action-card" style="flex-direction:column;gap:14px;padding:24px;border:1px solid var(--clr-primary);background:rgba(59,130,246,0.05);">
        <div style="display:flex;align-items:center;gap:10px;">
          <span style="font-size:24px;">✨</span>
          <span style="font-size:16px;font-weight:700;color:var(--clr-primary-l);">AI Answer</span>
        </div>
        <div style="color:var(--clr-text-main);line-height:1.8;padding-left:34px;">${markdownToHtml(answer)}</div>
      </div>
    `;

  } catch (err) {
    container.innerHTML = `<div style="text-align:center;padding:24px;color:var(--clr-red)">Error: ${err.message}</div>`;
  }
}

// ── Downloads ─────────────────────────────────────────────────────────────────

function downloadMd() {
  if (!currentJobId) { showToast('⚠️ No notes to download.', 'error'); return; }
  window.open(`/api/v1/notes/${currentJobId}/download`, '_blank');
}

function copyNotes() {
  if (!currentNotes?.markdown) { showToast('⚠️ No notes to copy.', 'error'); return; }
  navigator.clipboard.writeText(currentNotes.markdown)
    .then(() => showToast('📋 Notes copied to clipboard!'))
    .catch(() => showToast('❌ Copy failed.', 'error'));
}

// ── Toast ─────────────────────────────────────────────────────────────────────

let toastTimer = null;
function showToast(msg, type = 'info') {
  const toast   = document.getElementById('toast');
  const toastMsg = document.getElementById('toast-msg');
  toastMsg.textContent = msg;
  toast.classList.remove('hidden');
  toast.style.borderColor = type === 'error' ? 'rgba(239,68,68,0.4)' : 'rgba(255,255,255,0.12)';

  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add('hidden'), 4000);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatBytes(bytes) {
  if (bytes < 1024)            return bytes + ' B';
  if (bytes < 1024 * 1024)     return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024**3)         return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  return (bytes / (1024**3)).toFixed(2) + ' GB';
}

function formatSec(s) {
  s = Math.round(s || 0);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
}

function escHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
