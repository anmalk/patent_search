'use strict';

// ── Состояние ──
let selectedModel = 'e5-base';

// ── Выбор модели ──
document.querySelectorAll('.model-item').forEach(item => {
  item.addEventListener('click', () => {
    document.querySelectorAll('.model-item').forEach(i => i.classList.remove('active'));
    item.classList.add('active');
    selectedModel = item.dataset.model;
  });
});

// ── Счётчик символов ──
const queryEl    = document.getElementById('query');
const charCountEl = document.getElementById('char-count');

queryEl.addEventListener('input', () => {
  const n = queryEl.value.length;
  charCountEl.textContent = n.toLocaleString('ru') + ' символов';
  charCountEl.className = 'char-count' + (n > 2000 ? ' warn' : '');
});

// ── Ctrl+Enter ──
queryEl.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') doSearch();
});

// ── Мониторинг состояния системы ──
async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();

    setDot('dot-qdrant', d.qdrant_connected);
    setDot('dot-model',  d.model_loaded);

    document.getElementById('lbl-qdrant').textContent =
      d.qdrant_connected ? 'Qdrant' : 'Qdrant offline';

    document.getElementById('lbl-model').textContent =
      d.model_loaded
        ? (d.model_name || 'Модель')
        : 'Нет модели';

    const docLbl = document.getElementById('lbl-docs');
    if (d.documents_indexed > 0) {
      docLbl.textContent = d.documents_indexed.toLocaleString('ru') + ' патентов';
    } else {
      docLbl.textContent = '';
    }
  } catch (e) {
    setDot('dot-qdrant', false);
    setDot('dot-model',  false);
  }
}

function setDot(id, ok) {
  const el = document.getElementById(id);
  el.className = 'dot ' + (ok ? 'ok' : 'err');
}

checkHealth();
setInterval(checkHealth, 30_000);

// ── Поиск ──
async function doSearch() {
  const query = queryEl.value.trim();
  if (!query) { queryEl.focus(); return; }

  const btn       = document.getElementById('btn-search');
  const statusBar = document.getElementById('search-status');
  const errorBar  = document.getElementById('error-bar');
  const results   = document.getElementById('results');

  btn.classList.add('loading');
  btn.disabled = true;
  statusBar.style.display = 'none';
  errorBar.style.display  = 'none';

  try {
    const res = await fetch('/api/search', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k:      parseInt(document.getElementById('top-k').value),
        ipc_filter: document.getElementById('ipc-filter').value.trim() || null,
        model_name: selectedModel,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `Ошибка сервера: ${res.status}`);
    }

    const data = await res.json();
    renderResults(data, results);

    // Строка статуса
    statusBar.style.display = 'flex';
    statusBar.innerHTML = `
      <span>Найдено: <span class="hi">${data.total}</span></span>
      <span>Время: <span class="hi">${data.time_ms} мс</span></span>
      <span>Модель: <span class="tag">${data.model_used}</span></span>
    `;

  } catch (e) {
    errorBar.style.display = 'block';
    errorBar.textContent   = '⚠ ' + e.message;
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

// ── Рендер результатов ──
function renderResults(data, container) {
  if (!data.results.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">◌</div>
        <div class="empty-title">Ничего не найдено</div>
        <div class="empty-sub">Попробуйте изменить запрос или убрать фильтр МПК</div>
      </div>`;
    return;
  }

  // Порог схожести
  const SIM_THRESHOLD = 0.7;

  // Слова из запроса для подсветки
  const words = data.query.toLowerCase()
    .split(/\s+/)
    .filter(w => w.length > 4)
    .slice(0, 10)
    .map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));

  function highlight(text) {
    if (!words.length) return text;
    const re = new RegExp(`(${words.join('|')})`, 'gi');
    return text.replace(re, '<span class="hl">$1</span>');
  }

  const html = data.results.map((r, i) => {
    const score = r.score;
    const isSimilar = score >= SIM_THRESHOLD;
    const verdict = isSimilar ? '✓ Аналог (схож)' : '✗ Не похож';
    const verdictClass = isSimilar ? 'verdict-similar' : 'verdict-different';

    return `
    <div class="result-card" style="animation-delay:${i * 0.04}s">
      <div class="result-header">
        <span class="result-rank">${r.rank}.</span>
        <span class="result-id">${r.patent_id}</span>
        ${r.title ? `<span class="result-title">${r.title}</span>` : ''}
        <span class="result-ipc">${r.ipc || '—'}</span>
      </div>
      <div class="score-row">
        <div class="score-bar">
          <div class="score-fill" style="width:${Math.round(score * 100)}%"></div>
        </div>
        <span class="score-val">${score.toFixed(3)}</span>
        <span class="verdict ${verdictClass}">${verdict}</span>
      </div>
      <div class="result-abstract">${highlight(r.abstract)}</div>
    </div>`;
  }).join('');

  container.innerHTML = html;
}