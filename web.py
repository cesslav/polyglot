# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
import os
import onnxruntime as ort
import numpy as np
from flask import Flask, request, jsonify, render_template_string, send_from_directory, send_file
from transformers import AutoTokenizer


app = Flask(__name__)
HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Полиглот</title>

  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link
    href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@400;500&display=swap"
    rel="stylesheet"
  />

  <style>
    *, *::before, *::after {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    :root {
      --bg: #000000;
      --surface: #0d0d0d;
      --border: #5cb84b;
      --text: #ffffff;
      --muted: #888888;
      --accent: #5cb84b;
      --accent-dim: rgba(92,184,75,0.12);

      --radius: 12px;

      --font-head: 'Syne', sans-serif;
      --font-mono: 'DM Mono', monospace;
    }

    html {
      scroll-behavior: smooth;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-mono);
      min-height: 100vh;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 100;

      display: flex;
      align-items: center;
      gap: 20px;

      padding: 10px 24px;

      background: rgba(0,0,0,0.85);

      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);

      border-bottom: 1px solid rgba(92,184,75,0.2);
    }

    .logo {
      display: flex;
      align-items: center;
      flex-shrink: 0;
    }

    .logo img {
      max-height: 80px;
      width: auto;
      height: auto;
      display: block;
      object-fit: contain;
    }

    .model-wrap {
      margin-left: auto;

      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 3px;
    }

    .model-label {
      font-size: 0.65rem;
      color: var(--muted);

      text-transform: uppercase;
      letter-spacing: 0.1em;
      white-space: nowrap;
    }

    .model-select-container {
      position: relative;
    }

    select#modelSelect {
      appearance: none;
      -webkit-appearance: none;

      min-width: 140px;

      padding: 6px 32px 6px 12px;

      background: var(--surface);
      color: var(--text);

      border: 1px solid var(--border);
      border-radius: 8px;

      font-family: var(--font-mono);
      font-size: 0.82rem;

      cursor: pointer;
      outline: none;

      transition:
        border-color 0.2s,
        box-shadow 0.2s;
    }

    select#modelSelect:hover,
    select#modelSelect:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-dim);
    }

    .select-arrow {
      position: absolute;
      top: 50%;
      right: 10px;

      transform: translateY(-50%);

      pointer-events: none;

      color: var(--accent);
      font-size: 0.7rem;
    }

    main {
      min-height: 100vh;

      display: flex;
      flex-direction: column;
      justify-content: center;

      padding: 32px 24px 64px;
    }

    .translator-card {
      width: 100%;
      max-width: 1100px;
      margin: 0 auto;

      background: var(--surface);

      border: 1px solid rgba(92,184,75,0.22);
      border-radius: 12px;

      overflow: clip;

      isolation: isolate;

      position: relative;
    }

    .panels {
      display: grid;
      grid-template-columns: 1fr 1fr;
    }

    .panel {
      display: flex;
      flex-direction: column;
      min-height: 320px;

      min-width: 0;
    }

    .panel + .panel {
      border-left: 1px solid rgba(92,184,75,0.2);
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;

      padding: 14px 20px 10px;

      border-bottom: 1px solid rgba(92,184,75,0.15);

      color: var(--muted);

      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }

    .panel-header .char-count {
      font-size: 0.68rem;
    }

    textarea {
      flex: 1;

      width: 100%;
      min-width: 0;

      padding: 20px;

      background: transparent;
      color: var(--text);

      border: none;
      outline: none;

      resize: none;

      font-family: var(--font-mono);
      font-size: 1rem;
      line-height: 1.6;

      caret-color: var(--accent);

      overflow: auto;

      clip-path: inset(0 round 0);

      display: block;
    }

    textarea::placeholder {
      color: #444;
    }

    textarea::selection {
      background: rgba(92,184,75,0.28);
      color: #ffffff;
    }

    textarea::-moz-selection {
      background: rgba(92,184,75,0.28);
      color: #ffffff;
    }

    .output-area {
      flex: 1;

      padding: 20px;

      background: transparent;
      color: var(--text);

      font-family: var(--font-mono);
      font-size: 1rem;
      line-height: 1.6;

      white-space: pre-wrap;
      word-break: break-word;

      min-width: 0;
    }

    .output-area.empty {
      color: #444;
    }

    .spinner-wrap {
      display: none;

      align-items: center;
      justify-content: center;
      gap: 10px;

      flex: 1;
      padding: 20px;

      color: var(--muted);
      font-size: 0.8rem;
    }

    .spinner-wrap.active {
      display: flex;
    }

    .output-area.hidden {
      display: none;
    }

    .dot-spinner {
      display: flex;
      gap: 5px;
    }

    .dot-spinner span {
      width: 7px;
      height: 7px;

      border-radius: 50%;

      background: var(--accent);

      animation: bounce 1.2s infinite ease-in-out;
    }

    .dot-spinner span:nth-child(2) {
      animation-delay: 0.2s;
    }

    .dot-spinner span:nth-child(3) {
      animation-delay: 0.4s;
    }

    @keyframes bounce {
      0%, 80%, 100% {
        transform: scale(0.6);
        opacity: 0.4;
      }

      40% {
        transform: scale(1);
        opacity: 1;
      }
    }

    .panel-footer {
      display: flex;
      align-items: center;
      gap: 10px;

      min-height: 48px;

      padding: 10px 20px;

      border-top: 1px solid rgba(92,184,75,0.1);
    }

    .btn-copy {
      margin-left: auto;

      padding: 5px 14px;

      background: none;
      color: var(--accent);

      border: 1px solid rgba(92,184,75,0.4);
      border-radius: 6px;

      font-family: var(--font-mono);
      font-size: 0.75rem;

      cursor: pointer;

      transition:
        background 0.15s,
        border-color 0.15s;
    }

    .btn-copy:hover {
      background: var(--accent-dim);
      border-color: var(--accent);
    }

    .news-section {
      width: 100%;
      max-width: 1100px;

      margin: 72px auto 80px;
      padding: 0 0 40px;
    }

    .news-section h2 {
      margin-bottom: 28px;

      color: var(--accent);

      font-family: var(--font-head);
      font-size: 1.05rem;
      font-weight: 600;

      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .news-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 20px;
    }

    .news-card {
      padding: 22px 22px 18px;

      background: var(--surface);

      border: 1px solid rgba(92,184,75,0.18);
      border-radius: var(--radius);

      transition:
        border-color 0.2s,
        transform 0.2s;
    }

    .news-card:hover {
      border-color: var(--accent);
      transform: translateY(-2px);
    }

    .news-card .date {
      margin-bottom: 10px;

      color: var(--accent);

      font-size: 0.7rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }

    .news-card h3 {
      margin-bottom: 10px;

      font-family: var(--font-head);
      font-size: 0.95rem;
      font-weight: 600;
      line-height: 1.35;
    }

    .news-card p {
      color: var(--muted);

      font-size: 0.8rem;
      line-height: 1.6;
    }

    .download-section {
      width: 100%;
      max-width: 1100px;

      margin: 0 auto 80px;
      padding: 28px 22px;

      background: var(--surface);

      border: 1px solid rgba(92,184,75,0.18);
      border-radius: var(--radius);

      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 20px;
    }

    .download-section h2 {
      color: var(--accent);

      font-family: var(--font-mono);
      font-size: 1.05rem;
      font-weight: 600;

      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .btn-download {
      display: inline-flex;
      align-items: center;
      gap: 10px;

      padding: 10px 26px;

      background: var(--accent);
      color: #000;

      border: none;
      border-radius: 8px;

      font-family: var(--font-mono);
      font-size: 0.85rem;
      font-weight: 500;

      text-decoration: none;

      cursor: pointer;

      transition:
        opacity 0.15s,
        transform 0.15s,
        box-shadow 0.15s;
    }

    .btn-download:hover {
      opacity: 0.88;
      transform: translateY(-2px);
      box-shadow: 0 6px 24px rgba(92,184,75,0.28);
    }

    .btn-download svg {
      flex-shrink: 0;
    }

    .download-desc {
      color: var(--muted);

      font-size: 0.8rem;
      line-height: 1.7;
    }

    .download-desc strong {
      color: var(--text);
    }

    @media (max-width: 768px) {
      .download-section {
        margin: 0 0 60px;
        padding: 24px 20px;
      }
    }

    footer {
      padding: 20px 24px;

      text-align: center;

      border-top: 1px solid rgba(255,255,255,0.06);

      font-size: 0.72rem;
      color: #333;
    }

    footer a {
      color: #444;
      text-decoration: none;
    }

    footer a:hover {
      color: var(--accent);
    }

    @media (max-width: 768px) {

      header {
        gap: 12px;
        padding: 8px 16px;
      }

      .panels {
        grid-template-columns: 1fr;
      }

      .panel + .panel {
        border-left: none;
        border-top: 1px solid rgba(92,184,75,0.2);
      }

      .panel {
        min-height: 200px;
      }

      main {
        padding: 20px 16px 48px;
      }

      .news-section {
        margin-top: 48px;
        padding: 0 16px 40px;
      }
    }

    @media (max-width: 480px) {

      header {
        padding: 6px 16px;
      }

      .logo img {
        max-height: 72px;
      }

      select#modelSelect {
        min-width: 110px;
        font-size: 0.78rem;
      }

      textarea,
      .output-area {
        padding: 14px;
        font-size: 0.9rem;
      }

      .translator-card {
        border-radius: 12px;
      }

      .news-section h2 {
        font-size: 0.9rem;
      }

      .news-card h3 {
        font-size: 0.88rem;
      }
    }

    @media (min-width: 1400px) {

      main {
        padding: 48px 48px 80px;
      }

      .news-section {
        max-width: 1100px;
      }
    }

    @media print {

      body {
        background: #ffffff;
      }

      .translator-card {

        border: 1px solid #5cb84b !important;

        border-radius: 10px;

        overflow: hidden !important;

        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;

        box-shadow: inset 0 0 0 1px #5cb84b;
      }

      .panel + .panel {
        border-left: 1px solid rgba(92,184,75,0.35);
      }

      .panel-header,
      .panel-footer {
        border-color: rgba(92,184,75,0.25);
      }
    }

    :focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }

  </style>
</head>

<body>

  <header>

    <div class="logo">
      <img src="/logo" alt="Полиглот" />
    </div>

    <div class="model-wrap">

      <span class="model-label">
        Модель
      </span>

      <div class="model-select-container">

        <select
          id="modelSelect"
          title="Выберите модель перевода"
        >
          <option value="">
            — загрузка —
          </option>
        </select>

        <span class="select-arrow">
          ▾
        </span>

      </div>
    </div>

  </header>

  <main>

    <div class="translator-card">

      <div class="panels">

        <!-- Input -->
        <div class="panel">

          <div class="panel-header">

            <span>
              Исходный текст
            </span>

            <span
              class="char-count"
              id="charCount"
            >
              0 / 2000
            </span>

          </div>

          <textarea
            id="inputText"
            placeholder="Введите текст для перевода…"
            maxlength="2000"
            spellcheck="true"
            aria-label="Текст для перевода"
          ></textarea>

          <div class="panel-footer">

            <button
              class="btn-copy"
              id="btnCopySrc"
            >
              Копировать
            </button>

          </div>

        </div>

        <!-- Output -->
        <div class="panel">

          <div class="panel-header">

            <span>
              Перевод
            </span>

          </div>

          <div
            class="spinner-wrap"
            id="spinnerWrap"
          >

            <div class="dot-spinner">

              <span></span>
              <span></span>
              <span></span>

            </div>

            Перевожу…

          </div>

          <div
            class="output-area empty"
            id="outputText"
          >
            Здесь появится перевод…
          </div>

          <div class="panel-footer">

            <button
              class="btn-copy"
              id="btnCopyDst"
            >
              Копировать
            </button>

          </div>

        </div>

      </div>
    </div>

    <section
      class="news-section"
      aria-label="О проекте"
    >

      <h2>
        О проекте
      </h2>

      <div
        class="news-grid"
        id="newsGrid"
      ></div>

    </section>

    <section
      class="download-section"
      aria-label="Скачать приложение"
    >

      <h2>Скачать приложение</h2>

      <a
        class="btn-download"
        href="https://github.com/cesslav/Polyglot_Mobile/releases/latest/download/polyglot.apk"
        download="polyglot.apk"
        rel="noopener"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 3v13M7 11l5 5 5-5" stroke="#000" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M4 20h16" stroke="#000" stroke-width="2.2" stroke-linecap="round"/>
        </svg>
        Скачать для Android (.apk)
      </a>

      <p class="download-desc">
        <strong>Полиглот Mobile</strong> — это переводчик, который работает
        <strong>полностью без доступа в интернет</strong>. Языковые модели хранятся
        прямо на устройстве, поэтому переводы выполняются мгновенно и приватно —
        ваши тексты никуда не отправляются. Идеально для путешествий, работы
        в офлайн-режиме и использования в зонах без покрытия сети.
      </p>

    </section>

  </main>

  <footer>

    Распространяется по лицензии

    <a
      href="https://www.gnu.org/licenses/agpl-3.0.html"
      target="_blank"
      rel="noopener"
    >
      AGPLv3
    </a>

    &nbsp;·&nbsp;

    <a
      href="https://github.com/cesslav/polyglot"
      target="_blank"
      rel="noopener"
    >
      GitHub репозиторий проекта
    </a>

  </footer>

  <script>

    const $ = id => document.getElementById(id);

    async function loadModels() {

      try {

        const res =
          await fetch('/api/models');

        const { models } =
          await res.json();

        const sel =
          $('modelSelect');

        sel.innerHTML = '';

        if (
          !models ||
          models.length === 0
        ) {

          sel.innerHTML =
            '<option value="">Нет моделей</option>';

          return;
        }

        models.forEach((name, i) => {

          const opt =
            document.createElement('option');

          opt.value = name;
          opt.textContent = name;

          if (i === 0) {
            opt.selected = true;
          }

          sel.appendChild(opt);

        });

      } catch (e) {

        console.error(
          'Не удалось загрузить список моделей:',
          e
        );

      }
    }

    let debounce = null;

    function showSpinner() {

      $('spinnerWrap')
        .classList.add('active');

      $('outputText')
        .classList.add('hidden');
    }

    function hideSpinner() {

      $('spinnerWrap')
        .classList.remove('active');

      $('outputText')
        .classList.remove('hidden');
    }

    async function translate(text) {

      const model =
        $('modelSelect').value;

      if (!model) return;

      showSpinner();

      try {

        const res =
          await fetch('/api/translate', {

            method: 'POST',

            headers: {
              'Content-Type': 'application/json'
            },

            body: JSON.stringify({
              text,
              model
            })

          });

        const {
          translation,
          error
        } = await res.json();

        const out =
          $('outputText');

        if (error) {

          out.textContent =
            '⚠ ' + error;

          out.classList.add('empty');

        } else {

          out.textContent =
            translation;

          out.classList.remove('empty');

        }

      } catch (e) {

        $('outputText').textContent =
          '⚠ Ошибка соединения';

      } finally {

        hideSpinner();

      }
    }

    $('inputText')
      .addEventListener('input', e => {

        const text =
          e.target.value;

        $('charCount').textContent =
          `${text.length} / 2000`;

        clearTimeout(debounce);

        if (
          text.trim().length === 0
        ) {

          $('outputText').textContent =
            'Здесь появится перевод…';

          $('outputText')
            .classList.add('empty');

          return;
        }

        debounce =
          setTimeout(() => {

            translate(text);

          }, 250);

      });

    const PLACEHOLDER_OUT =
      'Здесь появится перевод…';

    function fallbackCopy(text) {

      const ta =
        document.createElement('textarea');

      ta.value = text;

      ta.style.cssText =
        'position:fixed;top:-9999px;left:-9999px;opacity:0';

      document.body.appendChild(ta);

      ta.focus();
      ta.select();

      try {
        document.execCommand('copy');
      } catch (_) {}

      document.body.removeChild(ta);
    }

    async function copyText(text, btn) {

      if (
        !text ||
        !text.trim()
      ) {
        return;
      }

      if (
        navigator.clipboard &&
        window.isSecureContext
      ) {

        try {

          await navigator.clipboard
            .writeText(text);

        } catch (_) {

          fallbackCopy(text);

        }

      } else {

        fallbackCopy(text);

      }

      const orig =
        btn.textContent;

      btn.textContent =
        'Скопировано ✓';

      setTimeout(() => {

        btn.textContent = orig;

      }, 1800);
    }

    $('btnCopySrc')
      .addEventListener('click', () => {

        copyText(
          $('inputText').value,
          $('btnCopySrc')
        );

      });

    $('btnCopyDst')
      .addEventListener('click', () => {

        const out =
          $('outputText');

        const text =
          out.classList.contains('empty')
            ? ''
            : out.textContent;

        copyText(
          text === PLACEHOLDER_OUT
            ? ''
            : text,
          $('btnCopyDst')
        );

      });

    async function loadNews() {

      try {

        const res =
          await fetch('/api/news');

        const { news } =
          await res.json();

        const grid =
          $('newsGrid');

        news.forEach(item => {

          const card =
            document.createElement('article');

          card.className =
            'news-card';

          card.innerHTML = `
            <div class="date">${item.date || ''}</div>
            <h3>${item.title}</h3>
            <p>${item.body}</p>
          `;

          grid.appendChild(card);

        });

      } catch (e) {

        console.error(
          'Ошибка загрузки новостей:',
          e
        );

      }
    }

    loadModels();
    loadNews();

  </script>

</body>
</html>
"""


def np_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def list_model_dirs():
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(os.path.join(MODELS_DIR, d))
    )


class ONNXTransformer:
    def __init__(self, model_dir):
        providers = ["CPUExecutionProvider"]
        encoder_path = os.path.join(model_dir, "encoder.onnx")
        decoder_path = os.path.join(model_dir, "decoder.onnx")
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)

    def encode(self, src):
        return self.encoder.run(["memory"], {"src": src.astype(np.int64)})[0]

    def decode(self, tgt, memory):
        return self.decoder.run(
            ["logits"],
            {"tgt": tgt.astype(np.int64), "memory": memory.astype(np.float32)}
        )[0]


def get_model(name: str):
    if name not in _model_cache:
        model_dir = os.path.join(MODELS_DIR, name)
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        tok_path = tokenizer_dir if os.path.isdir(tokenizer_dir) else model_dir
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
        model = ONNXTransformer(model_dir)
        _model_cache[name] = (tokenizer, model)
    return _model_cache[name]


def beam_search_onnx(model, tokenizer, src, beam_size=4, max_len=128):
    np.log_softmax = lambda x, axis: np.log(np_softmax(x))
    bos, eos = 0, 1
    src_np = src.cpu().numpy() if hasattr(src, "cpu") else np.array(src)
    memory = model.encode(src_np)
    beams = [(np.array([[bos]], dtype=np.int64), 0.0)]

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1] == eos:
                new_beams.append((seq, score))
                continue
            logits = model.decode(seq, memory)
            log_probs = np.log_softmax(logits[:, -1, :], axis=-1)
            topk_idx = np.argsort(-log_probs, axis=-1)[0][:beam_size]
            topk_log_probs = log_probs[0][topk_idx]
            for k in range(beam_size):
                new_seq = np.concatenate([seq, [[topk_idx[k]]]], axis=1)
                new_beams.append((new_seq, score + float(topk_log_probs[k])))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if all(seq[0, -1] == eos for seq, _ in beams):
            break

    return tokenizer.decode(beams[0][0][0], skip_special_tokens=True)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/logo")
def logo():
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polylogo.png")
    if not os.path.isfile(logo_path):
        return "", 404
    return send_file(logo_path, mimetype="image/jpeg")


@app.route("/api/models")
def api_models():
    """Return list of available model directories."""
    return jsonify({"models": list_model_dirs()})


@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    model_name = data.get("model", "")

    if not text:
        return jsonify({"translation": ""})

    if not model_name:
        return jsonify({"error": "Модель не выбрана"}), 400

    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        return jsonify({"error": f"Модель «{model_name}» не найдена"}), 404

    try:
        tokenizer, model = get_model(model_name)
    except Exception as e:
        return jsonify({"error": f"Ошибка загрузки модели: {e}"}), 500


    src = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=512
    )["input_ids"]

    result = beam_search_onnx(model, tokenizer, src)
    return jsonify({"translation": result})


@app.route("/api/news")
def api_news():
    return jsonify({"news": NEWS})


@app.route("/favicon.ico")
def favicon():
    return send_from_directory("./", "polylogo.ico")


if __name__ == "__main__":
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")

    NEWS = [
        {
            "title": "О проекте «Полиглот»",
            "body": f"«Полиглот» — это open-source переводчик нового поколения, построенный как открытая платформа для совместной разработки. Проект изначально задуман не просто как инструмент перевода, а как экосистема, в которой пользователи и разработчики по всему миру могут участвовать в создании и улучшении языковых моделей, функционала и пользовательского опыта.<br/><br/>"
                    f"Подход «Полиглота» близок к философии таких инициатив, как Linux Foundation, где ценность создаётся не одной компанией, а сообществом. Именно такая модель позволила технологиям на базе Linux стать фундаментом значительной части современной цифровой инфраструктуры.<br/>"
        },
        {
            "title": "Уникальность проекта «Полиглот»",
            "body": f"В отличие от классических переводчиков, таких как Google Translate или DeepL, «Полиглот» предлагает принципиально иной подход.<br/>"
                    f"Открытость вместо «чёрного ящика»<br/>"
                    f"Коммерческие сервисы работают как закрытые системы: пользователь не влияет на алгоритмы и не понимает, как формируется результат.<br/>"
                    f"«Полиглот» — это открытая архитектура:<br/>"
                    f"&#9;• доступ к коду и логике работы <br/>"
                    f"&#9;• возможность доработки и модификации <br/>"
                    f"&#9;• участие в развитии качества перевода <br/>"
                    f"Пользователь становится не просто потребителем, а участником создания продукта."
        },
        {
            "title": "Настраиваемость вместо универсальности",
            "body": f"Массовые переводчики вынуждены быть «усреднёнными»."
                    "«Полиглот» позволяет:<br/>"
                    "&#9;• создавать специализированные языковые пакеты (технические, медицинские, юридические и др.) <br/>"
                    "&#9;• адаптировать перевод под конкретные задачи <br/>"
                    "&#9;• учитывать контекст, стиль и терминологию <br/>"
                    "Это делает его особенно ценным в профессиональной среде, где точность критична.<br/>"
        },
    ]

    MODELS_DIR = "./onnx_export/for_web"
    _model_cache: dict = {}
    _current_model_name: str | None = None

    app.run(host="0.0.0.0", port=9090)