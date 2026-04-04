# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")


import onnxruntime as ort
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer

app = Flask(__name__)
np.log_softmax = lambda x, axis: np.log(np_softmax(x))


class ONNXTransformer:
    def __init__(self, encoder_path, decoder_path, device="cpu"):
        providers = ["CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)

    def encode(self, src):
        inputs = {
            "src": src.astype(np.int64),
        }

        memory = self.encoder.run(["memory"], inputs)[0]
        return memory

    def decode(self, tgt, memory):
        inputs = {
            "tgt": tgt.astype(np.int64),
            "memory": memory.astype(np.float32),
        }

        logits = self.decoder.run(["logits"], inputs)[0]
        return logits


tokenizer = AutoTokenizer.from_pretrained("./tokenizer/mixed48k")

model = ONNXTransformer(
    encoder_path="onnx_export/encoder_int8.onnx",
    decoder_path="onnx_export/decoder_int8.onnx"
)


def beam_search_onnx(model, tokenizer, src, beam_size=4, max_len=128):
    bos, eos = 0, 1

    src_np = src.cpu().numpy()

    memory = model.encode(src_np)

    beams = [(np.array([[bos]], dtype=np.int64), 0.0)]

    for _ in range(max_len):
        new_beams = []

        for seq, score in beams:
            if seq[0, -1] == eos:
                new_beams.append((seq, score))
                continue

            logits = model.decode(seq, memory)

            next_token_logits = logits[:, -1, :]
            log_probs = np.log_softmax(next_token_logits, axis=-1)

            topk_idx = np.argsort(-log_probs, axis=-1)[0][:beam_size]
            topk_log_probs = log_probs[0][topk_idx]

            for k in range(beam_size):
                next_token = topk_idx[k]
                new_seq = np.concatenate([seq, [[next_token]]], axis=1)
                new_score = score + float(topk_log_probs[k])

                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if all(seq[0, -1] == eos for seq, _ in beams):
            break

    best_seq = beams[0][0]
    return tokenizer.decode(best_seq[0], skip_special_tokens=True)


def np_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Переводчик</title>
  <style>
    body {
      margin: 0;
      font-family: Arial;
      background: linear-gradient(135deg, #6441a5, #b43ebb);
      color: white;
    }
    .container {
      max-width: 2000px;
      margin: 60px auto;
      background: rgba(255,255,255,0.05);
      padding: 20px;
      border-radius: 20px;
      backdrop-filter: blur(10px);
    }
    h1 { text-align: center; }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: -50px;
    }
    textarea {
      width: 90%;
      height: 200px;
      border-radius: 15px;
      border: none;
      padding: 15px;
      font-size: 16px;
      resize: none;
    }
    textarea:focus { outline: none; }
    .output {
      background: #eee;
      color: black;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Полиглот</h1>
    <div class="grid">
      <textarea id="input" placeholder="Введите текст для перевода..."></textarea>
      <textarea id="output" class="output" readonly placeholder="Здесь появится переведённый текст."></textarea>
    </div>
  </div>

<script>
let timeout = null;

async function translate(text) {
  const res = await fetch('/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });

  const data = await res.json();
  document.getElementById('output').value = data.translation;
}

document.getElementById('input').addEventListener('input', (e) => {
  const text = e.target.value;

  clearTimeout(timeout);
  timeout = setTimeout(() => {
    if (text.trim().length > 0) {
      translate(text);
    } else {
      document.getElementById('output').value = '';
    }
  }, 150); // debounce
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json()
    text = data.get("text", "")

    src = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )["input_ids"]

    result = beam_search_onnx(model, tokenizer, src)

    return jsonify({"translation": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)