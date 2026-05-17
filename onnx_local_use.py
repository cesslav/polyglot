# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
import sys
import os
import zipfile
import numpy as np
import onnxruntime as ort
from PyQt5 import QtWidgets, QtCore
from transformers import AutoTokenizer


def np_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class ONNXTransformer:
    def __init__(self, model_dir):
        self.encoder = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"))
        self.decoder = ort.InferenceSession(os.path.join(model_dir, "decoder.onnx"))

    def encode(self, src):
        return self.encoder.run(["memory"], {"src": src.astype(np.int64)})[0]

    def decode(self, tgt, memory):
        return self.decoder.run(
            ["logits"],
            {"tgt": tgt.astype(np.int64), "memory": memory.astype(np.float32)}
        )[0]


def extract_model(zip_path):
    name = os.path.splitext(os.path.basename(zip_path))[0]
    out_dir = os.path.join(EXTRACT_DIR, name)
    if not os.path.exists(out_dir):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(out_dir)
    return out_dir


def load_model(name):
    if name not in _model_cache:
        path = extract_model(os.path.join(MODELS_DIR, name))
        tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        model = ONNXTransformer(path)
        _model_cache[name] = (tok, model)
    return _model_cache[name]


def beam_search(model, tokenizer, src, beam_size=4, max_len=128):
    np.log_softmax = lambda x, axis: np.log(np_softmax(x))
    bos, eos = 0, 1
    memory = model.encode(src)
    beams = [(np.array([[bos]], dtype=np.int64), 0.0)]

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1] == eos:
                new_beams.append((seq, score))
                continue
            logits = model.decode(seq, memory)
            log_probs = np.log_softmax(logits[:, -1, :], axis=-1)
            topk = np.argsort(-log_probs)[0][:beam_size]
            for k in topk:
                new_seq = np.concatenate([seq, [[k]]], axis=1)
                new_beams.append((new_seq, score + float(log_probs[0][k])))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
    return tokenizer.decode(beams[0][0][0], skip_special_tokens=True)


class TranslatorApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Полиглот")
        self.resize(1000, 600)

        self.setStyleSheet("""
            QWidget {
                background-color: #000000;
                color: #ffffff;
                font-family: Consolas;
            }
            QComboBox, QTextEdit {
                background-color: #0d0d0d;
                border: 1.5px solid #5cb84b;
                border-radius: 12px;
                padding: 6px;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QHBoxLayout()
        self.model_select = QtWidgets.QComboBox()
        header.addStretch()
        header.addWidget(QtWidgets.QLabel("Модель:"))
        header.addWidget(self.model_select)
        layout.addLayout(header)

        panels = QtWidgets.QHBoxLayout()

        self.input_text = QtWidgets.QTextEdit()
        self.input_text.setPlaceholderText("Введите текст...")

        self.output_text = QtWidgets.QTextEdit()
        self.output_text.setReadOnly(True)

        panels.addWidget(self.input_text)
        panels.addWidget(self.output_text)

        layout.addLayout(panels)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(300)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.handle_translate)

        self.input_text.textChanged.connect(self.on_text_changed)
        self.model_select.currentIndexChanged.connect(self.handle_translate)

        self.load_models()

    def load_models(self):
        self.model_select.clear()
        self.models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]

        if not self.models:
            self.model_select.addItem("Нет моделей")
            return

        for m in self.models:
            self.model_select.addItem(os.path.splitext(m)[0])

    def on_text_changed(self):
        self.timer.start()

    def get_selected_model_file(self):
        idx = self.model_select.currentIndex()
        if idx < 0 or idx >= len(self.models):
            return None
        return self.models[idx]

    def handle_translate(self):
        text = self.input_text.toPlainText().strip()
        model_file = self.get_selected_model_file()

        if not text or not model_file:
            self.output_text.setPlainText("")
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        try:
            tok, model = load_model(model_file)
            src = tok(text, return_tensors="np", padding="max_length", max_length=128)["input_ids"]
            result = beam_search(model, tok, src)
            self.output_text.setPlainText(result)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()


if __name__ == '__main__':
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR)
    EXTRACT_DIR = os.path.join(BASE_DIR, "_extracted")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)


    _model_cache = {}
    app = QtWidgets.QApplication(sys.argv)
    win = TranslatorApp()
    win.show()
    sys.exit(app.exec_())
