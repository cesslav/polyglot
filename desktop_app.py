import os
import sys
import json
import time
import zipfile
import shutil
import requests
import numpy as np

from pathlib import Path
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QComboBox, QScrollArea, QFrame,
    QStackedWidget, QProgressBar, QLineEdit, QMessageBox, QSizePolicy
)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


COLOR_BG = "#000000"
COLOR_SURFACE = "#0d0d0d"
COLOR_BORDER = "#5cb84b"
COLOR_TEXT = "#ffffff"
COLOR_MUTED = "#888888"
COLOR_ACCENT = "#5cb84b"
COLOR_ACCENT_DIM = "#1F3A1F"
COLOR_ORANGE = "#FFA500"
COLOR_RED = "#b84848"
COLOR_SECONDARY = "#1F3A1F"

MAX_INPUT_CHARS = 2000
MAX_OUTPUT_LEN = 1024
TOKEN_LEN_STEPS = list(range(64, 513, 64))

CONTENT_MAX_WIDTH = 900
CONTENT_MAX_HEIGHT = 1000

DEFAULT_SERVER_URL = "http://igorpet.ru:9100"
MODELS_DIR = "./.models/"
MODELS_DIR = Path(MODELS_DIR)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = Path(__file__).resolve().parent


GLOBAL_STYLE = f"""
QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: "Consolas", "Courier New", monospace;
    font-size: 14px;
}}
QScrollBar:vertical {{
    background: {COLOR_BG};
    width: 8px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {COLOR_BORDER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {COLOR_BG};
    height: 8px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {COLOR_BORDER};
    border-radius: 4px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QComboBox {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1.5px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 32px;
}}
QComboBox:hover {{ border-color: {COLOR_ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{
    image: none;
    width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {COLOR_ACCENT};
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1.5px solid {COLOR_BORDER};
    selection-background-color: {COLOR_ACCENT_DIM};
    selection-color: {COLOR_TEXT};
    outline: none;
}}
QProgressBar {{
    background-color: #333333;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {COLOR_ACCENT};
    border-radius: 3px;
}}
"""


def accent_btn(text, small=False):
    btn = QPushButton(text)
    h = "36px" if small else "44px"
    fs = "12px" if small else "14px"
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {COLOR_ACCENT};
            color: #000000;
            border: none;
            border-radius: 8px;
            padding: 0 20px;
            min-height: {h};
            font-weight: bold;
            font-size: {fs};
        }}
        QPushButton:hover {{ background-color: #6ed45a; }}
        QPushButton:pressed {{ background-color: #4aa03a; }}
        QPushButton:disabled {{ background-color: #2a4a24; color: #555555; }}
    """)
    return btn


def outline_btn(text, color=COLOR_ACCENT, small=False):
    btn = QPushButton(text)
    h = "36px" if small else "44px"
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: transparent;
            color: {color};
            border: 1.5px solid {color};
            border-radius: 8px;
            padding: 0 16px;
            min-height: {h};
            font-weight: bold;
            font-size: 13px;
        }}
        QPushButton:hover {{ background-color: rgba(92,184,75,0.12); }}
        QPushButton:pressed {{ background-color: rgba(92,184,75,0.2); }}
        QPushButton:disabled {{ color: #555555; border-color: #555555; }}
    """)
    return btn


def delete_btn(text):
    btn = QPushButton(text)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: #8B0000;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 0 14px;
            min-height: 32px;
            font-weight: bold;
            font-size: 13px;
        }}
        QPushButton:hover {{ background-color: #aa0000; }}
        QPushButton:pressed {{ background-color: #6a0000; }}
    """)
    return btn


def nav_btn(text):
    btn = QPushButton(text)
    btn.setCheckable(True)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {COLOR_BG};
            color: {COLOR_TEXT};
            border: 1px solid {COLOR_BORDER};
            border-radius: 0;
            padding: 0;
            min-height: 48px;
            font-size: 12px;
            font-weight: bold;
        }}
        QPushButton:checked {{
            background-color: {COLOR_SECONDARY};
            color: {COLOR_ACCENT};
        }}
        QPushButton:hover:!checked {{ background-color: #111111; }}
    """)
    return btn


def neon_card():
    frame = QFrame()
    frame.setStyleSheet(f"""
        QFrame {{
            background-color: {COLOR_SURFACE};
            border: 1.5px solid {COLOR_BORDER};
            border-radius: 10px;
        }}
    """)
    return frame


class RoundedPanel(QFrame):

    STYLE = f"""
        RoundedPanel {{
            background-color: {COLOR_SURFACE};
            border: 1.5px solid {COLOR_BORDER};
            border-radius: 10px;
        }}
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(self.STYLE)
        self.setAttribute(Qt.WA_StyledBackground, True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        from PyQt5.QtGui import QPainterPath, QRegion
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 10, 10)
        region = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)


def text_edit_transparent():
    te = QTextEdit()
    te.setFrameShape(QFrame.NoFrame)
    te.setStyleSheet(f"""
        QTextEdit {{
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 14px;
            font-size: 15px;
            color: {COLOR_TEXT};
            selection-background-color: rgba(92,184,75,0.3);
        }}
    """)
    return te


class UnigramTokenizer:
    MAX_TOKEN_LEN = 32

    def __init__(self, model_dir: Path):
        tok_file = model_dir / "tokenizer" / "tokenizer.json"
        if not tok_file.exists():
            raise FileNotFoundError(f"tokenizer.json not found: {tok_file}")

        data = json.loads(tok_file.read_text(encoding="utf-8"))

        self.bos_id = 0
        self.eos_id = 1
        self.unk_id = 2
        self.pad_id = 3

        for tok in data.get("added_tokens", []):
            tid, content = tok["id"], tok.get("content", "")
            if "bos" in content.lower(): self.bos_id = tid
            elif "eos" in content.lower(): self.eos_id = tid
            elif "unk" in content.lower(): self.unk_id = tid
            elif "pad" in content.lower(): self.pad_id = tid

        vocab_arr = data["model"]["vocab"]
        self._id_to_token = [entry[0] for entry in vocab_arr]
        self._vocab_score = {entry[0]: float(entry[1]) for entry in vocab_arr}
        self._token_to_id = {entry[0]: i for i, entry in enumerate(vocab_arr)}

    def _viterbi(self, text: str) -> list:
        n = len(text)
        if n == 0:
            return []
        NEG_INF = float("-inf")
        dp_score = [NEG_INF] * (n + 1)
        dp_from = [-1] * (n + 1)
        dp_score[0] = 0.0
        for end in range(1, n + 1):
            for start in range(max(0, end - self.MAX_TOKEN_LEN), end):
                if dp_score[start] == NEG_INF:
                    continue
                sub = text[start:end]
                score = self._vocab_score.get(sub)
                if score is None:
                    continue
                total = dp_score[start] + score
                if total > dp_score[end]:
                    dp_score[end] = total
                    dp_from[end] = start
        if dp_score[n] == NEG_INF:
            return list(text)
        result, pos = [], n
        while pos > 0:
            start = dp_from[pos]
            result.append(text[start:pos])
            pos = start
        result.reverse()
        return result

    def _pretokenize(self, text: str) -> list:
        if not text:
            return []
        raw = "▁" + text.replace(" ", "▁")
        parts = raw.split("▁")
        return ["▁" + p for p in parts if p]

    def encode(self, text: str, max_length: int = 256) -> np.ndarray:
        tokens = [self.bos_id]
        for piece in self._pretokenize(text):
            for sub in self._viterbi(piece):
                tokens.append(self._token_to_id.get(sub, self.unk_id))
        tokens.append(self.eos_id)
        out = np.full(max_length, self.pad_id, dtype=np.int64)
        for i, t in enumerate(tokens[:max_length]):
            out[i] = t
        return out

    def decode(self, ids, skip_special: bool = True) -> str:
        skip = {self.bos_id, self.eos_id, self.pad_id} if skip_special else set()
        sb = []
        for tid in ids:
            tid = int(tid)
            if tid in skip or tid < 0 or tid >= len(self._id_to_token):
                continue
            sb.append(self._id_to_token[tid])
        return "".join(sb).replace("▁", " ").strip()


class OnnxTransformer:
    def __init__(self, model_dir: Path):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime не установлен. pip install onnxruntime")
        providers = ["CPUExecutionProvider"]
        enc = str(model_dir / "encoder.onnx")
        dec = str(model_dir / "decoder.onnx")
        if not os.path.exists(enc): raise FileNotFoundError(enc)
        if not os.path.exists(dec): raise FileNotFoundError(dec)
        self.encoder = ort.InferenceSession(enc, providers=providers)
        self.decoder = ort.InferenceSession(dec, providers=providers)

    def encode(self, src: np.ndarray) -> np.ndarray:
        return self.encoder.run(["memory"], {"src": src.astype(np.int64)})[0]

    def decode(self, tgt: np.ndarray, memory: np.ndarray) -> np.ndarray:
        return self.decoder.run(
            ["logits"],
            {"tgt": tgt.astype(np.int64), "memory": memory.astype(np.float32)},
        )[0]


def _np_log_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return np.log(e / np.sum(e, axis=-1, keepdims=True) + 1e-12)


def greedy_search(model, tokenizer, src_tokens, max_len=MAX_OUTPUT_LEN, on_token=None):
    bos, eos = tokenizer.bos_id, tokenizer.eos_id
    memory = model.encode(src_tokens[np.newaxis, :])
    tokens = [bos]
    for _ in range(max_len - 1):
        tgt = np.array([tokens], dtype=np.int64)
        logits = model.decode(tgt, memory)
        next_id = int(np.argmax(logits[0, -1, :]))
        tokens.append(next_id)
        if on_token:
            on_token(np.array(tokens, dtype=np.int64))
        if next_id == eos:
            break
    return np.array(tokens, dtype=np.int64)


def beam_search(model, tokenizer, src_tokens, beam_size=4, max_len=MAX_OUTPUT_LEN, on_token=None):
    bos, eos = tokenizer.bos_id, tokenizer.eos_id
    memory = model.encode(src_tokens[np.newaxis, :])
    beams = [(np.array([[bos]], dtype=np.int64), 0.0)]
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1] == eos:
                new_beams.append((seq, score)); continue
            logits = model.decode(seq, memory)
            log_probs = _np_log_softmax(logits[:, -1, :])
            topk = np.argsort(-log_probs, axis=-1)[0][:beam_size]
            for k in topk:
                new_beams.append((np.concatenate([seq, [[k]]], axis=1), score + float(log_probs[0, k])))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if on_token:
            on_token(beams[0][0][0])
        if all(s[0, -1] == eos for s, _ in beams):
            break
    return beams[0][0][0]


class ModelDownloadManager:
    base_url = DEFAULT_SERVER_URL

    @classmethod
    def ping(cls, url=None) -> bool:
        try:
            r = requests.get(f"{url or cls.base_url}/ping", timeout=10)
            return r.status_code == 200 and r.json().get("answer") == "available"
        except Exception:
            return False

    @classmethod
    def fetch_model_list(cls) -> list:
        r = requests.get(f"{cls.base_url}/models", timeout=15)
        r.raise_for_status()
        return r.json()

    @classmethod
    def download_model(cls, file: str, dest_dir: Path, on_progress=None):
        r = requests.get(f"{cls.base_url}/models/{file}", stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        zip_path = dest_dir / file
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress and total:
                        on_progress(int(downloaded * 100 / total), False)
        if on_progress:
            on_progress(100, True)
        stem = file.removesuffix(".zip")
        model_dir = dest_dir / stem
        model_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(model_dir)
        zip_path.unlink()
        return model_dir


class InferenceWorker(QThread):
    partial_result = pyqtSignal(str)
    finished = pyqtSignal(str, float, int)
    error = pyqtSignal(str)

    def __init__(self, model, tokenizer, text):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.text = text

    def run(self):
        try:
            tok = self.tokenizer
            mdl = self.model
            t0 = time.perf_counter()
            raw = tok.encode(self.text, TOKEN_LEN_STEPS[-1])
            nz = np.where(raw != tok.pad_id)[0]
            actual = int(nz[-1]) + 1 if len(nz) else 1
            src_len = next((n for n in TOKEN_LEN_STEPS if n >= actual), TOKEN_LEN_STEPS[-1])
            src = raw[:src_len]

            token_count = [0]

            def on_tok(tokens):
                token_count[0] += 1
                self.partial_result.emit(tok.decode(tokens))

            out = greedy_search(mdl, tok, src, max_len=MAX_OUTPUT_LEN, on_token=on_tok)
            elapsed = time.perf_counter() - t0
            self.finished.emit(tok.decode(out), elapsed, token_count[0])
        except Exception as e:
            self.error.emit(str(e))


class DownloadWorker(QThread):
    progress = pyqtSignal(str, int, bool)
    finished = pyqtSignal(str)
    error = pyqtSignal(str, str)

    def __init__(self, file: str, dest_dir: Path):
        super().__init__()
        self.file = file
        self.dest_dir = dest_dir

    def run(self):
        try:
            ModelDownloadManager.download_model(
                self.file, self.dest_dir,
                lambda p, ins: self.progress.emit(self.file, p, ins)
            )
            self.finished.emit(self.file)
        except Exception as e:
            self.error.emit(self.file, str(e))


class FetchListWorker(QThread):
    result = pyqtSignal(list)
    error = pyqtSignal(str)

    def run(self):
        try:
            self.result.emit(ModelDownloadManager.fetch_model_list())
        except Exception as e:
            self.error.emit(str(e))


class PingWorker(QThread):
    result = pyqtSignal(bool, str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def run(self):
        self.result.emit(ModelDownloadManager.ping(self.url), self.url)


class ModelCard(QFrame):
    download_clicked = pyqtSignal(dict)
    delete_clicked = pyqtSignal(dict)

    CARD_STYLE = f"""
        QFrame {{
            background-color: {COLOR_SURFACE};
            border: 1.5px solid {COLOR_BORDER};
            border-radius: 8px;
        }}
        QLabel {{ border: none; background: transparent; }}
    """

    def __init__(self, model_info: dict, installed: bool, parent=None):
        super().__init__(parent)
        self.model_info = model_info
        self.installed = installed
        self.setStyleSheet(self.CARD_STYLE)
        self._build()

    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        name_lbl = QLabel(self.model_info.get("name", "—"))
        name_lbl.setStyleSheet(f"color:{COLOR_TEXT}; font-weight:bold; font-size:13px;")
        size_lbl = QLabel(f"{self.model_info.get('size_mb','?')} МБ")
        size_lbl.setStyleSheet(f"color:{COLOR_MUTED}; font-size:11px;")
        size_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        info_row = QHBoxLayout()
        info_row.setSpacing(6)
        info_row.addWidget(name_lbl, 1)
        info_row.addWidget(size_lbl)

        left_col = QVBoxLayout()
        left_col.setSpacing(0)
        left_col.addLayout(info_row)

        self.pb = QProgressBar()
        self.pb.setFixedHeight(5)
        self.pb.setTextVisible(False)
        self.pb.setVisible(False)

        self.pb_lbl = QLabel("0%")
        self.pb_lbl.setVisible(False)
        self.pb_lbl.setStyleSheet(f"color:{COLOR_ACCENT}; font-size:11px;")
        self.pb_lbl.setFixedWidth(70)
        self.pb_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        pb_row = QHBoxLayout()
        pb_row.setSpacing(6)
        pb_row.addWidget(self.pb, 1)
        pb_row.addWidget(self.pb_lbl)
        left_col.addLayout(pb_row)

        layout.addLayout(left_col, 1)

        self.dl_btn = accent_btn("Скачать", small=True)
        self.del_btn = delete_btn("Удалить")
        self.dl_btn.setFixedWidth(90)
        self.del_btn.setFixedWidth(90)

        self.dl_btn.clicked.connect(lambda: self.download_clicked.emit(self.model_info))
        self.del_btn.clicked.connect(lambda: self.delete_clicked.emit(self.model_info))

        layout.addWidget(self.dl_btn)
        layout.addWidget(self.del_btn)

        self._refresh()

    def _refresh(self):
        if self.installed:
            self.dl_btn.setVisible(False)
            self.del_btn.setVisible(True)
        else:
            self.dl_btn.setVisible(True)
            self.del_btn.setVisible(False)
        self.pb.setVisible(False)
        self.pb_lbl.setVisible(False)

    def set_progress(self, pct: int, installing: bool):
        self.dl_btn.setVisible(False)
        self.del_btn.setVisible(False)
        self.pb.setVisible(True)
        self.pb_lbl.setVisible(True)
        if installing:
            self.pb.setRange(0, 0)
            self.pb_lbl.setText("Установка…")
        else:
            self.pb.setRange(0, 100)
            self.pb.setValue(pct)
            self.pb_lbl.setText(f"{pct}%")

    def mark_installed(self):
        self.installed = True
        self.pb.setRange(0, 100)
        self.pb.setValue(100)
        self._refresh()

    def mark_deleted(self):
        self.installed = False
        self._refresh()


class TranslateScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.model: OnnxTransformer | None = None
        self.tokenizer: UnigramTokenizer | None = None
        self.worker: InferenceWorker | None = None
        self._model_dirs: list = []
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 12)
        root.setSpacing(10)

        panels_row = QHBoxLayout()
        panels_row.setSpacing(12)

        in_panel = RoundedPanel()
        in_layout = QVBoxLayout(in_panel)
        in_layout.setContentsMargins(0, 0, 0, 0)
        in_layout.setSpacing(0)

        in_header = QWidget()
        in_header.setStyleSheet("background: transparent; border-bottom: 1px solid rgba(92,184,75,0.2);")
        ih_row = QHBoxLayout(in_header)
        ih_row.setContentsMargins(14, 8, 14, 8)
        in_lbl = QLabel("Исходный текст")
        in_lbl.setStyleSheet(f"color:{COLOR_MUTED}; font-size:11px; border:none; background:transparent;")
        self.char_count_lbl = QLabel("0 / 2000")
        self.char_count_lbl.setStyleSheet(f"color:{COLOR_MUTED}; font-size:11px; border:none; background:transparent;")
        ih_row.addWidget(in_lbl)
        ih_row.addStretch()
        ih_row.addWidget(self.char_count_lbl)
        in_layout.addWidget(in_header)

        self.input_edit = text_edit_transparent()
        self.input_edit.setPlaceholderText("Введите текст для перевода…")
        self.input_edit.textChanged.connect(self._on_input_changed)
        in_layout.addWidget(self.input_edit, 1)

        in_footer = QWidget()
        in_footer.setStyleSheet("background: transparent; border-top: 1px solid rgba(92,184,75,0.1);")
        if_row = QHBoxLayout(in_footer)
        if_row.setContentsMargins(14, 6, 14, 6)
        copy_src = outline_btn("Копировать", small=True)
        copy_src.clicked.connect(lambda: self._copy(self.input_edit.toPlainText()))
        if_row.addStretch()
        if_row.addWidget(copy_src)
        in_layout.addWidget(in_footer)

        out_panel = RoundedPanel()
        out_layout = QVBoxLayout(out_panel)
        out_layout.setContentsMargins(0, 0, 0, 0)
        out_layout.setSpacing(0)

        out_header = QWidget()
        out_header.setStyleSheet("background: transparent; border-bottom: 1px solid rgba(92,184,75,0.2);")
        oh_row = QHBoxLayout(out_header)
        oh_row.setContentsMargins(14, 8, 14, 8)
        out_lbl = QLabel("Перевод")
        out_lbl.setStyleSheet(f"color:{COLOR_MUTED}; font-size:11px; border:none; background:transparent;")
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color:{COLOR_MUTED}; font-size:11px; border:none; background:transparent;")
        oh_row.addWidget(out_lbl)
        oh_row.addStretch()
        oh_row.addWidget(self.status_lbl)
        out_layout.addWidget(out_header)

        self.output_edit = text_edit_transparent()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("Здесь появится перевод…")
        out_layout.addWidget(self.output_edit, 1)

        out_footer = QWidget()
        out_footer.setStyleSheet("background: transparent; border-top: 1px solid rgba(92,184,75,0.1);")
        of_row = QHBoxLayout(out_footer)
        of_row.setContentsMargins(14, 6, 14, 6)
        copy_dst = outline_btn("Копировать", small=True)
        copy_dst.clicked.connect(lambda: self._copy(self.output_edit.toPlainText()))
        of_row.addStretch()
        of_row.addWidget(copy_dst)
        out_layout.addWidget(out_footer)

        panels_row.addWidget(in_panel, 1)
        panels_row.addWidget(out_panel, 1)
        root.addLayout(panels_row, 1)

        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(12)

        self.translate_btn = accent_btn("Перевести")
        self.translate_btn.setMinimumWidth(160)
        self.translate_btn.setEnabled(False)
        self.translate_btn.clicked.connect(self._do_translate)
        bottom_bar.addWidget(self.translate_btn)

        bottom_bar.addStretch()

        model_lbl = QLabel("Языковой пакет:")
        model_lbl.setStyleSheet(f"color:{COLOR_MUTED}; font-size:12px;")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.setMaximumWidth(340)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        bottom_bar.addWidget(model_lbl)
        bottom_bar.addWidget(self.model_combo)

        root.addLayout(bottom_bar)

        self.refresh_models()


    def refresh_models(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self._model_dirs = []

        if MODELS_DIR.exists():
            for d in sorted(MODELS_DIR.iterdir()):
                if d.is_dir() and self._is_model_dir(d):
                    self._model_dirs.append(d)
                    self.model_combo.addItem(self._display(d.name))

        self.model_combo.blockSignals(False)

        if self._model_dirs:
            self._load_model(self._model_dirs[0])
        else:
            self.translate_btn.setEnabled(False)
            self.output_edit.setPlainText(
                "Языковой пакет не установлен.\nПерейдите в «Загрузки»."
            )

    @staticmethod
    def _is_model_dir(d: Path) -> bool:
        return ((d / "encoder.onnx").exists() and
                (d / "decoder.onnx").exists() and
                (d / "tokenizer" / "tokenizer.json").exists())

    @staticmethod
    def _display(stem: str) -> str:
        parts = stem.split("-")
        if (len(parts) >= 2 and
                all(p.isalpha() and len(p) <= 3 for p in parts[:2])):
            arrow = f"{parts[0].upper()} -> {parts[1].upper()}"
            suffix = " ".join(p.capitalize() for p in parts[2:])
            return f"{arrow} {suffix}".strip()
        return " ".join(p.capitalize() for p in parts)

    def _on_model_changed(self, idx: int):
        if 0 <= idx < len(self._model_dirs):
            self._load_model(self._model_dirs[idx])

    def _load_model(self, path: Path):
        self.translate_btn.setEnabled(False)
        self.status_lbl.setText("Загрузка…")
        try:
            self.tokenizer = UnigramTokenizer(path)
            self.model = OnnxTransformer(path)
            self.translate_btn.setEnabled(True)
            self.status_lbl.setText("")
        except Exception as e:
            self.status_lbl.setText(f"Ошибка: {e}")

    def _on_input_changed(self):
        text = self.input_edit.toPlainText()
        if len(text) > MAX_INPUT_CHARS:
            cursor = self.input_edit.textCursor()
            pos = cursor.position()
            self.input_edit.blockSignals(True)
            self.input_edit.setPlainText(text[:MAX_INPUT_CHARS])
            self.input_edit.blockSignals(False)
            cursor.setPosition(min(pos, MAX_INPUT_CHARS))
            self.input_edit.setTextCursor(cursor)
            text = text[:MAX_INPUT_CHARS]

        n = len(text)
        ratio = n / MAX_INPUT_CHARS
        color = COLOR_RED if ratio >= 1.0 else (COLOR_ORANGE if ratio >= 0.8 else COLOR_MUTED)
        self.char_count_lbl.setText(f"{n} / {MAX_INPUT_CHARS}")
        self.char_count_lbl.setStyleSheet(
            f"color:{color}; font-size:11px; border:none; background:transparent;"
        )
        if not text.strip():
            self.output_edit.clear()

    def _do_translate(self):
        if not self.model or not self.tokenizer:
            return
        text = self.input_edit.toPlainText().strip()
        if not text or (self.worker and self.worker.isRunning()):
            return

        self.translate_btn.setEnabled(False)
        self.status_lbl.setText("Перевод…")
        self.output_edit.clear()

        self.worker = InferenceWorker(self.model, self.tokenizer, text)
        self.worker.partial_result.connect(self.output_edit.setPlainText)
        self.status_lbl.setText("Готово!")
        self.translate_btn.setEnabled(True)
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _on_err(self, msg: str):
        self.output_edit.setPlainText(f"⚠ Ошибка: {msg}")
        self.status_lbl.setText("")
        self.translate_btn.setEnabled(True)

    @staticmethod
    def _copy(text: str):
        if text:
            QApplication.clipboard().setText(text)


class DownloadsScreen(QWidget):
    models_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._cards: dict = {}
        self._workers: dict = {}
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        self.conn_card = neon_card()
        cl = QHBoxLayout(self.conn_card)
        cl.setContentsMargins(14, 10, 14, 10)
        self.conn_lbl = QLabel("Подключение к серверу…")
        self.conn_lbl.setStyleSheet(f"color:{COLOR_ACCENT}; border:none;")
        cl.addWidget(self.conn_lbl)
        root.addWidget(self.conn_card)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.list_widget = QWidget()
        self.list_widget.setStyleSheet("background: transparent;")
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(5)
        self.list_layout.addStretch()
        scroll.setWidget(self.list_widget)
        root.addWidget(scroll, 1)

        sv_card = neon_card()
        sv_layout = QVBoxLayout(sv_card)
        sv_layout.setContentsMargins(14, 12, 14, 12)
        sv_layout.setSpacing(8)

        title = QLabel("Адрес сервера загрузок")
        title.setStyleSheet(f"color:{COLOR_ACCENT}; font-weight:bold; font-size:14px; border:none;")
        sv_layout.addWidget(title)

        url_row = QHBoxLayout()
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText(DEFAULT_SERVER_URL)
        self.url_edit.setStyleSheet(f"""
            QLineEdit {{
                background:{COLOR_BG}; color:{COLOR_TEXT};
                border:1.5px solid {COLOR_BORDER}; border-radius:8px;
                padding:6px 10px; min-height:30px;
            }}
        """)
        confirm_btn = accent_btn("✓", small=True)
        confirm_btn.setFixedWidth(60)
        confirm_btn.clicked.connect(self._on_confirm_url)
        url_row.addWidget(self.url_edit)
        url_row.addWidget(confirm_btn)
        sv_layout.addLayout(url_row)

        hint = QLabel("Адрес будет проверен перед сохранением.")
        hint.setStyleSheet(f"color:{COLOR_MUTED}; font-size:12px; border:none;")
        sv_layout.addWidget(hint)

        reset_btn = outline_btn("Сбросить адрес сервера", small=True)
        reset_btn.clicked.connect(self._on_reset_url)
        sv_layout.addWidget(reset_btn)

        root.addWidget(sv_card)

    def load(self):
        self.conn_lbl.setText("Подключение к серверу…")
        self.conn_card.setStyleSheet(neon_card().styleSheet())
        self.conn_lbl.setStyleSheet(f"color:{COLOR_ACCENT}; border:none;")
        self.conn_card.setVisible(True)
        self._clear_list()

        installed = self._installed_stems()
        for stem in installed:
            d = MODELS_DIR / stem
            size_mb = max(1, sum(
                f.stat().st_size for f in d.rglob("*") if f.is_file()
            ) // (1024 * 1024))
            info = {"name": TranslateScreen._display(stem),
                    "file": f"{stem}.zip", "size_mb": size_mb}
            self._add_card(info, installed=True)

        worker = FetchListWorker()
        worker.result.connect(self._on_server_list)
        worker.error.connect(self._on_server_error)
        worker.setParent(self)
        self._fetch_worker = worker
        worker.start()

    def _installed_stems(self) -> list:
        if not MODELS_DIR.exists():
            return []
        return [d.name for d in sorted(MODELS_DIR.iterdir())
                if d.is_dir() and (d / "encoder.onnx").exists()]

    def _on_server_list(self, models: list):
        self.conn_card.setVisible(False)
        installed = set(self._installed_stems())
        for m in models:
            stem = m["file"].removesuffix(".zip")
            if stem not in installed:
                self._add_card(m, installed=False)

    def _on_server_error(self, msg: str):
        self.conn_lbl.setText("Ошибка подключения. Проверьте интернет или адрес сервера.")
        self.conn_card.setStyleSheet(f"""
            QFrame {{
                background-color:#1A0A00;
                border:1.5px solid {COLOR_ORANGE};
                border-radius:10px;
            }}
        """)
        self.conn_lbl.setStyleSheet(f"color:{COLOR_ORANGE}; border:none;")

    def _clear_list(self):
        self._cards.clear()
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _add_card(self, info: dict, installed: bool):
        card = ModelCard(info, installed)
        card.download_clicked.connect(self._start_download)
        card.delete_clicked.connect(self._delete_model)
        self._cards[info["file"]] = card
        self.list_layout.insertWidget(self.list_layout.count() - 1, card)

    def _start_download(self, info: dict):
        file = info["file"]
        if file in self._workers:
            return
        card = self._cards.get(file)
        if card:
            card.set_progress(0, False)
        worker = DownloadWorker(file, MODELS_DIR)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_done)
        worker.error.connect(self._on_dl_error)
        self._workers[file] = worker
        worker.start()

    def _on_progress(self, file, pct, installing):
        if card := self._cards.get(file):
            card.set_progress(pct, installing)

    def _on_done(self, file: str):
        self._workers.pop(file, None)
        if card := self._cards.get(file):
            card.mark_installed()
        self.models_changed.emit()

    def _on_dl_error(self, file: str, msg: str):
        self._workers.pop(file, None)
        if card := self._cards.get(file):
            card.mark_deleted()
        QMessageBox.warning(self, "Ошибка загрузки", f"Не удалось загрузить {file}:\n{msg}")

    def _delete_model(self, info: dict):
        stem = info["file"].removesuffix(".zip")
        if QMessageBox.question(
            self, "Удалить", f"Удалить «{info['name']}»?",
            QMessageBox.Yes | QMessageBox.No
        ) != QMessageBox.Yes:
            return
        d = MODELS_DIR / stem
        if d.exists():
            shutil.rmtree(d)
        if card := self._cards.get(info["file"]):
            card.mark_deleted()
        self.models_changed.emit()

    def _on_confirm_url(self):
        raw = self.url_edit.text().strip().rstrip("/")
        if not raw:
            return
        if not raw.startswith("http"):
            raw = f"http://{raw}"
        self.conn_lbl.setText("Проверка адреса…")
        self.conn_card.setVisible(True)
        w = PingWorker(raw)
        w.result.connect(self._on_ping)
        w.setParent(self)
        self._ping_worker = w
        w.start()

    def _on_ping(self, ok: bool, url: str):
        if ok:
            ModelDownloadManager.base_url = url
            self.url_edit.clear()
            self.url_edit.setPlaceholderText(url)
            QMessageBox.information(self, "Готово", "Адрес сервера обновлён.")
            self.load()
        else:
            QMessageBox.warning(self, "Недоступен", f"Сервер недоступен:\n{url}")
            self.conn_card.setVisible(False)

    def _on_reset_url(self):
        ModelDownloadManager.base_url = DEFAULT_SERVER_URL
        self.url_edit.clear()
        self.url_edit.setPlaceholderText(DEFAULT_SERVER_URL)
        QMessageBox.information(self, "Готово", "Адрес сервера сброшен.")


class AboutScreen(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        desc_card = neon_card()
        dc = QVBoxLayout(desc_card)
        dc.setContentsMargins(14, 14, 14, 14)
        desc = QLabel(
            '"Полиглот" — это open-source платформа для перевода, которая превращает пользователей в соавторов '
            'продукта. В отличие от закрытых решений, таких как Google Translate, проект позволяет создавать и '
            'адаптировать перевод под конкретные задачи, формируя экосистему языковых решений. За счёт '
            'open-source модели "Полиглот" имеет потенциал масштабироваться через сообщество и стать '
            'инфраструктурным стандартом в области перевода.'
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color:{COLOR_TEXT}; font-size:14px; border:none;")
        dc.addWidget(desc)
        layout.addWidget(desc_card)

        def link_section(title: str, links: list):
            card = neon_card()
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 12, 14, 12)
            cl.setSpacing(8)
            t = QLabel(title)
            t.setStyleSheet(f"color:{COLOR_ACCENT}; font-weight:bold; font-size:13px; border:none;")
            cl.addWidget(t)
            for lbl, url in links:
                btn = accent_btn(lbl)
                btn.clicked.connect(lambda _, u=url: __import__("webbrowser").open(u))
                cl.addWidget(btn)
            return card

        layout.addWidget(link_section("Для пользователей", [
            ("Сайт проекта", "http://igorpet.ru:9090"),
        ]))
        layout.addWidget(link_section("Для разработчиков", [
            ("Репозиторий приложения", "https://github.com/cesslav/Polyglot_Mobile"),
            ("Инструментарий разработки", "https://github.com/cesslav/polyglot"),
        ]))

        lic = QLabel(
            "Распространяется по лицензии "
            f"<a href='https://www.gnu.org/licenses/agpl-3.0.html' style='color:{COLOR_ACCENT}'>AGPLv3</a>"
        )
        lic.setOpenExternalLinks(True)
        lic.setAlignment(Qt.AlignCenter)
        lic.setStyleSheet(f"color:{COLOR_MUTED}; font-size:12px; border:none;")
        layout.addWidget(lic)

        layout.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll)


class Header(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(108)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: rgba(0,0,0,210);
                border-bottom: 1px solid rgba(92,184,75,0.25);
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 6, 18, 6)

        logo_path = SCRIPT_DIR / "polylogo.png"
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            pix = pix.scaledToHeight(90, Qt.SmoothTransformation)
            logo_lbl = QLabel()
            logo_lbl.setPixmap(pix)
            logo_lbl.setStyleSheet("border: none; background: transparent;")
        else:
            logo_lbl = QLabel("Полиглот")
            logo_lbl.setStyleSheet(
                f"font-size:26px; font-weight:bold; color:{COLOR_ACCENT};"
                "border:none; background:transparent;"
            )

        layout.addWidget(logo_lbl)
        layout.addStretch()


class NavBar(QWidget):
    tab_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(50)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._btns = []
        for i, lbl in enumerate(["Перевод", "Загрузки", "О проекте"]):
            btn = nav_btn(lbl)
            btn.clicked.connect(lambda _, idx=i: self._click(idx))
            layout.addWidget(btn)
            self._btns.append(btn)
        self._btns[0].setChecked(True)

    def _click(self, idx: int):
        for i, btn in enumerate(self._btns):
            btn.setChecked(i == idx)
        self.tab_changed.emit(idx)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Полиглот")
        self.setMinimumSize(700, 550)
        self.setMaximumWidth(CONTENT_MAX_WIDTH)
        self.setMaximumHeight(900)
        self.resize(1000, 680)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = Header()
        root.addWidget(self.header)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        self.translate_screen = TranslateScreen()
        self.downloads_screen = DownloadsScreen()
        self.about_screen = AboutScreen()

        self.stack.addWidget(self.translate_screen)
        self.stack.addWidget(self.downloads_screen)
        self.stack.addWidget(self.about_screen)

        self.nav = NavBar()
        self.nav.tab_changed.connect(self._on_tab)
        root.addWidget(self.nav)

        self.downloads_screen.models_changed.connect(self.translate_screen.refresh_models)

    def _on_tab(self, idx: int):
        self.stack.setCurrentIndex(idx)
        if idx == 1:
            self.downloads_screen.load()


def main():
    print("This file is distributed under the open license AGPLv3, "
          "source code: https://github.com/cesslav/polyglot.")
    if not ONNX_AVAILABLE:
        print("ВНИМАНИЕ: onnxruntime не найден — перевод недоступен.")
        print("Установите: pip install onnxruntime")

    app = QApplication(sys.argv)
    app.setApplicationName("Полиглот")
    app.setStyle("Fusion")
    app.setStyleSheet(GLOBAL_STYLE)

    palette = QPalette()
    palette.setColor(QPalette.Window,     QColor(COLOR_BG))
    palette.setColor(QPalette.WindowText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Base,       QColor(COLOR_SURFACE))
    palette.setColor(QPalette.Text,       QColor(COLOR_TEXT))
    palette.setColor(QPalette.Button,     QColor(COLOR_SURFACE))
    palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Highlight, QColor(COLOR_ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#000000"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()