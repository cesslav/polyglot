# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
import json
import os
import zipfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel


app = FastAPI(
    title="Polyglot Model Server",
    description="Раздача ONNX-моделей для Android-приложения Polyglot Mobile",
    version="1.0.0",
)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "./onnx_export/models"))
REQUIRED_FILES = {"encoder.onnx", "decoder.onnx", "tokenizer/tokenizer.json", "model_config.json"}


class ModelInfo(BaseModel):
    name: str
    file: str
    size_mb: int
    input_language: str = ""
    output_language: str = ""
    bidirectional: bool = False


def zip_is_valid(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return REQUIRED_FILES.issubset(set(zf.namelist()))
    except zipfile.BadZipFile:
        return False


def read_zip_model_config(path: Path) -> dict:
    try:
        with zipfile.ZipFile(path) as zf:
            if "model_config.json" in zf.namelist():
                return json.loads(zf.read("model_config.json").decode("utf-8"))
    except Exception:
        pass
    return {}


def make_display_name(stem: str, cfg: dict) -> str:
    src = cfg.get("input_language", "").strip().upper()
    tgt = cfg.get("output_language", "").strip().upper()
    bidir = cfg.get("bidirectional", False)

    if src and tgt:
        arrow = "<->" if bidir else "->"
        return f"{src} {arrow} {tgt}"

    parts = stem.split("-")
    if len(parts) >= 2 and all(p.isalpha() and len(p) <= 3 for p in parts[:2]):
        arrow = "->"
        suffix = " " + " ".join(p.capitalize() for p in parts[2:]) if parts[2:] else ""
        return f"{parts[0].upper()} {arrow} {parts[1].upper()}{suffix}"

    return " ".join(p.capitalize() for p in parts)


@app.get("/models", response_model=list[ModelInfo], summary="Список доступных моделей")
def list_models():
    if not MODELS_DIR.is_dir():
        return []

    result = []
    for entry in sorted(MODELS_DIR.iterdir()):
        if entry.suffix.lower() != ".zip":
            continue
        if not zip_is_valid(entry):
            continue

        size_mb = max(1, round(entry.stat().st_size / 1_048_576))
        cfg = read_zip_model_config(entry)

        result.append(ModelInfo(
            name=make_display_name(entry.stem, cfg),
            file=entry.name,
            size_mb=size_mb,
            input_language=cfg.get("input_language", ""),
            output_language=cfg.get("output_language", ""),
            bidirectional=cfg.get("bidirectional", False),
        ))

    return result


@app.get("/models/{filename}", summary="Скачать архив модели")
def download_model(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Недопустимое имя файла")

    path = MODELS_DIR / filename
    if not path.exists() or path.suffix.lower() != ".zip":
        raise HTTPException(status_code=404, detail=f"Файл не найден: {filename}")

    return FileResponse(path=path, media_type="application/zip", filename=filename)


@app.get("/ping", summary="Проверить доступность сервера")
def ping():
    return {"answer": "available"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")
    import uvicorn

    if not MODELS_DIR.exists():
        print(f"[!] Папка моделей не найдена: {MODELS_DIR.resolve()}")
        print(" Создайте её и положите туда zip-архивы, затем перезапустите сервер.")

    uvicorn.run("model_dist_server:app", host="0.0.0.0", port=9100, reload=True)