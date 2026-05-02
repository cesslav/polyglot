import os
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

MODELS_DIR = Path(os.getenv("MODELS_DIR", "./onnx_export/models"))

REQUIRED_FILES = {"encoder.onnx", "decoder.onnx", "tokenizer/tokenizer.json"}


app = FastAPI(
    title="Polyglot Model Server",
    description="Раздача ONNX-моделей для Android-приложения Polyglot Mobile",
    version="1.0.0",
)


class ModelInfo(BaseModel):
    name:    str
    file:    str
    size_mb: int


def zip_is_valid(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            return REQUIRED_FILES.issubset(names)
    except zipfile.BadZipFile:
        return False


def make_display_name(stem: str) -> str:
    parts = stem.split("-")

    if len(parts) >= 2 and all(p.isalpha() and len(p) <= 3 for p in parts[:2]):
        src, tgt = parts[0].upper(), parts[1].upper()
        suffix   = " " + " ".join(p.capitalize() for p in parts[2:]) if parts[2:] else ""
        return f"{src} -> {tgt}{suffix}"

    return " ".join(p.capitalize() for p in parts)


def scan_models() -> list[ModelInfo]:
    if not MODELS_DIR.is_dir():
        return []

    result = []
    for entry in sorted(MODELS_DIR.iterdir()):
        if entry.suffix.lower() != ".zip":
            continue
        if not zip_is_valid(entry):
            continue

        size_mb = max(1, round(entry.stat().st_size / 1_048_576))
        result.append(ModelInfo(
            name    = make_display_name(entry.stem),
            file    = entry.name,
            size_mb = size_mb,
        ))

    return result

@app.get("/models", response_model=list[ModelInfo], summary="Список доступных моделей")
def list_models():
    return scan_models()


@app.get("/models/{filename}", summary="Скачать архив модели")
def download_model(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Недопустимое имя файла")

    path = MODELS_DIR / filename

    if not path.exists() or path.suffix.lower() != ".zip":
        raise HTTPException(status_code=404, detail=f"Файл не найден: {filename}")

    return FileResponse(
        path             = path,
        media_type       = "application/zip",
        filename         = filename,
    )

@app.get("/ping", summary="Проверить доступность сервера")
def ping():
    return {"answer": "available"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn

    if not MODELS_DIR.exists():
        print(f"[!] Папка моделей не найдена: {MODELS_DIR.resolve()}")
        print(f"    Создайте её и положите туда zip-архивы, затем перезапустите сервер.")

    uvicorn.run("model_dist_server:app", host="0.0.0.0", port=9100, reload=True)