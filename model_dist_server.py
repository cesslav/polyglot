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
    """Схема описания одной доступной к скачиванию ONNX-модели.
            Конструктор:
                name (str) - отображаемое имя;
                file (str) - имя zip-архива;
                size_mb (int) - размер в МБ;
                input_language (str) - язык-источник (по умолчанию "");
                output_language (str) - язык перевода (по умолчанию "");
                bidirectional (bool) - двунаправленность (по умолчанию False).
    """
    name: str
    file: str
    size_mb: int
    input_language: str = ""
    output_language: str = ""
    bidirectional: bool = False


def zip_is_valid(path: Path) -> bool:
    """Проверяет, что zip-архив корректен и содержит все обязательные файлы модели.
            Входы:
                path (Path) - путь к zip-файлу.
            Выходы:
                is_valid (bool) - True, если архив читаем и включает REQUIRED_FILES.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            return REQUIRED_FILES.issubset(set(zf.namelist()))
    except zipfile.BadZipFile:
        return False


def read_zip_model_config(path: Path) -> dict:
    """Читает model_config.json из архива модели.
            Входы:
                path (Path) - путь к zip-файлу.
            Выходы:
                config (dict) - конфигурация модели; пустой dict, если файла нет или он нечитаем.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            if "model_config.json" in zf.namelist():
                return json.loads(zf.read("model_config.json").decode("utf-8"))
    except Exception:
        pass
    return {}


def make_display_name(stem: str, cfg: dict) -> str:
    """Формирует человекочитаемое имя модели: из конфигурации (RU <-> EN) или из имени архива (ru-en-...).
            Входы:
                stem (str) - имя архива без расширения;
                cfg (dict) - model_config.json из архива.
            Выходы:
                name (str) - отображаемое имя модели.
    """
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
    """GET /models — список валидных zip-моделей из MODELS_DIR с именами, размерами и языками.
            Входы:
                None - данные берутся из каталога MODELS_DIR.
            Выходы:
                models (list[ModelInfo]) - JSON-массив описаний моделей.
    """
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
    """GET /models/{filename} — отдаёт zip-архив модели на скачивание (с защитой от path traversal).
            Входы:
                filename (str) - имя zip-файла из MODELS_DIR.
            Выходы:
                response (FileResponse) - бинарный поток архива; 400 при недопустимом имени, 404 при отсутствии.
    """
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Недопустимое имя файла")

    path = MODELS_DIR / filename
    if not path.exists() or path.suffix.lower() != ".zip":
        raise HTTPException(status_code=404, detail=f"Файл не найден: {filename}")

    return FileResponse(path=path, media_type="application/zip", filename=filename)


@app.get("/ping", summary="Проверить доступность сервера")
def ping():
    """GET /ping — проверка живости сервера.
            Входы:
                None.
            Выходы:
                response (dict) - {"answer": "available"}.
    """
    return {"answer": "available"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """GET /favicon.ico — заглушка для иконки (пустой ответ, чтобы не было 404 в логах).
            Входы:
                None.
            Выходы:
                response (Response) - HTTP 204 No Content.
    """
    return Response(status_code=204)


if __name__ == "__main__":
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")
    import uvicorn

    if not MODELS_DIR.exists():
        print(f"[!] Папка моделей не найдена: {MODELS_DIR.resolve()}")
        print(" Создайте её и положите туда zip-архивы, затем перезапустите сервер.")

    uvicorn.run("model_dist_server:app", host="0.0.0.0", port=9100, reload=True)