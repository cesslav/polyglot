# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.

import torch
from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message
from transformers import AutoTokenizer
import numpy as np
import onnxruntime as ort
import asyncio

CONFIG = {
    "telegram_token": "YOUR_BOT_TOKEN_HERE",
    "proxies": [
        {
            "host": "proxy.example.com",
            "port": 1234,
            "secret": "proxy_secret_key"
        }
    ]
}


class ProxyConfig:
    """Описание одного MTProto-прокси для подключения бота к Telegram.
            Конструктор:
                host (str) - адрес прокси;
                port (int) - порт;
                secret (str|None) - секретный ключ прокси.
    """

    def __init__(self, host: str, port: int, secret: str = None):
        self.host = host
        self.port = port
        self.secret = secret
        self.type = "MTProto"

    def get_proxy_dict(self) -> dict:
        """Формирует словарь прокси в формате, ожидаемом aiogram (scheme/host/port/secret).
                Входы:
                    None - поля берутся из объекта.
                Выходы:
                    proxy (dict) - параметры прокси для Bot(proxy=...).
        """
        proxy = {
            "scheme": self.type.lower(),
            "host": self.host,
            "port": self.port
        }
        if self.secret:
            proxy["secret"] = self.secret
        return proxy


class ProxyManager:
    """Управляет набором прокси: отслеживает доступность и переключается на следующий при сбое.
            Конструктор:
                proxies (list[ProxyConfig]) - список прокси по порядку переключения.
    """
    def __init__(self, proxies: list[ProxyConfig]):
        self.proxies = proxies
        self.available = [True] * len(proxies)
        self.current_index = 0

    def get_current_proxy(self) -> ProxyConfig | None:
        """Возвращает текущий активный прокси.
                Входы:
                    None.
                Выходы:
                    proxy (ProxyConfig|None) - текущий прокси; None, если список пуст.
        """
        if not self.proxies:
            return None
        return self.proxies[self.current_index]

    def get_current_proxy_dict(self) -> dict | None:
        """Возвращает текущий прокси в виде словаря aiogram.
                Входы:
                    None.
                Выходы:
                    proxy (dict|None) - словарь прокси; None, если прокси нет.
        """
        proxy = self.get_current_proxy()
        return proxy.get_proxy_dict() if proxy else None

    async def check_proxy(self, proxy: ProxyConfig) -> bool:
        """Проверяет доступность прокси попыткой открыть (и тут же закрыть) сессию бота через него.
                Входы:
                    proxy (ProxyConfig) - проверяемый прокси.
                Выходы:
                    is_available (bool) - True, если соединение удалось.
        """
        try:
            proxy_dict = proxy.get_proxy_dict()
            async with Bot(token="dummy", proxy=proxy_dict) as dummy_bot:
                await dummy_bot.session.close()
            return True
        except Exception:
            return False

    async def validate_proxies(self) -> bool:
        """Проверяет все прокси и помечает доступные, печатает статус каждого.
                Входы:
                    None.
                Выходы:
                    is_available (bool) - True, если хотя бы один прокси доступен (или список пуст).
        """
        if not self.proxies:
            return True
        for i, proxy in enumerate(self.proxies):
            self.available[i] = await self.check_proxy(proxy)
            print(f"Прокси {i+1} ({proxy.host}:{proxy.port}): {'доступен' if self.available[i] else 'недоступен'}")
        return any(self.available)

    def switch_proxy(self):
        """Переключает активный прокси на следующий доступный в цикле.
                Входы:
                    None.
                Выходы:
                    None - current_index сдвигается; при отсутствии доступных бросает RuntimeError.
        """
        if not self.proxies:
            return
        for _ in range(len(self.proxies)):
            self.current_index = (self.current_index + 1) % len(self.proxies)
            if self.available[self.current_index]:
                proxy = self.proxies[self.current_index]
                print(f"Переключение на прокси {self.current_index + 1} ({proxy.host}:{proxy.port})")
                return
        raise RuntimeError("Все прокси недоступны")


class ONNXTransformer:
    """Обёртка над ONNX-сессиями кодера и декодера для инференса в боте.
            Конструктор:
                encoder_path (str) - путь к encoder.onnx;
                decoder_path (str) - путь к decoder.onnx;
                device (str) - зарезервированный параметр (используется CPUExecutionProvider).
    """

    def __init__(self, encoder_path, decoder_path, device="cpu"):
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)

    def encode(self, src, src_mask):
        """Прогоняет исходную последовательность через ONNX-кодер.
                Входы:
                    src (ndarray int64) - id токенов;
                    src_mask (ndarray bool) - маска не-pad.
                Выходы:
                    memory (ndarray float32) - память кодера.
        """
        return self.encoder.run(["memory"], {
            "src": src.astype(np.int64),
            "src_mask": src_mask
        })[0]

    def decode(self, tgt, memory, src_mask):
        """Прогоняет целевую последовательность через ONNX-декодер с головкой.
                Входы:
                    tgt (ndarray int64) - id токенов;
                    memory (ndarray float32) - память кодера;
                    src_mask (ndarray bool) - маска.
                Выходы:
                    logits (ndarray float32) - logits по словарю.
        """
        return self.decoder.run(["logits"], {
            "tgt": tgt.astype(np.int64),
            "memory": memory.astype(np.float32),
            "src_mask": src_mask
        })[0]


def beam_search_onnx(model, tokenizer, src, beam_size=4, max_len=256):
    """Beam search по ONNX-модели: токенизирует вход, генерирует перевод и декодирует лучший луч.
            Входы:
                model (ONNXTransformer) - модель;
                tokenizer (AutoTokenizer) - токенизатор;
                src (Tensor) - id входных токенов (1, S);
                beam_size (int) - ширина луча;
                max_len (int) - максимальная длина перевода.
            Выходы:
                translation (str) - итоговый перевод.
    """
    def log_softmax(x, axis=-1):
        """Лога-softmax по оси с вычитанием максимума для численной устойчивости.
                Входы:
                    x (ndarray) - тензор logits;
                    axis (int) - ось применения (по умолчанию -1).
                Выходы:
                    result (ndarray) - логарифм softmax той же формы.
        """
        x = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(x)
        return np.log(exp / np.sum(exp, axis=axis, keepdims=True))
    
    bos, eos = 0, 1
    src_np = src.cpu().numpy()
    src_mask = (src_np != 3)
    memory = model.encode(src_np, src_mask)
    beams = [(np.array([[bos]], dtype=np.int64), 0.0)]
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1] == eos:
                new_beams.append((seq, score))
                continue
            logits = model.decode(seq, memory, src_mask)
            log_probs = log_softmax(logits[:, -1, :], axis=-1)
            topk_idx = np.argsort(-log_probs, axis=-1)[0][:beam_size]
            topk_log_probs = log_probs[0][topk_idx]
            for k in range(beam_size):
                new_seq = np.concatenate([seq, [[topk_idx[k]]]], axis=1)
                new_score = score + float(topk_log_probs[k])
                new_beams.append((new_seq, new_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if all(seq[0, -1] == eos for seq, _ in beams):
            break
    
    best_seq = beams[0][0]
    return tokenizer.decode(best_seq[0], skip_special_tokens=True)


@dp.message(CommandStart())
async def start(message: Message):
    """Хендлер команды /start: присылает пользователю подсказку.
            Входы:
                message (Message) - входящее сообщение.
            Выходы:
                None - ответ отправляется в чат.
    """
    await message.answer("Отправь текст, и я переведу его.")


@dp.message()
async def translate(message: Message):
    """Хендлер обычных сообщений: переводит текст ONNX-моделью; при ошибке переключает прокси и сообщает об ошибке.
            Входы:
                message (Message) - сообщение с текстом для перевода.
            Выходы:
                None - перевод (или текст ошибки) отправляется в чат.
    """
    global proxy_manager
    text = message.text
    current_proxy = proxy_manager.get_current_proxy() if proxy_manager else None
    current_proxy_dict = proxy_manager.get_current_proxy_dict() if proxy_manager else None
    
    try:
        src = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")[
            "input_ids"].to(device)
        translation = beam_search_onnx(model, tokenizer, src, beam_size=4)
        await message.answer(translation)
    except Exception as e:
        if current_proxy and proxy_manager:
            print(f"Ошибка с прокси {current_proxy.host}:{current_proxy.port}: {e}")
            try:
                proxy_manager.switch_proxy()
            except RuntimeError:
                pass
        await message.answer(f"Ошибка перевода: {e}")


async def main():
    """Точка входа бота: загружает токенизатор и ONNX-модель, настраивает прокси и запускает long polling.
            Входы:
                None - настройки берутся из CONFIG.
            Выходы:
                None - бот работает до прерывания.
    """
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")

    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    model = ONNXTransformer(
        encoder_path="encoder.onnx",
        decoder_path="decoder.onnx"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global bot, dp, proxy_manager
    bot = Bot(token=CONFIG["telegram_token"])
    dp = Dispatcher()
    
    proxy_configs = []
    for p in CONFIG.get("proxies", []):
        if isinstance(p, dict):
            proxy_configs.append(ProxyConfig(
                host=p.get("host", ""),
                port=p.get("port"),
                secret=p.get("secret")
            ))
    
    proxy_manager = ProxyManager(proxy_configs) if proxy_configs else None
    
    if proxy_manager:
        if not await proxy_manager.validate_proxies():
            print("Внимание: ни один прокси не доступен. Попытка подключения без прокси...")
    
    print("Started.")
    
    proxy_dict = proxy_manager.get_current_proxy_dict() if proxy_manager else None
    await dp.start_polling(bot, proxy=proxy_dict)


if __name__ == "__main__":
    asyncio.run(main())
