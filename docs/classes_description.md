# Справочник по API проекта «Полиглот»  

---

## Содержание

 - [1. imp.py — архитектура модели](#1-imppy--архитектура-модели)
     - [1.1 RMSNorm](#11-rmsnorm)
     - [1.2 Residual](#12-residual)
     - [1.3 PreNorm](#13-prenorm)
     - [1.4 FeedForward](#14-feedforward)
     - [1.5 ALiBi](#15-alibi)
     - [1.6 MultiQuerySelfAttention](#16-multiqueryselfattention)
     - [1.7 MultiQueryCrossAttention](#17-multiquerycrossattention)
     - [1.8 Encoder](#18-encoder)
     - [1.9 Decoder](#19-decoder)
     - [1.10 Transformer](#110-transformer)
 - [2. train_ddp.py — обучение модели](#2-train_ddppy--обучение-модели)
     - [2.1 get_transformer_scheduler](#21-get_transformer_scheduler)
     - [2.2 get_transformer_lrd](#22-get_transformer_lrd)
     - [2.3 init_weights](#23-init_weights)
     - [2.4 save](#24-save)
     - [2.5 snapshot_state](#25-snapshot_state)
     - [2.6 restore_state](#26-restore_state)
     - [2.7 train_epoch](#27-train_epoch)
     - [2.8 evaluate](#28-evaluate)
 - [3. pre_tokenize.py — подготовка данных](#3-pre_tokenizepy--подготовка-данных)
     - [3.1 _get_lid_model](#31-_get_lid_model)
     - [3.2 _get_labse_model](#32-_get_labse_model)
     - [3.3 _get_labse_pool](#33-_get_labse_pool)
     - [3.4 _stage1_length_ratio](#34-_stage1_length_ratio)
     - [3.5 _stage2_langid](#35-_stage2_langid)
     - [3.6 _stage3_labse](#36-_stage3_labse)
     - [3.7 _run_labse_bulk](#37-_run_labse_bulk)
     - [3.8 tokenization_wmt](#38-tokenization_wmt)
     - [3.9 tokenization_fine](#39-tokenization_fine)
     - [3.10 tokenization_flores](#310-tokenization_flores)
     - [3.11 tokenization_tatoeba](#311-tokenization_tatoeba)
     - [3.12 _non_pad_count](#312-_non_pad_count)
     - [3.13 apply_quality_filters](#313-apply_quality_filters)
 - [4. bleu_score.py — оценка качества перевода](#4-bleu_scorepy--оценка-качества-перевода)
     - [4.1 load_model](#41-load_model)
     - [4.2 load_hf_model](#42-load_hf_model)
     - [4.3 resolve_generate_kwargs](#43-resolve_generate_kwargs)
     - [4.4 beam_search](#44-beam_search)
     - [4.5 make_own_translate_fn](#45-make_own_translate_fn)
     - [4.6 make_hf_translate_fn](#46-make_hf_translate_fn)
     - [4.7 evaluate](#47-evaluate)
     - [4.8 run_comparison](#48-run_comparison)
     - [4.9 format_comparison_table](#49-format_comparison_table)
     - [4.10 plot_comparison](#410-plot_comparison)
     - [4.11 write_compare_log](#411-write_compare_log)
     - [4.12 load_flores](#412-load_flores)
 - [5. raw_use.py — инференс PyTorch-модели](#5-raw_usepy--инференс-pytorch-модели)
     - [5.1 beam_search](#51-beam_search)
 - [6. onnx_export.py — экспорт и квантизация модели](#6-onnx_exportpy--экспорт-и-квантизация-модели)
     - [6.1 EncoderWrapper](#61-encoderwrapper)
     - [6.2 DecoderWrapper](#62-decoderwrapper)
     - [6.3 export_fp32](#63-export_fp32)
     - [6.4 classify_nodes](#64-classify_nodes)
     - [6.5 quantize_int8](#65-quantize_int8)
     - [6.6 quantize_fp16](#66-quantize_fp16)
     - [6.7 apply_quantization_config](#67-apply_quantization_config)
     - [6.8 save_model_config](#68-save_model_config)
     - [6.9 load_model](#69-load_model)
 - [7. onnx_use.py — инференс ONNX в консоли](#7-onnx_usepy--инференс-onnx-в-консоли)
     - [7.1 np_softmax](#71-np_softmax)
     - [7.2 ONNXTransformer](#72-onnxtransformer)
     - [7.3 beam_search_stream](#73-beam_search_stream)
     - [7.4 print_stats](#74-print_stats)
 - [8. model_dist_server.py — сервер раздачи моделей](#8-model_dist_serverpy--сервер-раздачи-моделей)
     - [8.1 ModelInfo](#81-modelinfo)
     - [8.2 zip_is_valid](#82-zip_is_valid)
     - [8.3 read_zip_model_config](#83-read_zip_model_config)
     - [8.4 make_display_name](#84-make_display_name)
     - [8.5 Эндпоинты](#85-эндпоинты)
 - [9. web.py — веб-переводчик](#9-webpy--веб-переводчик)
     - [9.1 np_softmax](#91-np_softmax)
     - [9.2 list_model_dirs](#92-list_model_dirs)
     - [9.3 read_model_config](#93-read_model_config)
     - [9.4 make_display_name](#94-make_display_name)
     - [9.5 ONNXTransformer](#95-onnxtransformer)
     - [9.6 get_model](#96-get_model)
     - [9.7 beam_search_onnx](#97-beam_search_onnx)
     - [9.8 Эндпоинты](#98-эндпоинты)
 - [10. telegram.py — Telegram-бот](#10-telegrampy--telegram-бот)
     - [10.1 ProxyConfig](#101-proxyconfig)
     - [10.2 ProxyManager](#102-proxymanager)
     - [10.3 ONNXTransformer](#103-onnxtransformer)
     - [10.4 beam_search_onnx](#104-beam_search_onnx)
     - [10.5 Хендлеры](#105-хендлеры)
     - [10.6 main](#106-main)
 - [11. desktop_app.py — десктопное приложение](#11-desktop_apppy--десктопное-приложение)
     - [11.1 Фабрики виджетов](#111-фабрики-виджетов)
     - [11.2 RoundedPanel](#112-roundedpanel)
     - [11.3 UnigramTokenizer](#113-unigramtokenizer)
     - [11.4 OnnxTransformer](#114-onnxtransformer)
     - [11.5 greedy_search](#115-greedy_search)
     - [11.6 read_model_config](#116-read_model_config)
     - [11.7 make_display_name](#117-make_display_name)
     - [11.8 ModelDownloadManager](#118-modeldownloadmanager)
     - [11.9 Фоновые треды](#119-фоновые-треды)
     - [11.10 ModelCard](#1110-modelcard)
     - [11.11 TranslateScreen](#1111-translatescreen)
     - [11.12 DownloadsScreen](#1112-downloadsscreen)
     - [11.13 AboutScreen](#1113-aboutscreen)
     - [11.14 Header](#1114-header)
     - [11.15 NavBar](#1115-navbar)
     - [11.16 MainWindow](#1116-mainwindow)
     - [11.17 main](#1117-main)
 - [12. tok_learn.py — обучение униграмм-токенизатора](#12-tok_learnpy--обучение-униграмм-токенизатора)
     - [12.1 get_training_corpus](#121-get_training_corpus)
 - [13. tok_learn_incase.py — токенизатор с inline-регистром](#13-tok_learn_incasepy--токенизатор-с-inline-регистром)
     - [13.1 apply_inline_casing](#131-apply_inline_casing)
     - [13.2 restore_inline_casing](#132-restore_inline_casing)
     - [13.3 get_training_corpus](#133-get_training_corpus)

---

## 1. imp.py — архитектура модели

Модуль `imp.py` содержит реализацию архитектуры нейросети-переводчика — seq2seq-трансформера на базе PyTorch. Модель использует механизм multi-query attention с линейными смещениями ALiBi вместо позиционного кодирования, нормализацию RMSNorm, полносвязные слои в схеме GLU (SwiGLU), а также связывание весов эмбеддингов и выходного логит-слоя. Каждый компонент выделен в отдельный класс, что позволяет использовать его по отдельности.

---

### 1.1 RMSNorm

**Класс.** Нормализация по среднеквадратичному значению (RMS, Root Mean Square) с обучаемым коэффициентом масштабирования `gamma`.

| Аргумент | Тип     | По умолчанию | Описание                                   |
| -------- | ------- | ------------ | ------------------------------------------ |
| `dim`    | `int`   | —            | Размерность нормализуемого последнего измерения. |
| `eps`    | `float` | `1e-8`       | Слагаемое численной стабильности.          |

Атрибут `gamma` — обучаемый параметр, вектор длины `dim`. Метод `forward(x)` нормирует вход по RMS и масштабирует гаммой; форма тензора сохраняется (`[..., dim]`).

```python
import torch
from imp import RMSNorm

norm  = RMSNorm(dim=256, eps=1e-8)
out   = norm(torch.randn(2, 512, 256))
print(norm.gamma.shape)   # torch.Size([256])
print(out.shape)          # torch.Size([2, 512, 256])
```

---

### 1.2 Residual

**Класс.** Обёртка остаточного соединения: применяет внутреннее преобразование к входу и складывает результат с самим входом (`fn(x) + x`).

| Аргумент | Тип         | Описание                    |
| -------- | ----------- | --------------------------- |
| `fn`     | `nn.Module` | Модуль-преобразование.      |

Метод `forward(x, **kwargs)` пробрасывает дополнительные аргументы (например, `mask`) во внутреннее преобразование.

```python
from imp import Residual, MultiQuerySelfAttention

block = Residual(MultiQuerySelfAttention(dim=256, heads=8, dim_head=32))
x     = torch.randn(1, 32, 256)
mask  = torch.ones(1, 32, dtype=torch.bool)
out   = block(x, mask=mask)   # out.shape == (1, 32, 256)
```

---

### 1.3 PreNorm

**Класс.** Пре-нормализация в стиле pre-LN Transformer: сначала применяется `RMSNorm`, затем внутреннее преобразование `fn`.

| Аргумент | Тип         | Описание                                 |
| -------- | ----------- | ---------------------------------------- |
| `dim`    | `int`       | Размерность входного тензора.            |
| `fn`     | `nn.Module` | Преобразование, применяемое после нормализации. |

```python
from imp import PreNorm, FeedForward

layer = PreNorm(dim=256, fn=FeedForward(dim=256, mult=4, dropout=0.1))
out   = layer(torch.randn(2, 64, 256))   # (2, 64, 256)
```

---

### 1.4 FeedForward

**Класс.** Позиционно-независимый многослойный перцептрон в схеме GLU: три проекции `w_gate`, `w_up`, `w_down` без смещений и активация SiLU. Внутренняя размерность равна `dim * mult`.

| Аргумент   | Тип     | По умолчанию | Описание                                  |
| ---------- | ------- | ------------ | ----------------------------------------- |
| `dim`      | `int`   | —            | Размерность входа и выхода.               |
| `mult`     | `int`   | `4`          | Множитель внутренней размерности.         |
| `dropout`  | `float` | `0.0`        | Вероятность отбрасывания.                 |

Метод `forward(x)` вычисляет `down(SiLU(gate(x)) * up(x))` с dropout; форма тензора сохраняется.

```python
from imp import FeedForward

ff  = FeedForward(dim=128, mult=4, dropout=0.0)
out = ff(torch.randn(4, 10, 128))   # (4, 10, 128)
```

---

### 1.5 ALiBi

**Класс.** Генератор матриц смещений ALiBi (Attention with Linear Biases): линейное расстояние между позициями с индивидуальным сужением для каждой головы внимания. Позиционные эмбеддинги не используются.

| Аргумент     | Тип   | По умолчанию | Описание                                        |
| ------------ | ----- | ------------ | ----------------------------------------------- |
| `heads`      | `int` | —            | Число голов внимания.                           |
| `max_seq_len`| `int` | `1024`       | Максимальная длина последовательности.          |

В конструкторе вычисляется и регистрируется как буфер тензор `_bias` формы `(heads, max_seq_len, max_seq_len)` (не сохраняется в контрольные точки).

Методы:

- `_slopes(n)` — статический метод, возвращает вектор наклонностей по геометрической прогрессии `2^(-(2^-(m-4-i)))`; корректно обрабатывает значения `n`, не являющиеся степенью двойки;
- `get_bias(q_len, k_len)` — вырезает подматрицу смещений формы `(heads, q_len, k_len)` для текущих длин запроса и ключей.

```python
from imp import ALiBi

alibi  = ALiBi(heads=8, max_seq_len=512)
bias   = alibi.get_bias(q_len=16, k_len=512)   # (8, 16, 512)
slopes = ALiBi._slopes(n=12)                   # тензор (12,)
```

---

### 1.6 MultiQuerySelfAttention

**Класс.** Самовнимание в схеме multi-query attention: множество query-голов и одна общая K/V-пара, линейные смещения ALiBi, опциональная каузальная маска.

| Аргумент     | Тип     | По умолчанию | Описание                                          |
| ------------ | ------- | ------------ | ------------------------------------------------- |
| `dim`        | `int`   | —            | Размерность входного тензора.                     |
| `heads`      | `int`   | `8`          | Число query-голов.                                |
| `dim_head`   | `int`   | `64`         | Размерность одной головы.                         |
| `causal`     | `bool`  | `False`      | Использовать ли каузальную (нижнетреугольную) маску. |
| `dropout`    | `float` | `0.0`        | Dropout внимания (применяется только в режиме `train`). |
| `max_seq_len`| `int`   | `1024`       | Максимальная длина последовательности для ALiBi.  |

Метод `forward(x, mask=None)` принимает `x` формы `(batch, seq, dim)` и булеву маску `mask` формы `(batch, seq)`, где `True` соответствует валидному (не pad) токену; возвращает тензор формы `(batch, seq, dim)`.

```python
import torch
from imp import MultiQuerySelfAttention

attn = MultiQuerySelfAttention(dim=256, heads=8, dim_head=32,
                               causal=True, max_seq_len=256)
x    = torch.randn(2, 32, 256)
out  = attn(x, mask=torch.ones(2, 32, dtype=torch.bool))
# out.shape == (2, 32, 256)
```

---

### 1.7 MultiQueryCrossAttention

**Класс.** Кросс-внимание в схеме multi-query attention: запросы формируются из последовательности декодера, а общие K/V-проекции применяются к контексту кодера.

| Аргумент      | Тип         | По умолчанию | Описание                                          |
| ------------- | ----------- | ------------ | ------------------------------------------------- |
| `dim`         | `int`       | —            | Размерность последовательности декодера.          |
| `context_dim` | `int` или `None` | `None` (= `dim`) | Размерность контекста кодера.               |
| `heads`       | `int`       | `8`          | Число query-голов.                                |
| `dim_head`    | `int`       | `64`         | Размерность одной головы.                         |
| `dropout`     | `float`     | `0.0`        | Dropout внимания.                                 |

Метод `forward(x, context, mask=None, context_mask=None)` принимает `x` формы `(batch, n, dim)`, контекст `context` формы `(batch, m, context_dim)` и маски `(batch, n)` и `(batch, m)` соответственно.

```python
import torch
from imp import MultiQueryCrossAttention

cross = MultiQueryCrossAttention(dim=256, heads=8, dim_head=32)
dec   = torch.randn(1, 8, 256)
mem   = torch.randn(1, 32, 256)
out   = cross(dec, context=mem,
              mask=torch.ones(1, 8, dtype=torch.bool),
              context_mask=torch.ones(1, 32, dtype=torch.bool))
# out.shape == (1, 8, 256)
```

---

### 1.8 Encoder

**Класс.** Кодер seq2seq-модели: эмбеддинги токенов, стек из `depth` блоков «самовнимание (ALiBi) + полносвязный слой» с пре-нормализацией и остаточными соединениями, финальная нормализация RMSNorm.

| Аргумент     | Тип     | По умолчанию | Описание                                        |
| ------------ | ------- | ------------ | ----------------------------------------------- |
| `dim`        | `int`   | —            | Размерность скрытого слоя.                      |
| `num_tokens` | `int`   | —            | Размер словаря.                                 |
| `depth`      | `int`   | —            | Число слоёв.                                    |
| `heads`      | `int`   | `8`          | Число голов внимания.                           |
| `dim_head`   | `int`   | `64`         | Размерность одной головы.                       |
| `mlp_mult`   | `int`   | `4`          | Множитель размера полносвязного слоя.           |
| `dropout`    | `float` | `0.0`        | Вероятность отбрасывания.                       |
| `max_seq_len`| `int`   | `1024`       | Максимальная длина последовательности для ALiBi.|

Метод `forward(x, mask=None)` принимает id токенов `x` формы `(batch, seq)` и маску `(batch, seq)`; возвращает память кодера `memory` формы `(batch, seq, dim)`.

```python
import torch
from imp import Encoder

enc    = Encoder(dim=256, num_tokens=48000, depth=4, heads=8, dim_head=32)
x      = torch.randint(0, 48000, (2, 64))
memory = enc(x, mask=torch.ones(2, 64, dtype=torch.bool))
print(memory.shape)   # torch.Size([2, 64, 256])
```

---

### 1.9 Decoder

**Класс.** Декодер seq2seq-модели: эмбеддинги токенов, стек из `depth` блоков «каузальное самовнимание + кросс-внимание + полносвязный слой», финальная нормализация RMSNorm. Параметры конструктора совпадают с параметрами класса `Encoder`.

Метод `forward(x, context, mask=None, context_mask=None)` принимает id токенов `x` формы `(batch, tgt_seq)`, память кодера `context` формы `(batch, src_seq, dim)` и маски `(batch, tgt_seq)` / `(batch, src_seq)`; возвращает представления декодера формы `(batch, tgt_seq, dim)`.

```python
import torch
from imp import Decoder

dec     = Decoder(dim=256, num_tokens=48000, depth=4, heads=8, dim_head=32)
tgt     = torch.randint(0, 48000, (2, 16))
context = torch.randn(2, 64, 256)
out     = dec(tgt, context,
              mask=torch.ones(2, 16, dtype=torch.bool),
              context_mask=torch.ones(2, 64, dtype=torch.bool))
# out.shape == (2, 16, 256)
```

---

### 1.10 Transformer

**Класс.** Полная seq2seq-модель: кодер, декодер и выходной логит-слой. Веса логит-слоя связаны с эмбеддингами декодера, а при `tie_token_emb=True` дополнительно — с эмбеддингами кодера (в этом случае размеры словарей должны совпадать, иначе выбрасывается `ValueError`).

| Аргумент         | Тип     | По умолчанию | Описание                                        |
| ---------------- | ------- | ------------ | ----------------------------------------------- |
| `dim`            | `int`   | —            | Размерность скрытого слоя.                      |
| `enc_num_tokens` | `int`   | —            | Размер словаря кодера.                          |
| `enc_depth`      | `int`   | —            | Число слоёв кодера.                             |
| `enc_heads`      | `int`   | —            | Число голов внимания кодера.                    |
| `enc_dim_head`   | `int`   | —            | Размерность головы внимания кодера.             |
| `enc_mlp_mult`   | `int`   | —            | Множитель MLP кодера.                           |
| `dec_num_tokens` | `int`   | —            | Размер словаря декодера.                        |
| `dec_depth`      | `int`   | —            | Число слоёв декодера.                           |
| `dec_heads`      | `int`   | —            | Число голов внимания декодера.                  |
| `dec_dim_head`   | `int`   | —            | Размерность головы внимания декодера.           |
| `dec_mlp_mult`   | `int`   | —            | Множитель MLP декодера.                         |
| `dropout`        | `float` | `0.0`        | Вероятность отбрасывания.                       |
| `max_seq_len`    | `int`   | `1024`       | Максимальная длина последовательности для ALiBi.|
| `tie_token_emb`  | `bool`  | `True`       | Связывать ли веса эмбеддингов кодера и декодера.|

Атрибуты: `encoder` (класс `Encoder`), `decoder` (класс `Decoder`), `to_logits` (линейный слой, веса связаны с эмбеддингами декодера).

Метод `forward(src, tgt, src_mask=None, tgt_mask=None)` принимает id исходных `src` (`(batch, src_seq)`) и целевых `tgt` (`(batch, tgt_seq)`) токенов; возвращает логиты формы `(batch, tgt_seq, dec_num_tokens)`.

```python
import torch
from imp import Transformer

model = Transformer(
    dim=512,
    enc_num_tokens=48000, enc_depth=4, enc_heads=8,
    enc_dim_head=32, enc_mlp_mult=4,
    dec_num_tokens=8000,  dec_depth=4, dec_heads=8,
    dec_dim_head=32, dec_mlp_mult=4,
    dropout=0.1,
    max_seq_len=512,
)
src    = torch.randint(0, 48000, (2, 64))
tgt    = torch.randint(0, 8000, (2, 48))
logits = model(src, tgt,
               src_mask=torch.ones(2, 64, dtype=torch.bool),
               tgt_mask=torch.ones(2, 48, dtype=torch.bool))
print(logits.shape)   # torch.Size([2, 48, 8000])
```

---

## 2. train_ddp.py — обучение модели

Модуль `train_ddp.py` реализует распределённое обучение модели на нескольких устройствах (DistributedDataParallel, бэкенд NCCL, автокаст bfloat16). Обучение ведётся в обоих направлениях за один проход по батчу (см. раздел 3.3 `README.md`), применяется градиентное накопление, послойное затухание шага обучения и адаптивный клип градиента; при обнаружении некорректных значений в весах выполняется откат на последний исправный снимок.

---

### 2.1 get_transformer_scheduler

**Функция.** Создаёт расписание шага обучения: линейный warmup до `warmup_steps` шагов, после чего — степенной спад `(warmup/step)^0.65`.

| Аргумент       | Тип         | Описание                          |
| -------------- | ----------- | --------------------------------- |
| `optimizer`    | `Optimizer` | Оптимизатор, к которому применяется расписание. |
| `warmup_steps` | `int`       | Количество шагов warmup.          |

**Возвращает** объект `LambdaLR`, готовый к вызову `scheduler.step()` после каждого обновления параметров.

```python
import torch
from train_ddp import get_transformer_scheduler, get_transformer_lrd

opt   = torch.optim.AdamW(get_transformer_lrd(model), betas=(0.9, 0.98))
sched = get_transformer_scheduler(opt, warmup_steps=6000)

for _ in range(3):
    opt.step()
    sched.step()
```

---

### 2.2 get_transformer_lrd

**Функция.** Формирует группы параметров оптимизатора с послойным затуханием шага обучения (layer-wise LR decay): шаг для слоя с глубиной `depth` вычисляется как `base_lr * decay ** (max_depth - depth)`. Параметры `bias`, `gamma` и `beta` выносятся в отдельные группы без весовой стабилизации.

| Аргумент       | Тип         | По умолчанию | Описание                                |
| -------------- | ----------- | ------------ | --------------------------------------- |
| `model`        | `Transformer` | —          | Модель, параметры которой группируются. |
| `base_lr`      | `float`     | `1e-4`       | Базовый шаг обучения для верхнего слоя. |
| `decay`        | `float`     | `0.9`        | Множитель затухания шага на слой вниз.  |
| `weight_decay` | `float`     | `0.01`       | Коэффициент весовой стабилизации.       |

**Возвращает** список словарей с ключами `params`, `lr`, `weight_decay` — готовый аргумент конструктора оптимизатора.

```python
groups = get_transformer_lrd(model, base_lr=1e-4, decay=0.9, weight_decay=0.02)
opt    = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-9, fused=True)
for g in groups:
    print(f"lr={g['lr']:.2e}  wd={g['weight_decay']}  n={len(g['params'])}")
```

---

### 2.3 init_weights

**Функция.** Инициализация весов модулей: для линейных слоёв (`nn.Linear`) применяется инициализация `xavier_uniform_` (смещения заполняются значением `0.01`), для слоёв эмбеддингов (`nn.Embedding`) — нормальное распределение со средним `0.0` и стандартным отклонением `0.02`. Предназначена для вызова через `model.apply(init_weights)`.

```python
from train_ddp import init_weights

model.apply(init_weights)
```

---

### 2.4 save

**Функция.** Сохраняет контрольную точку `transformer_epoch_{epoch}.pt` в директорию `checkpoint_dir`: веса модели, состояния оптимизатора и расписания, текущие метрики, конфигурацию модели и время сохранения. Запись выполняется только на rank 0; каждые пятое сохранение дополнительно дублируется в директорию `./cold_saves/` с метрикой в имени файла.

| Аргумент      | Тип          | По умолчанию | Описание                                   |
| ------------- | ------------ | ------------ | ------------------------------------------ |
| `transformer` | `DDP` или `Transformer` | —   | Модель (с обёрткой DDP или без).         |
| `epoch`       | `int`        | —            | Номер эпохи.                               |
| `optimizer`   | `Optimizer`  | —            | Оптимизатор.                               |
| `scheduler`   | `LambdaLR`   | —            | Расписание шага обучения.                  |
| `train_loss`  | `float`      | `0`          | Потеря обучения (для имени cold-сохранения). |
| `val_loss`    | `float` или `str` | `"NaN"` | Потеря валидации.                      |
| `progress`    | `float`      | `0`          | Прогресс по текущей эпохе (от 0 до 1).    |

```python
save(ddp_model, epoch=12, optimizer=opt, scheduler=sched,
     train_loss=2.31, val_loss=3.02, progress=0.42)
```

> **Примечание.** Контрольные точки сохраняются автоматически: раз в час в ходе эпохи и по её завершении (см. [2.7](#27-train_epoch)).

---

### 2.5 snapshot_state

**Функция.** Делает снимок текущего состояния модели и оптимизатора (глубокая копия словаря состояний), который используется для отката при обнаружении некорректных значений в весах.

| Аргумент    | Тип        | Описание                              |
| ----------- | ---------- | ------------------------------------- |
| `model`     | `DDP` или `Transformer` | Модель.                  |
| `optimizer` | `Optimizer`| Оптимизатор.                          |

**Возвращает** кортеж `(model_state, optimizer_state)`, где каждый элемент — глубокая копия соответствующего состояния.

```python
from train_ddp import snapshot_state

snap = snapshot_state(ddp_model, opt)
```

---

### 2.6 restore_state

**Функция.** Восстанавливает веса модели и состояние оптимизатора из снимка, ранее созданного функцией `snapshot_state`.

| Аргумент   | Тип        | Описание                                    |
| ---------- | ---------- | ------------------------------------------- |
| `model`    | `DDP` или `Transformer` | Модель.                        |
| `optimizer`| `Optimizer`| Оптимизатор.                                |
| `snapshot` | `tuple`    | Результат вызова `snapshot_state`.          |

```python
from train_ddp import restore_state

restore_state(ddp_model, opt, snap)
```

---

### 2.7 train_epoch

**Функция.** Обучение одной эпохи. На каждом батче модель дважды прогоняется в обоих направлениях (src → tgt и tgt → src) в режиме автокаста bfloat16; градиенты накапливаются на протяжении `accumulation_steps` шагов. Клип градиента адаптивный: пока окно наблюдения не заполнено, применяется `clip_default`, затем — среднее по окну `clip_window`, умноженное на `clip_mult`. Батчи с ошибкой обратного прохода и шаги с неконечной нормой градиента пропускаются; при неконечной норме весов выполняется откат на последний снимок. Снимок обновляется каждые `snapshot_interval` шагов.

| Аргумент             | Тип          | По умолчанию | Описание                                       |
| -------------------- | ------------ | ------------ | ---------------------------------------------- |
| `model`              | `DDP`        | —            | Модель.                                        |
| `loader`             | `DataLoader` | —            | Источник обучающих батчей.                     |
| `optimizer`          | `Optimizer`  | —            | Оптимизатор.                                   |
| `scheduler`          | `LambdaLR`   | —            | Расписание шага обучения.                      |
| `criterion`          | `nn.Module`  | —            | Функция потерь (с `ignore_index` для pad).     |
| `device`             | `str`        | —            | Устройство вычислений.                         |
| `num`                | `int`        | —            | Номер эпохи (отображается в прогресс-баре).    |
| `accumulation_steps` | `int`        | `12`         | Число шагов накопления градиентов.             |
| `clip_window`        | `int`        | `30`         | Размер окна наблюдения норм градиента.         |
| `clip_mult`          | `float`      | `1.25`       | Множитель к среднему клипу.                    |
| `clip_default`       | `float`      | `3`          | Клип до заполнения окна.                       |
| `snapshot_interval`  | `int`        | `100`        | Период обновления снимка для отката.           |

**Возвращает** среднюю потерю за эпоху (`float`).

```python
import torch.nn as nn
from train_ddp import train_epoch

criterion  = nn.CrossEntropyLoss(ignore_index=3, label_smoothing=0.07)
train_loss = train_epoch(ddp_model, train_loader, opt, sched,
                         criterion, device, num=1,
                         accumulation_steps=12)
```

---

### 2.8 evaluate

**Функция.** Валидация: вычисляет среднюю потерю в обоих направлениях (fwd: src → tgt и bwd: tgt → src) без обратного прохода. Прогресс отображается в прогресс-баре с текущими значениями потерь.

| Аргумент    | Тип        | Описание                                    |
| ----------- | ---------- | ------------------------------------------- |
| `model`     | `Transformer` или `DDP` | Модель.                      |
| `loader`    | `DataLoader`| Валидационные батчи.                        |
| `criterion` | `nn.Module`| Функция потерь (pad-токены игнорируются).   |
| `device`    | `str`      | Устройство вычислений.                      |

**Возвращает** кортеж `(avg_loss, fwd_loss, bwd_loss)`.

```python
sum_loss, fwd, bwd = evaluate(model.module, val_loader, criterion, device)
print(f"val: sum={sum_loss:.4f}  fwd={fwd:.4f}  bwd={bwd:.4f}")
```

> **Примечание.** Скрипт запускается командой `python train_ddp.py` (при отсутствии переменной окружения `LOCAL_RANK` он сам пересоздаётся через `torch.distributed.run`) либо непосредственно: `torchrun --nproc_per_node=<N> train_ddp.py`.

---

## 3. pre_tokenize.py — подготовка данных

Модуль `pre_tokenize.py` выполняет токенизацию параллельных корпусов (WMT19, FineTranslations, Tatoeba) и их многоступенчатую фильтрацию по качеству, после чего сохраняет тренировочный и валидационный наборы данных в директории `sources/`. Настройка фильтров задаётся глобальной константой `FILTER_CONFIG`; длина тензоров — глобальной константой `tensor_size` (512), языковая пара — `SRC_LANG`/`TGT_LANG`.

---

### 3.1 _get_lid_model

**Функция.** Ленивая загрузка модели определения языка fastText: файл модели загружается однократно из пути, заданного в `FILTER_CONFIG["langid"]["model_path"]`.

```python
FILTER_CONFIG["langid"]["enabled"] = True
FILTER_CONFIG["langid"]["model_path"] = "./lid.176.bin"

m = _get_lid_model()
print(m.predict("Привет, мир", k=1))
```

---

### 3.2 _get_labse_model

**Функция.** Ленивая загрузка модели эмбеддингов предложений LaBSE (`sentence-transformers`); имя модели берётся из `FILTER_CONFIG["labse"]["model_name"]`, загрузка выполняется однократно.

```python
labse = _get_labse_model()
emb = labse.encode(["Привет", "Hello"], normalize_embeddings=True)
```

---

### 3.3 _get_labse_pool

**Функция.** Ленивое создание мультипроцессного пула для массовых эмбеддингов: пул запускается на двух GPU при их наличии, иначе на CPU.

```python
pool = _get_labse_pool()
emb  = labse.encode(texts, pool=pool, batch_size=512)
```

---

### 3.4 _stage1_length_ratio

**Функция.** Первый этап фильтрации: ограничение длины пар в токенах (`min_tokens`/`max_tokens`) и соотношение длин исходного и целевого текста в символах (не более `max_ratio`).

| Аргумент  | Тип    | Описание                                                                 |
| --------- | ------ | ------------------------------------------------------------------------ |
| `example` | `dict` | Запись датасета с полями `input`/`output` (токены) и `src_text`/`tgt_text` (тексты). |

**Возвращает** `True`, если пара проходит оба фильтра.

```python
example = {"input": [0, 5, 7, 3, 3], "output": [0, 9, 3, 3, 3],
           "src_text": "Привет", "tgt_text": "Hello"}
keep = _stage1_length_ratio(example)   # True
```

---

### 3.5 _stage2_langid

**Функция.** Второй этап фильтрации: проверка языка исходника и перевода с помощью fastText-классификатора; пара сохраняется, если определённый язык совпадает с ожидаемым (`src_lang`/`tgt_lang` из `FILTER_CONFIG`).

```python
keep = _stage2_langid({"src_text": "Привет, мир",
                       "tgt_text": "Hello, world"})
```

---

### 3.6 _stage3_labse

**Функция.** Третий этап фильтрации (батчевый): косинусная близость эмбеддингов исходного и целевого предложений; пары с близостью ниже `min_cosine` отбрасываются.

| Аргумент | Тип    | Описание                                        |
| -------- | ------ | ----------------------------------------------- |
| `batch`  | `dict` | Батч записей со списками `src_text`/`tgt_text`. |

**Возвращает** список булевых меток по одной на пару.

```python
keep = _stage3_labse({"src_text": ["Привет", "Пока"],
                      "tgt_text": ["Hello", "Bye"]})   # [True, True]
```

---

### 3.7 _run_labse_bulk

**Функция.** Массовая фильтрация датасета по семантической близости: обработка чанками с использованием мультипроцессного пула (см. [3.3](#33-_get_labse_pool)); возвращает датасет, отобранный по индексов прошедших пар.

```python
dataset = _run_labse_bulk(dataset)
```

---

### 3.8 tokenization_wmt

**Функция.** Токенизация пары предложений из набора `wmt/wmt19`: извлекаются русская и английская части поля `translation`, обе стороны токенизируются с padding до `tensor_size`.

| Аргумент  | Тип    | Описание                                |
| --------- | ------ | --------------------------------------- |
| `example` | `dict` | Запись датасета `wmt/wmt19`.            |
| `num`     | `int`  | Индекс записи (не используется).        |

**Возвращает** словарь `{"input": uint16[512], "output": uint16[512], "src_text": str, "tgt_text": str}`.

```python
example = {"translation": {"ru": "Добрый день", "en": "Good day"}}
res     = tokenization_wmt(example, num=0)
```

---

### 3.9 tokenization_fine

**Функция.** Токенизация пары из набора `HuggingFaceFW/finetranslations` (поля `og_full_text`/`translated_text`). Пары с оценкой `og_quality_score` не выше `0.5` заменяются pad-заглушкой длиной `tensor_size + 1` с пустыми текстами.

```python
example = {"og_full_text": "Привет", "translated_text": "Hello",
           "og_quality_score": 0.8}
res     = tokenization_fine(example, num=0)
```

---

### 3.10 tokenization_flores

**Функция.** Токенизация пары из набора `openlanguagedata/flores_plus`; текст берётся по индексу `num` из глобальных списков `ds[pairs[0]]` и `ds[pairs[1]]`, поэтому функция применяется к датасету исходного языка.

```python
pairs = [0, 1]   # dev: (русский, английский)
dataset1 = load_dataset("openlanguagedata/flores_plus", "rus_Cyrl",
                        split="dev").map(tokenization_flores,
                                         with_indices=True)
```

---

### 3.11 tokenization_tatoeba

**Функция.** Токенизация пары из набора `ymoslem/Tatoeba-Translations`. Если направление пары совпадает с целевым, стороны сохраняются; при обратном направлении — меняются местами; пары для других языков заменяются pad-заглушкой.

```python
example = {"lang_src": "rus", "lang_tgt": "eng",
           "sentence_src": "Привет", "sentence_tgt": "Hello"}
res     = tokenization_tatoeba(example, num=0)
```

---

### 3.12 _non_pad_count

**Функция.** Подсчёт количества токенов, отличных от `PAD_ID`; работает и со списками, и с массивами.

```python
_non_pad_count([3, 3, 5, 7])      # 2
_non_pad_count(np.array([3, 5]))  # 1
```

---

### 3.13 apply_quality_filters

**Функция.** Прогоняет токенизированный датасет через включённые в `FILTER_CONFIG` этапы фильтрации (длина и соотношение длин → языковая идентификация → семантическая близость LaBSE), удаляет текстовые столбцы и переводит датасет в torch-формат со столбцами `input`/`output`. Размеры датасета на каждом этапе выводятся в консоль.

| Аргумент             | Тип    | По умолчанию | Описание                                   |
| -------------------- | ------ | ------------ | ------------------------------------------ |
| `dataset`            | `Dataset` | —          | Токенизированный датасет.                  |
| `num_proc_cheap`     | `int`  | `20`         | Число процессов для «дешёвых» фильтров.    |
| `skip_model_filters` | `bool` | `False`      | Зарезервированный флаг пропуска модельных фильтров. |

```python
from datasets import load_dataset

wmt = load_dataset("wmt/wmt19", "ru-en", split="train")
wmt = wmt.map(tokenization_wmt, with_indices=True, num_proc=20)
wmt = apply_quality_filters(wmt)
wmt.save_to_disk("./sources/s512_clear")
```

> **ВНИМАНИЕ!** Временные файлы, создаваемые при обработке, могут занимать значительный объём дискового пространства.

---

## 4. bleu_score.py — оценка качества перевода

Модуль `bleu_score.py` выполняет количественную оценку качества перевода по метрике BLEU (`sacrebleu`) на корпусе `openlanguagedata/flores_plus`. Скрипт загружает обученную модель из контрольной точки, переводит корпус лучевым поиском и (при включённом режиме `COMPARE_MODE`) сравнивает результат с открытыми моделями-переводчиками, перечисленными в `HF_MODELS`. По завершении формируются текстовый отчёт и график построчного BLEU.

---

### 4.1 load_model

**Функция.** Загружает модель Polyglot из PyTorch-чекпоинта в режиме inference: архитектура восстанавливается из конфигурации, содержащейся в чекпоинте.

| Аргумент          | Тип   | Описание                          |
| ----------------- | ----- | --------------------------------- |
| `checkpoint_path` | `str` | Путь к файлу контрольной точки (`.pt`). |
| `device`          | `str` | Устройство вычислений (`"cuda"` / `"cpu"`). |

**Возвращает** кортеж `(model, config)`.

```python
model, config = load_model("./t5s/transformer_epoch_1.pt", "cuda")
```

---

### 4.2 load_hf_model

**Функция.** Загружает seq2seq-модель и токенизатор из репозитория HuggingFace; модель переводится в режим `eval`, в консоль выводится число параметров.

| Аргумент   | Тип   | Описание                                   |
| ---------- | ----- | ------------------------------------------ |
| `model_id` | `str` | Идентификатор репозитория HuggingFace.     |
| `device`   | `str` | Устройство вычислений.                     |

**Возвращает** кортеж `(tokenizer, model)`.

```python
tok, m = load_hf_model("Helsinki-NLP/opus-mt-ru-en", "cuda")
```

---

### 4.3 resolve_generate_kwargs

**Функция.** Разрешает строковое значение `forced_bos_token_id` (код языка) в числовой идентификатор токена; используется для моделей NLLB, где целевой язык задаётся принудительным bos-токеном.

| Аргумент          | Тип      | Описание                                |
| ----------------- | -------- | --------------------------------------- |
| `tokenizer`       | `AutoTokenizer` | Токенизатор модели.             |
| `generate_kwargs` | `dict`   | Аргументы метода `model.generate`.      |

**Возвращает** словарь той же структуры, в котором `forced_bos_token_id` заменён на идентификатор токена.

```python
kwargs = resolve_generate_kwargs(nllb_tok, {"forced_bos_token_id": "eng_Latn"})
```

---

### 4.4 beam_search

**Функция.** Поиск лучом для собственной модели Polyglot: кодер прогоняется один раз, после чего декодер последовательно выдаёт токены до `max_len` или досрочного завершения, когда все лучи завершены.

| Аргумент    | Тип              | Описание                          |
| ----------- | ---------------- | --------------------------------- |
| `model`     | `Transformer`    | Модель.                           |
| `tokenizer` | `AutoTokenizer`  | Токенизатор.                      |
| `src`       | `Tensor`         | Id входных токенов `(batch, seq)`.|
| `src_mask`  | `Tensor`         | Маска не-pad позиций.             |
| `beam_size` | `int`            | Ширина луча.                      |
| `max_len`   | `int`            | Максимальная длина перевода.      |
| `device`    | `str`            | Устройство вычислений.            |

**Возвращает** декодированный лучший перевод (`str`).

```python
src = tokenizer(text, return_tensors="pt", padding="max_length",
                truncation=True, max_length=512)["input_ids"].to("cuda")
out = beam_search(model, tokenizer, src, src.ne(3),
                  beam_size=1, max_len=512, device="cuda")
```

---

### 4.5 make_own_translate_fn

**Функция.** Фабрика функции перевода одной строки собственной моделью: возвращаемая функция токенизирует вход с padding до 512 и передаёт его в `beam_search`.

| Аргумент    | Тип         | Описание                  |
| ----------- | ----------- | ------------------------- |
| `model`     | `Transformer` | Модель.                 |
| `tokenizer` | `AutoTokenizer` | Токенизатор.           |
| `beam_size` | `int`       | Ширина луча.              |
| `max_len`   | `int`       | Максимальная длина перевода. |
| `device`    | `str`       | Устройство вычислений.    |

**Возвращает** функцию `translate_one(text: str) -> str`.

```python
translate = make_own_translate_fn(model, tokenizer, 1, 512, "cuda")
print(translate("Привет, как дела?"))
```

---

### 4.6 make_hf_translate_fn

**Функция.** Фабрика функции перевода одной строки моделью из HuggingFace: возвращаемая функция использует `model.generate` с beam search и произвольными дополнительными аргументами (например, `forced_bos_token_id` для NLLB).

| Аргумент          | Тип                    | Описание                          |
| ----------------- | ---------------------- | --------------------------------- |
| `model`           | `AutoModelForSeq2SeqLM`| Модель HuggingFace.               |
| `tokenizer`       | `AutoTokenizer`        | Её токенизатор.                   |
| `beam_size`       | `int`                  | Ширина луча.                      |
| `max_len`         | `int`                  | Максимальная длина перевода.      |
| `device`          | `str`                  | Устройство вычислений.            |
| `generate_kwargs` | `dict` или `None`      | Дополнительные аргументы `generate`. |

**Возвращает** функцию `translate_one(text: str) -> str`.

```python
translate = make_hf_translate_fn(hf_model, hf_tok, 1, 512, "cuda",
                                 generate_kwargs={"forced_bos_token_id": 253436})
```

---

### 4.7 evaluate

**Функция.** Оценивает качество перевода корпуса: все предложения переводятся переданной функцией, после чего вычисляются corpus BLEU, построчные значения BLEU и показатели скорости.

| Аргумент        | Тип            | Описание                                  |
| --------------- | -------------- | ----------------------------------------- |
| `translate_fn`  | `callable`     | Функция перевода одного текста.           |
| `src_sentences` | `list[str]`    | Исходные предложения.                     |
| `ref_sentences` | `list[str]`    | Эталонные переводы.                       |
| `desc`          | `str`          | Подпись прогресс-бара.                    |

**Возвращает** кортеж `(corpus_result, hypotheses, sentence_bleu, elapsed, total_chars)`.

```python
corpus, hyps, sbleu, t, chars = evaluate(translate, srcs, refs,
                                         desc="Polyglot")
print(corpus.score, t)
```

---

### 4.8 run_comparison

**Функция.** Прогоняет оценку BLEU для всех моделей-участниц сравнения на одном и том же корпусе.

| Аргумент        | Тип                        | Описание                          |
| --------------- | -------------------------- | --------------------------------- |
| `models`        | `list[tuple[str, callable]]` | Пары «имя модели, функция перевода». |
| `src_sentences` | `list[str]`                | Исходные предложения.             |
| `ref_sentences` | `list[str]`                | Эталонные переводы.               |
| `split`         | `str`                      | Название сплита (для отчёта).     |

**Возвращает** словарь, где каждому имени модели соответствует словарь с ключами `corpus`, `hypotheses`, `sentence_bleu`, `elapsed`, `total_chars`.

```python
results = run_comparison([("Polyglot", own_fn),
                          ("opus-mt-ru-en", hf_fn)],
                         srcs, refs, split="devtest")
```

---

### 4.9 format_comparison_table

**Функция.** Формирует текстовую таблицу сравнения метрик: для каждой модели — corpus BLEU и минимальное, среднее и максимальное значения построчного BLEU.

| Аргумент      | Тип         | Описание                          |
| ------------- | ----------- | --------------------------------- |
| `results`     | `dict`      | Результат функции `run_comparison`. |
| `model_names` | `list[str]` | Порядок имён моделей в таблице.   |

**Возвращает** список строк таблицы (заголовок, разделитель, строки моделей).

```python
for line in format_comparison_table(results,
                                    ["Polyglot", "opus-mt-ru-en"]):
    print(line)
```

---

### 4.10 plot_comparison

**Функция.** Строит график построчного BLEU всех моделей, сохраняет его в файл PNG и пытается отобразить на экране; при отсутствии дисплея вывод пропускается с уведомлением в консоль.

| Аргумент      | Тип         | Описание                          |
| ------------- | ----------- | --------------------------------- |
| `results`     | `dict`      | Результат функции `run_comparison`. |
| `model_names` | `list[str]` | Имена моделей.                    |
| `save_path`   | `str`       | Путь для сохранения графика.      |

```python
plot_comparison(results, ["Polyglot", "opus-mt-ru-en"],
                "bleu_compare_dev.png")
```

---

### 4.11 write_compare_log

**Функция.** Записывает полный отчёт о сравнении в текстовый файл: конфигурацию модели, метрики и скорость каждой модели, таблицу сравнения и до двадцати примеров переводов в формате «исходный текст, переводы моделей, эталон».

| Аргумент        | Тип         | Описание                          |
| --------------- | ----------- | --------------------------------- |
| `log_path`      | `str`       | Путь к файлу отчёта.              |
| `own_config`    | `dict`      | Конфигурация модели Polyglot.     |
| `split`         | `str`       | Название сплита.                  |
| `src_sentences` | `list[str]` | Исходные тексты.                  |
| `ref_sentences` | `list[str]` | Эталонные переводы.               |
| `results`       | `dict`      | Результат функции `run_comparison`. |
| `model_names`   | `list[str]` | Имена моделей.                    |

```python
write_compare_log("bleu_compare_devtest.txt", config, "devtest",
                  srcs, refs, results, ["Polyglot", "opus-mt-ru-en"])
```

---

### 4.12 load_flores

**Функция.** Загружает пары предложений из набора `openlanguagedata/flores_plus` для оценки.

| Аргумент   | Тип   | По умолчанию | Описание                                  |
| ---------- | ----- | ------------ | ----------------------------------------- |
| `src_lang` | `str` | —            | Код языка-источника (ISO 15924, например `rus_Cyrl`). |
| `tgt_lang` | `str` | —            | Код языка перевода.                       |
| `split`    | `str` | `"devtest"`  | Сплит датасета (`"dev"` / `"devtest"`).   |

**Возвращает** кортеж `(src_sentences, ref_sentences)`.

```python
srcs, refs = load_flores("rus_Cyrl", "eng_Latn", split="dev")
```

---

## 5. raw_use.py — инференс PyTorch-модели

Модуль `raw_use.py` реализует интерактивный перевод в консоли с использованием обученной PyTorch-модели. Путь к контрольной точке, имя файла и устройство указываются в начале блока `__main__`; программа работает в цикле до `Ctrl+C`.

---

### 5.1 beam_search

**Функция.** Поиск лучом с штрафом за повторы: логиты уже выданных токенов делятся (или умножаются) на `repetition_penalty`, что подавляет циклическую генерацию. Луч выбирается по наибольшей средней лог-вероятности на токен.

| Аргумент             | Тип            | По умолчанию | Описание                              |
| -------------------- | -------------- | ------------ | ------------------------------------- |
| `transformer`        | `Transformer`  | —            | Модель.                               |
| `tokenizer`          | `AutoTokenizer`| —            | Токенизатор.                          |
| `src`                | `Tensor`       | —            | Id входных токенов `(1, seq)`.        |
| `src_mask`           | `Tensor`       | —            | Маска не-pad позиций.                 |
| `beam_size`          | `int`          | `5`          | Ширина луча.                          |
| `max_len`            | `int`          | `256`        | Максимальная длина перевода.          |
| `repetition_penalty` | `float`        | `1.3`        | Штраф за повтор уже выданных токенов. |
| `device`             | `str`          | `"cpu"`      | Устройство вычислений.                |

**Возвращает** декодированный перевод (`str`).

```python
import torch
from transformers import AutoTokenizer
from imp import Transformer
from raw_use import beam_search

tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
ckpt      = torch.load("./t5s/transformer.pt", weights_only=False)
c         = ckpt["config"]
model     = Transformer(dim=c["d_model"],
                        enc_num_tokens=c["vocab_size"], enc_depth=c["num_layers"],
                        enc_heads=c["num_heads"], enc_dim_head=c["dim_head"],
                        enc_mlp_mult=c["mlp_mult"],
                        dec_num_tokens=c["vocab_size"],
                        dec_depth=c["num_layers"] + c["dec_depth_diff"],
                        dec_heads=c["num_heads"], dec_dim_head=c["dim_head"],
                        dec_mlp_mult=c["mlp_mult"], dropout=c["dropout"],
                        tie_token_emb=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

src = tokenizer("Добрый день", return_tensors="pt", padding="max_length",
                max_length=512, truncation=True)["input_ids"]
print(beam_search(model, tokenizer, src, src.ne(3), beam_size=5))
```

---

## 6. onnx_export.py — экспорт и квантизация модели

Модуль `onnx_export.py` преобразует контрольную точку в модель, готовую к распространению: кодер и декодер экспортируются в ONNX (FP32) с динамическими размерами, после чего применяется избирательная квантизация по группам весов (см. `QUANT_CONFIG` в [6.7](#67-apply_quantization_config)). В результате в директории сохранения появляются файлы `encoder_fp32.onnx`, `decoder_fp32.onnx`, `encoder_quant.onnx`, `decoder_quant.onnx` и `model_config.json`.

---

### 6.1 EncoderWrapper

**Класс.** Обёртка кодера для экспорта в ONNX: фиксирует сигнатуру `forward(src, src_mask) -> memory`.

| Аргумент  | Тип       | Описание                    |
| --------- | --------- | --------------------------- |
| `encoder` | `Encoder` | Кодер модели `Transformer`. |

```python
w      = EncoderWrapper(model.encoder)
memory = w(dummy_src, dummy_src_mask)
```

---

### 6.2 DecoderWrapper

**Класс.** Обёртка «декодер + логит-слой» для экспорта в ONNX: сигнатура `forward(tgt, memory, src_mask) -> logits`.

| Аргумент | Тип         | Описание                                     |
| -------- | ----------- | -------------------------------------------- |
| `model`  | `Transformer` | Полная модель, из которой берутся `decoder` и `to_logits`. |

```python
w      = DecoderWrapper(model)
logits = w(dummy_tgt, dummy_memory, dummy_mask)
```

---

### 6.3 export_fp32

**Функция.** Экспортирует кодер и декодер в файлы `encoder_fp32.onnx` и `decoder_fp32.onnx` с динамическими размерами batch и последовательности.

| Аргумент   | Тип         | Описание                          |
| ---------- | ----------- | --------------------------------- |
| `model`    | `Transformer` | Обученная модель.               |
| `config`   | `dict`      | Конфигурация модели (в том числе `vocab_size`, `d_model`). |
| `save_dir` | `str`       | Каталог для сохранения ONNX-файлов. |

```python
export_fp32(model, config, "onnx_export")
```

---

### 6.4 classify_nodes

**Функция.** Группирует узлы ONNX-модели по ролям в зависимости от имён операций: `attention`, `logits`, `embeddings`, `feedforward`. Группировка используется при выборочной квантизации.

| Аргумент    | Тип  | Описание                  |
| ----------- | ----- | ------------------------- |
| `onnx_path` | `str` | Путь к ONNX-модели.       |

**Возвращает** словарь из четырёх списков имён узлов.

```python
groups = classify_nodes("onnx_export/encoder_fp32.onnx")
print(len(groups["feedforward"]))
```

---

### 6.5 quantize_int8

**Функция.** Динамическая INT8-квантизация выбранного набора узлов (`onnxruntime.quantization.quantize_dynamic`); при `per_channel=True` применяется поканальная квантизация весов.

| Аргумент            | Тип        | По умолчанию | Описание                        |
| ------------------- | ----------- | ------------ | ------------------------------- |
| `input_path`        | `str`       | —            | Исходная ONNX-модель.           |
| `output_path`       | `str`       | —            | Путь к файлу результата.        |
| `nodes_to_quantize` | `list[str]` | —            | Имена узлов для квантизации.    |
| `per_channel`       | `bool`      | `True`       | Поканальная квантизация весов.  |

```python
quantize_int8("enc_fp32.onnx", "enc_int8.onnx",
              nodes_to_quantize=groups["feedforward"])
```

---

### 6.6 quantize_fp16

**Функция.** Переводит только указанные узлы в пониженную точность FP16 (half precision), остальные сохраняет в FP32 (через пакет `onnxconverter-common`).

| Аргумент     | Тип        | Описание                                  |
| ------------ | ----------- | ----------------------------------------- |
| `input_path` | `str`       | Исходная ONNX-модель.                     |
| `output_path`| `str`       | Путь к файлу результата.                  |
| `target_nodes`| `list[str]`| Имена узлов, подлежащих переводу в FP16.  |

```python
quantize_fp16("enc_int8.onnx", "enc_mixed.onnx",
              target_nodes=groups["embeddings"])
```

---

### 6.7 apply_quantization_config

**Функция.** Применяет смешанную квантизацию согласно конфигурации: сначала в INT8 переводятся группы, указанные в `config` с режимом `"int8"`, затем в FP16 — группы с режимом `"fp16"`; группы с режимом `"fp32"` не изменяются. Промежуточные файлы размещаются во временном каталоге, итоговый результат копируется в `output_path`.

| Аргумент    | Тип          | Описание                                          |
| ----------- | ------------- | ------------------------------------------------- |
| `fp32_path` | `str`         | Исходная ONNX-модель в FP32.                      |
| `output_path`| `str`        | Путь к итоговому файлу.                           |
| `config`    | `dict[str, str]` | Соответствие «группа узлов → режим» (`int8`/`fp16`/`fp32`). |
| `tmp_dir`   | `str`         | Временный каталог для промежуточных файлов.       |

Пример конфигурации (глобальная константа `QUANT_CONFIG`):

```python
QUANT_CONFIG = {
    "attention":   "fp32",
    "feedforward": "int8",
    "logits":      "fp16",
    "embeddings":  "fp16",
}
```

```python
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    apply_quantization_config("onnx_export/encoder_fp32.onnx",
                              "onnx_export/encoder_quant.onnx",
                              QUANT_CONFIG, tmp)
```

---

### 6.8 save_model_config

**Функция.** Сохраняет файл `model_config.json` с метаданными модели: языки, версия, архитектура и схема квантизации. Файл читается приложениями при отображении имени модели и сервером раздачи (см. [8.2](#82-zip_is_valid)).

| Аргумент       | Тип    | Описание                        |
| -------------- | ------- | ------------------------------- |
| `save_dir`     | `str`   | Каталог сохранения.             |
| `config`       | `dict`  | Конфигурация архитектуры модели.|
| `quant_config` | `dict`  | Схема квантизации.              |

```python
save_model_config("onnx_export", config, QUANT_CONFIG)
```

---

### 6.9 load_model

**Функция.** Восстанавливает модель `Transformer` по конфигурации из контрольной точки и загружает в неё обученные веса.

| Аргумент          | Тип  | Описание                        |
| ----------------- | ----- | ------------------------------- |
| `checkpoint_path` | `str` | Путь к файлу контрольной точки (`.pt`). |

**Возвращает** кортеж `(model, config)`.

```python
model, config = load_model("./transformer.pt")
```

---

## 7. onnx_use.py — инференс ONNX в консоли

Модуль `onnx_use.py` реализует интерактивный перевод в консоли с использованием ONNX Runtime. Особенность — стриминговый вывод: частичный перевод отображается по мере генерации, после чего под ним выводится строка статистики производительности.

---

### 7.1 np_softmax

**Функция.** Численно устойчивый softmax по последнему измерению: перед экспонированием из входных значений вычитается их максимум.

```python
import numpy as np
from onnx_use import np_softmax

p = np_softmax(np.array([[1.0, 2.0, 3.0]]))
assert abs(p.sum() - 1.0) < 1e-6
```

---

### 7.2 ONNXTransformer

**Класс.** Обёртка над ONNX-сессиями кодера и декодера для инференса на CPU (провайдер `CPUExecutionProvider`).

| Аргумент       | Тип  | Описание                      |
| -------------- | ----- | ----------------------------- |
| `encoder_path` | `str` | Путь к файлу `encoder.onnx`.  |
| `decoder_path` | `str` | Путь к файлу `decoder.onnx`.  |

Методы:

- `encode(src, src_mask)` — прогоняет исходную последовательность через кодер; `src` — массив `int64` `(batch, seq)`, маска — `bool`; возвращает память кодера `float32` `(batch, seq, d_model)`;
- `decode(tgt, memory, src_mask)` — прогоняет целевую последовательность через декодер с логит-слоем; возвращает логиты `(batch, tgt_seq, vocab_size)`.

```python
import numpy as np
from onnx_use import ONNXTransformer
from transformers import AutoTokenizer

tok   = AutoTokenizer.from_pretrained("./tokenizer/")
model = ONNXTransformer("./encoder.onnx", "./decoder.onnx")

src      = tok("Привет", return_tensors="pt", padding="max_length",
               max_length=64, truncation=True)["input_ids"]
src_mask = src.ne(3).numpy()
memory   = model.encode(src.numpy(), src_mask)
logits   = model.decode(np.array([[0]], dtype=np.int64), memory, src_mask)
```

---

### 7.3 beam_search_stream

**Функция.** Стриминговый поиск лучом: генератор, на каждом шаге которого возвращает текущий лучший частичный перевод. По завершении (все лучи завершены или достигнута `max_len`) итоговая статистика передаётся в атрибуте `value` исключения `StopIteration`.

| Аргумент    | Тип              | По умолчанию | Описание                     |
| ----------- | ----------------- | ------------ | ---------------------------- |
| `model`     | `ONNXTransformer` | —            | Модель.                      |
| `tokenizer` | `AutoTokenizer`   | —            | Токенизатор.                 |
| `src`       | `Tensor` или `ndarray` | —       | Id входных токенов.          |
| `src_mask`  | `ndarray` (`bool`)| —            | Маска не-pad позиций.        |
| `beam_size` | `int`             | `4`          | Ширина луча.                 |
| `max_len`   | `int`             | `128`        | Максимальная длина перевода. |

**Возвращает** (при завершении) кортеж `(full_text, num_out_tokens, num_chars, elapsed)`.

```python
gen = beam_search_stream(model, tok, src, src_mask, beam_size=1)
try:
    while True:
        partial = next(gen)
        print("\r" + partial, end="", flush=True)
except StopIteration as e:
    text, n_tok, n_chr, elapsed = e.value
```

---

### 7.4 print_stats

**Функция.** Выводит в консоль строку статистики производительности: время генерации, скорость в токенах и символах в секунду, параметры входа и выхода.

| Аргумент        | Тип    | Описание                          |
| --------------- | ------- | --------------------------------- |
| `num_in_tokens` | `int`   | Число входных токенов.            |
| `padding`       | `int`   | Длина последовательности после padding. |
| `num_out_tokens`| `int`   | Число выданных токенов.           |
| `num_chars`     | `int`   | Число символов перевода.          |
| `elapsed`       | `float` | Время генерации (секунды).        |

```python
print_stats(num_in_tokens=12, padding=64, num_out_tokens=9,
            num_chars=41, elapsed=0.42)
# [0.42 с | 21.4 tok/s | 97.6 chr/s | вход: 12 tok (pad 64) | выход: 9 tok, 41 chr]
```

---

## 8. model_dist_server.py — сервер раздачи моделей

Модуль `model_dist_server.py` реализует сервер на базе FastAPI, который сканирует директорию с zip-архивами моделей и предоставляет REST API для их перечисления и скачивания мобильным приложением. Директория задаётся переменной окружения `MODELS_DIR` (по умолчанию `./onnx_export/models`); порт — `9100`.

---

### 8.1 ModelInfo

**Класс.** Схема (Pydantic) описания одной доступной к скачиванию модели; используется как тип ответа эндпоинта `/models`.

| Поле               | Тип    | По умолчанию | Описание                          |
| ------------------ | ------- | ------------ | --------------------------------- |
| `name`             | `str`   | —            | Отображаемое имя модели.          |
| `file`             | `str`   | —            | Имя zip-архива.                   |
| `size_mb`          | `int`   | —            | Размер архива в мегабайтах.       |
| `input_language`   | `str`   | `""`         | Язык-источник.                    |
| `output_language`  | `str`   | `""`         | Язык перевода.                    |
| `bidirectional`    | `bool`  | `False`      | Двунаправленность модели.         |

```python
info = ModelInfo(name="RU <-> EN", file="ru-en-0.1.zip", size_mb=120,
                 input_language="RU", output_language="EN",
                 bidirectional=True)
```

---

### 8.2 zip_is_valid

**Функция.** Проверяет, что zip-архив является корректным архивом моделей: читается и содержит все обязательные файлы (`encoder.onnx`, `decoder.onnx`, `tokenizer/tokenizer.json`, `model_config.json`).

| Аргумент | Тип   | Описание                |
| -------- | ------ | ----------------------- |
| `path`   | `Path` | Путь к zip-файлу.       |

**Возвращает** `True`, если архив валиден.

```python
ok = zip_is_valid(Path("onnx_export/models/ru-en-0.1.zip"))
```

---

### 8.3 read_zip_model_config

**Функция.** Читает файл `model_config.json` из архива модели без его распаковки.

| Аргумент | Тип   | Описание                |
| -------- | ------ | ----------------------- |
| `path`   | `Path` | Путь к zip-файлу.       |

**Возвращает** конфигурацию в виде словаря; при отсутствии файла или ошибке чтения — пустой словарь.

```python
cfg = read_zip_model_config(Path("ru-en-0.1.zip"))
print(cfg.get("input_language"))
```

---

### 8.4 make_display_name

**Функция.** Формирует человекочитаемое имя модели: при наличии языков в конфигурации — строка вида `RU <-> EN` (или `RU -> EN` для однонаправленной модели), в противном случае — разбор имени архива (например, `ru-en-base` → `RU -> EN Base`).

| Аргумент | Тип   | Описание                          |
| -------- | ------ | --------------------------------- |
| `stem`   | `str`  | Имя архива без расширения.        |
| `cfg`    | `dict` | Конфигурация из архива.           |

```python
make_display_name("ru-en-0.1", {"input_language": "RU",
                                "output_language": "EN",
                                "bidirectional": True})
# 'RU <-> EN'
```

---

### 8.5 Эндпоинты

| Метод | Путь                 | Обработчик          | Описание                              |
| ----- | -------------------- | ------------------- | ------------------------------------- |
| GET   | `/models`            | `list_models()`     | Список валидных моделей (JSON).       |
| GET   | `/models/{filename}` | `download_model()`  | Скачивание zip-архива модели; 400 при недопустимом имени (защита от path traversal), 404 при отсутствии. |
| GET   | `/ping`              | `ping()`            | Проверка доступности: `{"answer": "available"}`. |
| GET   | `/favicon.ico`       | `favicon()`         | Заглушка иконки (ответ 204).          |

```bash
python model_dist_server.py
```

---

## 9. web.py — веб-переводчик

Модуль `web.py` реализует веб-интерфейс перевода на базе Flask: две панели текста (исходный и перевод) с живым переводом по мере ввода, выбор языковой модели, раздел «О проекте» и ссылка на скачивание мобильного приложения. Модели загружаются из поддиректорий каталога `MODELS_DIR` (по умолчанию `./onnx_export/for_web`) и кэшируются в памяти.

---

### 9.1 np_softmax

**Функция.** Численно устойчивый softmax по последнему измерению (аналог [7.1](#71-np_softmax)).

---

### 9.2 list_model_dirs

**Функция.** Возвращает список имён каталогов моделей в `MODELS_DIR` в алфавитном порядке.

```python
dirs = list_model_dirs()   # ["ru-en-0.1", "ru-en-0.2"]
```

---

### 9.3 read_model_config

**Функция.** Читает файл `model_config.json` каталога модели.

| Аргумент         | Тип  | Описание                                |
| ---------------- | ----- | --------------------------------------- |
| `model_dir_name` | `str` | Имя каталога модели внутри `MODELS_DIR`.|

**Возвращает** конфигурацию в виде словаря; при отсутствии или повреждении файла — пустой словарь.

```python
cfg = read_model_config("ru-en-0.1")
```

---

### 9.4 make_display_name

**Функция.** Формирует отображаемое имя модели для выпадающего списка: `SRC -> TGT` (или `SRC <-> TGT` для двунаправленной) из конфигурации, иначе — имя каталога.

| Аргумент   | Тип   | Описание                                |
| ---------- | ------ | --------------------------------------- |
| `dir_name` | `str`  | Имя каталога модели.                    |
| `cfg`      | `dict` | Конфигурация модели.                    |

```python
make_display_name("ru-en-0.1", {"input_language": "ru",
                                "output_language": "en"})
# 'RU -> EN'
```

---

### 9.5 ONNXTransformer

**Класс.** Обёртка над ONNX-сессиями одной модели, загружаемой из каталога.

| Аргумент    | Тип  | Описание                                  |
| ----------- | ----- | ----------------------------------------- |
| `model_dir` | `str` | Каталог с `encoder.onnx`, `decoder.onnx` и токенизатором. |

Методы `encode(src, src_mask)` и `decode(tgt, memory, src_mask)` совпадают по семантике с методами класса `ONNXTransformer` из модуля `onnx_use.py` (см. [7.2](#72-onnxtransformer)).

```python
model  = ONNXTransformer("./onnx_export/for_web/ru-en-0.1")
memory = model.encode(src_np, src_mask)
logits = model.decode(tgt_np, memory, src_mask)
```

---

### 9.6 get_model

**Функция.** Ленивая загрузка модели с кэшированием: при первом обращении к модели считываются токенизатор и ONNX-сессии, последующие обращения обслуживаются из кэша.

| Аргумент | Тип  | Описание                    |
| -------- | ----- | --------------------------- |
| `name`   | `str` | Имя каталога модели.        |

**Возвращает** кортеж `(tokenizer, model)`.

```python
tokenizer, model = get_model("ru-en-0.1")
```

---

### 9.7 beam_search_onnx

**Функция.** Поиск лучом по ONNX-модели: вход кодируется один раз, генерация продолжается до завершения всех лучей или достижения `max_len`, после чего лучший луч декодируется в текст.

| Аргумент    | Тип              | По умолчанию | Описание               |
| ----------- | ----------------- | ------------ | ---------------------- |
| `model`     | `ONNXTransformer` | —            | Модель.                |
| `tokenizer` | `AutoTokenizer`   | —            | Токенизатор.           |
| `src`       | `Tensor` или `ndarray` | —       | Id входных токенов.    |
| `beam_size` | `int`             | `4`          | Ширина луча.           |
| `max_len`   | `int`             | `128`        | Максимальная длина перевода. |

**Возвращает** итоговый перевод (`str`).

```python
src = tokenizer("Привет", return_tensors="pt",
                padding="max_length", max_length=512)["input_ids"]
print(beam_search_onnx(model, tokenizer, src))
```

---

### 9.8 Эндпоинты

| Метод | Путь              | Обработчик      | Описание                                    |
| ----- | ----------------- | --------------- | ------------------------------------------- |
| GET   | `/`               | `index()`       | Главная страница (встроенный HTML-шаблон).  |
| GET   | `/logo`           | `logo()`        | Логотип `polylogo.png`.                     |
| GET   | `/api/models`     | `api_models()`  | Список моделей: `{"models": [{"id", "name"}]}`. |
| POST  | `/api/translate`  | `api_translate()` | Перевод текста; тело — `{"text", "model"}`; ответ — `{"translation"}`. Ошибки: 400 (модель не выбрана), 404 (модель не найдена), 500 (сбой загрузки). |
| GET   | `/api/news`       | `api_news()`    | Карточки раздела «О проекте».               |
| GET   | `/favicon.ico`    | `favicon()`     | Иконка сайта (`polylogo.ico`).              |

```bash
python web.py
```

---

## 10. telegram.py — Telegram-бот

Модуль `telegram.py` реализует Telegram-бота на базе библиотеки aiogram: бот переводит любое входящее сообщение с помощью ONNX-модели и возвращает результат в чат. Поддерживается подключение через MTProto-прокси с автоматической проверкой доступности и переключением между прокси при сбоях. Настройки задаются в глобальной константе `CONFIG` (токен бота, список прокси).

---

### 10.1 ProxyConfig

**Класс.** Описание одного MTProto-прокси для подключения бота к Telegram.

| Аргумент | Тип        | По умолчанию | Описание                  |
| -------- | ----------- | ------------ | ------------------------- |
| `host`   | `str`       | —            | Адрес прокси.             |
| `port`   | `int`       | —            | Порт.                     |
| `secret` | `str` или `None` | `None`   | Секретный ключ прокси.    |

Метод `get_proxy_dict()` возвращает словарь параметров (`scheme`, `host`, `port` и, при наличии, `secret`), ожидаемый конструктором `Bot(proxy=...)`.

```python
p = ProxyConfig("proxy.example.com", 1234, secret="key")
print(p.get_proxy_dict())
# {'scheme': 'mtproto', 'host': 'proxy.example.com', 'port': 1234, 'secret': 'key'}
```

---

### 10.2 ProxyManager

**Класс.** Управление набором прокси: отслеживает доступность каждого, ведёт счётчик текущего активного прокси и переключается на следующий при сбое.

| Аргумент | Тип                | Описание                                |
| -------- | ------------------- | --------------------------------------- |
| `proxies`| `list[ProxyConfig]` | Список прокси в порядке переключения.   |

Методы:

- `get_current_proxy()` — возвращает текущий активный прокси (`None`, если список пуст);
- `get_current_proxy_dict()` — текущий прокси в виде словаря aiogram;
- `await check_proxy(proxy)` — проверка доступности: открывается и сразу закрывается тестовая сессия бота через данный прокси;
- `await validate_proxies()` — проверка всех прокси с выводом статуса каждого; возвращает `True`, если доступен хотя бы один;
- `switch_proxy()` — переключение на следующий доступный прокси (циклически); при отсутствии доступных выбрасывается `RuntimeError`.

```python
mgr  = ProxyManager([ProxyConfig("host1", 1),
                     ProxyConfig("host2", 2)])
ok   = await mgr.validate_proxies()
proxy = mgr.get_current_proxy_dict()
await dp.start_polling(bot, proxy=proxy)
```

---

### 10.3 ONNXTransformer

**Класс.** Обёртка над ONNX-сессиями кодера и декодера для инференса в боте; по семантике совпадает с классом `ONNXTransformer` из модуля `onnx_use.py` (см. [7.2](#72-onnxtransformer)), отличается дополнительным зарезервированным параметром `device` в конструкторе.

```python
model = ONNXTransformer(encoder_path="encoder.onnx",
                        decoder_path="decoder.onnx")
```

---

### 10.4 beam_search_onnx

**Функция.** Поиск лучом по ONNX-модели: вход кодируется один раз, генерация продолжается до завершения всех лучей или достижения `max_len`; лучший луч декодируется в текст.

| Аргумент    | Тип              | По умолчанию | Описание               |
| ----------- | ----------------- | ------------ | ---------------------- |
| `model`     | `ONNXTransformer` | —            | Модель.                |
| `tokenizer` | `AutoTokenizer`   | —            | Токенизатор.           |
| `src`       | `Tensor`          | —            | Id входных токенов `(1, seq)`. |
| `beam_size` | `int`             | `4`          | Ширина луча.           |
| `max_len`   | `int`             | `256`        | Максимальная длина перевода. |

**Возвращает** итоговый перевод (`str`).

```python
src = tokenizer(text, truncation=True, padding="max_length",
                max_length=512, return_tensors="pt")["input_ids"].to(device)
translation = beam_search_onnx(model, tokenizer, src, beam_size=4)
```

---

### 10.5 Хендлеры

| Функция     | Триггер                | Назначение                                                        |
| ----------- | ---------------------- | ----------------------------------------------------------------- |
| `start(message)` | команда `/start`  | Присылает пользователю подсказку.                                 |
| `translate(message)` | любое сообщение  | Переводит текст; при ошибке переключает прокси и сообщает о сбое. |

```python
# Регистрация выполняется декораторами при импорте модуля:
@dp.message(CommandStart())
async def start(message: Message): ...

@dp.message()
async def translate(message: Message): ...
```

---

### 10.6 main

**Функция.** Точка входа бота: загружает токенизатор и ONNX-модель, формирует список прокси из `CONFIG`, при необходимости проверяет их доступность и запускает long polling.

```bash
python telegram.py
```

> **Примечание.** Перед запуском в конфиге `CONFIG` необходимо указать токен API бота и (опционально) данные MTProto-прокси.

---

## 11. desktop_app.py — десктопное приложение

Модуль `desktop_app.py` реализует десктопное приложение на базе PyQt5 — аналог мобильного приложения для ПК и ноутбуков. Приложение состоит из трёх вкладок: «Перевод» (локальный инференс ONNX-модели с построчным выводом), «Загрузки» (скачивание и удаление языковых пакетов с сервера раздачи) и «О проекте». Языковые пакеты хранятся в директории `./.models/`, адрес сервера по умолчанию — `http://igorpet.ru:9100`.

---

### 11.1 Фабрики виджетов

Группа функций, создающих стилизованные виджеты в фирменном тёмно-зелёном оформлении приложения.

| Функция                    | Описание                                          |
| -------------------------- | ------------------------------------------------- |
| `accent_btn(text, small=False)`   | Заполненная кнопка основного действия.            |
| `outline_btn(text, color, small=False)` | Контурная кнопка второстепенного действия.    |
| `delete_btn(text)`               | Кнопка удаления (красная).                        |
| `nav_btn(text)`                  | Переключаемая кнопка нижней навигации.            |
| `neon_card()`                     | Рамка-«карточка» с фирменной обводкой.            |
| `text_edit_transparent()`         | Прозрачное многострочное поле ввода.              |

```python
from desktop_app import accent_btn

btn = accent_btn("Перевести")
btn.clicked.connect(on_translate)
```

---

### 11.2 RoundedPanel

**Класс.** Панель со скруглёнными углами: маска скругления пересчитывается при каждом изменении размера (метод `resizeEvent`).

| Аргумент | Тип          | Описание                    |
| -------- | ------------- | --------------------------- |
| `parent` | `QWidget` или `None` | Родительский виджет.    |

```python
panel = RoundedPanel()
panel.resizeEvent   # пересчёт маски вызывается автоматически
```

---

### 11.3 UnigramTokenizer

**Класс.** Собственный униграмм-токенизатор, работающий напрямую с файлом `tokenizer/tokenizer.json` без зависимости от библиотеки `transformers`: препре-токенизация по схеме Metaspace и декомпозиция слов в токены словаря алгоритмом Витерби.

| Аргумент    | Тип   | Описание                                              |
| ----------- | ------ | ----------------------------------------------------- |
| `model_dir` | `Path` | Каталог модели, содержащий `tokenizer/tokenizer.json`; при отсутствии файла выбрасывается `FileNotFoundError`. |

Атрибуты: идентификаторы служебных токенов `bos_id`, `eos_id`, `unk_id`, `pad_id`, константа `MAX_TOKEN_LEN = 32`.

Методы:

- `_viterbi(text)` — декомпозиция препре-токена в последовательность токенов словаря, максимизирующая суммарную лог-вероятность; при отсутствии разбивки возвращает список символов;
- `_pretokenize(text)` — препре-токенизация: пробелы заменяются маркером `▁`, строка нарезается на слова;
- `encode(text, max_length=256)` — кодирует текст в последовательность `[bos, ..., eos]` с padding до `max_length`; возвращает массив `int64`;
- `decode(ids, skip_special=True)` — декодирует последовательность id в текст, маркер `▁` преобразуется в пробел.

```python
from pathlib import Path
from desktop_app import UnigramTokenizer

tok = UnigramTokenizer(Path("./.models/ru-en-0.1"))
ids = tok.encode("Привет, мир", max_length=64)
print(tok.decode(ids))        # Привет, мир
print(ids[:5])                # [0, ..., 1, ...]
```

---

### 11.4 OnnxTransformer

**Класс.** Обёртка над ONNX-сессиями кодера и декодера модели из локального каталога; при отсутствии пакета `onnxruntime` конструктор выбрасывает `ImportError`.

| Аргумент    | Тип   | Описание                                      |
| ----------- | ------ | --------------------------------------------- |
| `model_dir` | `Path` | Каталог с файлами `encoder.onnx` и `decoder.onnx`. |

Методы `encode(src, src_mask)` и `decode(tgt, memory, src_mask)` совпадают по семантике с методами класса `ONNXTransformer` из модуля `onnx_use.py` (см. [7.2](#72-onnxtransformer)).

```python
model  = OnnxTransformer(Path("./.models/ru-en-0.1"))
memory = model.encode(src[np.newaxis, :], (src != 3)[np.newaxis, :])
logits = model.decode(tgt, memory, src_mask)
```

---

### 11.5 greedy_search

**Функция.** Жадная генерация перевода по одному токену: кодер прогоняется один раз, после чего на каждом шаге выбирается токен с максимальной вероятностью. Вызываемая функция `on_token(tokens)` срабатывает после каждого нового токена и используется для построчного вывода частичного перевода.

| Аргумент     | Тип                  | По умолчанию | Описание                          |
| ------------ | --------------------- | ------------ | --------------------------------- |
| `model`      | `OnnxTransformer`     | —            | Модель.                           |
| `tokenizer`  | `UnigramTokenizer`    | —            | Токенизатор.                      |
| `src_tokens` | `ndarray` (`int64`)   | —            | Id входных токенов.               |
| `max_len`    | `int`                 | `1024`       | Максимальная длина вывода.        |
| `on_token`   | `callable` или `None` | `None`       | Вызов на каждом новом токене.     |

**Возвращает** последовательность `[bos, ..., eos]` (`ndarray`, `int64`).

```python
out = greedy_search(model, tok, src_tokens, max_len=1024,
                    on_token=lambda t: print(tok.decode(t), end="\r"))
print(tok.decode(out))
```

---

### 11.6 read_model_config

**Функция.** Читает файл `model_config.json` каталога модели.

| Аргумент    | Тип   | Описание                |
| ----------- | ------ | ----------------------- |
| `model_dir` | `Path` | Каталог модели.         |

**Возвращает** конфигурацию в виде словаря; при отсутствии или повреждении файла — пустой словарь.

```python
cfg = read_model_config(Path("./.models/ru-en-0.1"))
```

---

### 11.7 make_display_name

**Функция.** Формирует отображаемое имя модели для выпадающего списка: при наличии языков в конфигурации — строка вида `RU -> EN` (или `RU <-> EN`), в противном случае — разбор имени каталога (например, `ru-en-0.1` → `RU -> EN 0.1`).

| Аргумент | Тип  | Описание                                    |
| -------- | ----- | ------------------------------------------- |
| `stem`   | `str` | Имя каталога модели (пустая строка — корень). |

```python
make_display_name("ru-en-0.1")   # 'RU -> EN 0.1'
```

---

### 11.8 ModelDownloadManager

**Класс.** Сетевой менеджер загрузки моделей: проверка доступности сервера, получение списка моделей, скачивание и распаковка архивов. Адрес сервера хранится в классовом атрибуте `base_url` (по умолчанию `http://igorpet.ru:9100`).

| Метод (классовый)                              | Описание                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------------ |
| `ping(url=None)`                               | Проверка доступности по эндпоинту `/ping`; возвращает `True`, если сервер ответил `available`. |
| `fetch_model_list()`                           | Запрос списка доступных моделей; возвращает JSON-массив описаний.        |
| `download_model(file, dest_dir, on_progress=None)` | Скачивание архива (пакетами по 8 КБ), распаковка в каталог и удаление архива; `on_progress(pct, installing)` — callback прогресса. |

```python
from pathlib import Path
from desktop_app import ModelDownloadManager

if ModelDownloadManager.ping():
    for m in ModelDownloadManager.fetch_model_list():
        print(m["name"], m["size_mb"], "MB")
    ModelDownloadManager.download_model(
        "ru-en-0.1.zip", Path("./.models/"),
        on_progress=lambda p, i: print(p, "%"))
```

---

### 11.9 Фоновые треды

Все длительные операции (генерация, скачивание, запрос списка, проверка адреса) выполняются в фоновых тредях `QThread`, которые общаются с интерфейсом через сигналы.

| Класс              | Конструктор                  | Сигналы                                                        |
| ------------------ | ---------------------------- | -------------------------------------------------------------- |
| `InferenceWorker`  | `(model, tokenizer, text)`   | `partial_result(str)`; `finished(str, float, int)`; `error(str)`. |
| `DownloadWorker`   | `(file, dest_dir)`           | `progress(str, int, bool)`; `finished(str)`; `error(str, str)`. |
| `FetchListWorker`  | —                            | `result(list)`; `error(str)`.                                  |
| `PingWorker`       | `(url)`                      | `result(bool, str)`.                                           |

```python
from desktop_app import InferenceWorker

worker = InferenceWorker(model, tok, "Привет")
worker.partial_result.connect(lambda s: print(s, end="\r"))
worker.finished.connect(lambda t, e, n:
                        print(f"\n{t}  ({e:.2f} c, {n} tok)"))
worker.error.connect(lambda m: print("Ошибка:", m))
worker.start()
```

---

### 11.10 ModelCard

**Класс.** Карточка модели в списке загрузок: имя, размер, прогресс-бар и кнопки «Скачать» / «Удалить». Видимость кнопок зависит от статуса установки.

| Аргумент     | Тип              | Описание                          |
| ------------ | ----------------- | --------------------------------- |
| `model_info` | `dict`            | Данные модели (`name`, `file`, `size_mb`). |
| `installed`  | `bool`            | Установлена ли модель.            |
| `parent`     | `QWidget` или `None` | Родительский виджет.          |

Сигналы: `download_clicked(dict)`, `delete_clicked(dict)`.

Методы: `set_progress(pct, installing)` — отображение прогресса загрузки/установки; `mark_installed()` — отметка об установке; `mark_deleted()` — отметка об удалении.

```python
from desktop_app import ModelCard

card = ModelCard({"name": "RU <-> EN", "file": "ru-en-0.1.zip",
                  "size_mb": 120}, installed=False)
card.download_clicked.connect(start_download)
card.set_progress(37, installing=False)
card.mark_installed()
```

---

### 11.11 TranslateScreen

**Класс.** Экран перевода: поле ввода (лимит 2000 символов со счётчиком и цветовой индикацией), поле результата, выбор языкового пакета и кнопка «Перевести». При создании загружается первая найденная модель.

Ключевые методы:

- `refresh_models()` — перечитывает каталог моделей, обновляет выпадающий список и загружает первую модель;
- `_load_model(path)` — загружает токенизатор и ONNX-модель из каталога;
- `_do_translate()` — запускает фоновый тред `InferenceWorker` (см. [11.9](#119-фоновые-треды));
- `_on_done(text, elapsed, tokens)` — завершение: выводит результат и скорость в токенах в секунду.

```python
screen = TranslateScreen()
screen.refresh_models()
```

---

### 11.12 DownloadsScreen

**Класс.** Экран загрузок: статус соединения с сервером, список карточек моделей и блок настройки адреса сервера.

Сигнал: `models_changed` — состав моделей изменился (экран перевода перечитывает каталог).

Ключевые методы:

- `load()` — перезагружает экран: показывает установленные модели и запрашивает список у сервера;
- `_start_download(info)` — запускает фоновое скачивание;
- `_delete_model(info)` — удаление модели (каталог удаляется после подтверждения);
- `_on_confirm_url()` — проверка введённого адреса сервера (ping в фоне) перед сохранением;
- `_on_reset_url()` — сброс адреса к значению по умолчанию.

---

### 11.13 AboutScreen

**Класс.** Экран «О проекте»: описание платформы, ссылки для пользователей и разработчиков, лицензия.

---

### 11.14 Header

**Класс.** Верхняя панель окна: логотип (`polylogo.png`, при отсутствии файла — текстовая заглушка), фиксированная высота 108 пикселей.

---

### 11.15 NavBar

**Класс.** Нижняя навигационная панель с тремя вкладками: «Перевод», «Загрузки», «О проекте».

Сигнал: `tab_changed(int)` — индекс выбранной вкладки.

```python
nav = NavBar()
nav.tab_changed.connect(on_tab_changed)
```

---

### 11.16 MainWindow

**Класс.** Главное окно: шапка, три экрана в `QStackedWidget` и нижняя навигация. Сигнал `models_changed` экрана загрузок связан с перечитыванием списка моделей на экране перевода; при переходе на вкладку «Загрузки» список пересчитывается.

---

### 11.17 main

**Функция.** Точка входа приложения: инициализация `QApplication`, стиль Fusion, палитра и глобальный лист стилей, показ главного окна. При отсутствии пакета `onnxruntime` выводится предупреждение о недоступности перевода.

```bash
python desktop_app.py
```

---

## 12. tok_learn.py — обучение униграмм-токенизатора

Модуль `tok_learn.py` обучает униграмм-токенизатор (библиотека `tokenizers`) на токенизированных данных из глобального датасета `dataset`. Результатом являются файлы `tokenizer/tokenizer.json` и `tokenizer/tokenizer_config.json`, сохраняемые в директорию `tokenizer/`.

---

### 12.1 get_training_corpus

**Функция.** Генератор корпуса для обучения: порциями выдаёт токенизированные последовательности `input` и `output` глобального датасета.

| Аргумент | Тип  | По умолчанию | Описание                        |
| -------- | ----- | ------------ | ------------------------------- |
| `batch`  | `int` | `10`         | Размер порции записей.          |

```python
from datasets import load_from_disk
from itertools import islice
import tok_learn

tok_learn.dataset = load_from_disk("./sources/s512_clear")

chunks = list(islice(tok_learn.get_training_corpus(batch=1000), 2))
print(len(chunks[0]))   # 1000 последовательностей
```

> **Примечание.** Обучение, установка постпроцессора и декодера Metaspace, а также упаковка в `PreTrainedTokenizerFast` выполняются на уровне модуля при запуске скрипта: `python tok_learn.py`.

---

## 13. tok_learn_incase.py — токенизатор с inline-регистром

Модуль `tok_learn_incase.py` реализует экспериментальный (beta) вариант токенизатора с inline-обозначением регистра: вместо хранения в словаре отдельных вхождений для строчного и заглавного написания слова текст приводится к нижнему регистру, а информация о регистре кодируется служебными токенами (`<|upall|>` — слово целиком в верхнем регистре, `<|up|>` — последовательность заглавных, `<|cap|>` — одиночная заглавная). Обучение ведётся на стриминговом корпусе `wmt/wmt19`. Направление рассматривается как потенциальная замена текущего униграмм-токенизатора (см. раздел 6.2 `README.md`).

---

### 13.1 apply_inline_casing

**Функция.** Кодирует регистр текста служебными токенами: если все буквы слова заглавные, перед ним ставится `<|upall|>`; последовательности заглавных букв заменяются на `<|up|>` плюс нижний регистр; одиночные заглавные — на `<|cap|>` плюс нижний регистр.

| Аргумент | Тип  | Описание          |
| -------- | ----- | ----------------- |
| `text`   | `str` | Исходная строка.  |

**Возвращает** строку с закодированным регистром.

```python
from tok_learn_incase import apply_inline_casing

apply_inline_casing("ВНИМАНИЕ, ПРОВЕРКА СВЯЗИ!")
# '<|upall|> внимание, проверка связи!'
apply_inline_casing("Смотрю YouTube и покупаю на eBay.")
# 'Смотрю <|up|>youtube и покупаю на <|cap|>ebay.'
```

---

### 13.2 restore_inline_casing

**Функция.** Обратное преобразование функции `apply_inline_casing`: восстанавливает исходный регистр текста по маркерам.

| Аргумент | Тип  | Описание                        |
| -------- | ----- | ------------------------------- |
| `text`   | `str` | Строка с токенами регистра.     |

**Возвращает** строку с восстановленным регистром.

```python
from tok_learn_incase import apply_inline_casing, restore_inline_casing

s = apply_inline_casing("Привет, Мир!")
print(restore_inline_casing(s))   # Привет, Мир!
```

---

### 13.3 get_training_corpus

**Функция.** Генератор корпуса для обучения: порциями выдаёт тексты оригинала и перевода (поля `og_full_text`/`translated_text` стримингового датасета `wmt/wmt19`) с закодированным регистром.

| Аргумент | Тип  | По умолчанию | Описание                        |
| -------- | ----- | ------------ | ------------------------------- |
| `batch`  | `int` | `10`         | Размер порции записей.          |

```python
chunks = list(islice(get_training_corpus(batch=100), 2))
print(chunks[0][0])   # первый текст порции с токенами регистра
```

> **Примечание.** Обучение, проверка на тестовых строках и сохранение токенизатора выполняются на уровне модуля при запуске скрипта: `python tok_learn_incase.py`.

---

_Проект распространяется под лицензией AGPLv3. Репозиторий открыт для вкладов от сообщества._
