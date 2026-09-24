---
title: Tổng quan kiến trúc
description: Ba dạng sản phẩm của Unsloth (Desktop, Studio, Core), CLI, API và phân biệt inference với training.
---

# Tổng quan kiến trúc

## Ba dạng sản phẩm

Docs cài đặt của Unsloth nêu ba cách dùng tách biệt:

1. **Unsloth Desktop**: app native (ứng dụng cài trực tiếp lên hệ điều hành), xây trên Tauri, miễn phí, mã nguồn mở, đang ở bản Beta. GitHub README ghi đây là cách được khuyến nghị.
2. **Unsloth Studio**: web UI (giao diện chạy trên trình duyệt), no-code (không cần viết code), bản Beta. Cài thủ công bằng script rồi mở qua trình duyệt.
3. **Unsloth Core**: gói Python gốc, dùng code để training và inference.

### Unsloth Desktop

- Tải bộ cài cho macOS (`.dmg`), Windows (`.exe`), Linux (`.deb` x64/ARM64, AppImage) từ https://unsloth.ai/download hoặc GitHub Releases.
- Sau khi mở app: vào dropdown "Select model" hoặc tab "Model hub", chọn model và mức quantization vừa với máy, tải về, rồi chat.
- Tính năng docs liệt kê: chat với tool calling "tự sửa lỗi" (self-healing, docs ghi chính xác hơn 50%), chạy Bash/Python trong môi trường cô lập, web search và deep research, sinh/train model diffusion ảnh/video, audio (TTS, Whisper, Qwen3-ASR), train không cần code, phục vụ model qua LAN hoặc Cloudflare HTTPS, nối nhà cung cấp cloud (OpenAI, Anthropic, Ollama, vLLM...).
- Có cơ chế cấp quyền: model không được đọc/sửa file hay truy cập internet nếu chưa được bạn cho phép.

### Unsloth Studio

- Mô tả trong docs: GUI local trên trình duyệt để fine-tune LLM không cần code, lo phần nạp model, định dạng dataset, cấu hình hyperparameter (siêu tham số) và theo dõi training trực tiếp.
- Trang Studio ghi cách dễ nhất để có Studio là cài app Desktop; lệnh cài thủ công chỉ dành cho ai muốn tự cài Studio.
- Trang chủ Studio có 4 khu vực: Model, Dataset, Parameters, Training/Config. Ngoài ra có Chat, Data Recipes, Export.
- Có notebook Google Colab miễn phí chạy Studio trên GPU T4; docs ghi train và chạy được hầu hết model tới 22B tham số.
- Giấy phép: phần Studio UI theo AGPL-3.0, gói Unsloth core theo Apache 2.0.

### Unsloth Core

- Gói Python `unsloth`, cài qua uv/pip (hướng dẫn chi tiết ở trang "uv, pip install & venv" của docs).
- Dùng trong code hoặc notebook: ví dụ `FastLanguageModel.from_pretrained(...)` để nạp model và `FastLanguageModel.for_inference(model)` để bật inference nhanh.
- README có bảng notebook Colab miễn phí cho từng model (Gemma 4, Qwen3.5, gpt-oss, Llama 3.1...).

Lệnh cài Core trên Linux/WSL theo GitHub README:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```

Chi tiết cài đặt từng nền tảng: [Cài đặt & phần cứng](/cai-dat).

**Nguồn:** https://unsloth.ai/docs/get-started/install, https://unsloth.ai/docs/desktop, https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/basics/inference-and-deployment/unsloth-inference, https://github.com/unslothai/unsloth

## Bảng so sánh

| Tiêu chí | Desktop | Studio | Core |
| --- | --- | --- | --- |
| **Dạng** | App native (Tauri) | Web UI trên trình duyệt | Thư viện Python |
| **Ai dùng** | Người muốn cài nhanh, không code (README: khuyến nghị) | Người muốn UI no-code nhưng tự cài, chạy trên server/Colab/Docker | Người viết code train/inference, dùng notebook |
| **Cài thế nào** | Tải `.dmg` / `.exe` / `.deb` / AppImage | Script `install.sh` / `install.ps1`, sau đó chạy lệnh khởi động Studio (các trang ghi lệnh khác nhau, xem mục CLI); hoặc Docker `unsloth/unsloth`; hoặc Colab | `uv pip install unsloth --torch-backend=auto` |
| **Chạy (inference)** | Chat GGUF, MLX, safetensors, diffusion, audio; API | Chat GGUF, safetensors; so sánh model song song; API | Inference trong code (`FastLanguageModel`) |
| **Train** | Không code: LoRA, full fine-tuning, pretraining; diffusion LoRA | QLoRA, LoRA, full fine-tuning; Text, Vision, Audio, Embeddings | LoRA, QLoRA, full fine-tuning, RL (GRPO, DPO...) qua code/notebook |
| **Export** | Có (qua giao diện) | GGUF, safetensors, LoRA | Cần kiểm tra lại (không có trong các trang nguồn đã dùng) |
| **Nền tảng** | macOS, Windows, Linux, WSL | macOS, Linux, WSL, Windows (PowerShell) | Linux, WSL, Windows theo README; macOS cần kiểm tra lại |
| **Giấy phép** | Cần kiểm tra lại | Studio UI: AGPL-3.0 | Apache 2.0 |

::: warning Docs chưa thống nhất
Các trang ghi khả năng **train** trên từng loại phần cứng khác nhau:

- Train no-code "start training instantly on **NVIDIA**" (mục No-code training): [new/studio](https://unsloth.ai/docs/new/studio)
- "**MacOS:** Training, MLX and GGUF inference all work inside of Unsloth": [new/studio](https://unsloth.ai/docs/new/studio)
- "**AMD** training: Train, run RL, chat and deploy on AMD GPUs across Windows, WSL and Linux": [GitHub README](https://github.com/unslothai/unsloth)
- Hỗ trợ "MacOS, Linux, Windows, NVIDIA, AMD, **Intel** and CPU setups", không tách riêng inference hay training: [docs](https://unsloth.ai/docs), [basics/api](https://unsloth.ai/docs/basics/api)
- Hỗ trợ "NVIDIA, Intel, AMD and Mac GPUs/CPUs", phần cứng cũ có thể không được hỗ trợ tốt: [desktop](https://unsloth.ai/docs/desktop)

Không trang nào trong các nguồn trên ghi rõ việc train trên GPU Intel hoặc chỉ dùng CPU.
:::

::: warning Model GGUF chỉ dùng để inference
Trong Studio, model định dạng GGUF bị loại khỏi danh sách train vì chỉ dùng cho inference.
:::

::: tip Kiến thức nền
GGUF, MLX, safetensors và các mức quantization: xem [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa). Ý nghĩa "22B tham số": xem [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho).
:::

**Nguồn:** https://unsloth.ai/docs/get-started/install, https://unsloth.ai/docs/desktop, https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://github.com/unslothai/unsloth

## Đối chiếu sang CLI

Sau khi cài Studio (hoặc Desktop), lệnh `unsloth` có trong terminal. Các lệnh xuất hiện trong nguồn:

| Lệnh | Tác dụng |
| --- | --- |
| `unsloth studio -H 0.0.0.0 -p 8888` | Mở Studio, bind mọi địa chỉ mạng, cổng 8888 (lệnh trong trang cài đặt, Studio) |
| `unsloth studio` | Mở Studio, không kèm flag (lệnh trong GitHub README) |
| `unsloth studio --secure` | Mở Studio qua HTTPS bằng tunnel Cloudflare miễn phí |
| `unsloth studio reset-password` | Đặt lại mật khẩu |
| `unsloth run --model ...` | Nạp một model GGUF, bật server API, in ra URL endpoint và API key |
| `unsloth start claude` | Nối agent (Claude Code, Codex, OpenCode, Hermes, OpenClaw...) với model local |

Ví dụ nạp model từ CLI (nguyên văn docs API):

```bash
unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
```

Mở Studio theo trang Get started, rồi vào `http://127.0.0.1:8888` trên trình duyệt:

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

::: warning Docs chưa thống nhất
| Thông số | Nguồn A: [get-started/install](https://unsloth.ai/docs/get-started/install), [new/studio](https://unsloth.ai/docs/new/studio), [new/studio/start](https://unsloth.ai/docs/new/studio/start) | Nguồn B: [GitHub README](https://github.com/unslothai/unsloth) |
| --- | --- | --- |
| Lệnh khởi động Studio | `unsloth studio -H 0.0.0.0 -p 8888` | `unsloth studio` (mục Launch); `unsloth studio -p 8888` (mục cài bản developer) |
| Địa chỉ bind | `-H 0.0.0.0` ngay trong lệnh khởi động cơ bản | `-H 0.0.0.0` được xếp vào mục "Remote HTTPS & LAN Access"; muốn không ra mạng thì dùng `-H 127.0.0.1` |
| Tạo mật khẩu | Mở `http://127.0.0.1:8888` trên trình duyệt, tại đó tạo mật khẩu mới ([new/studio/start](https://unsloth.ai/docs/new/studio/start)) | Khi mở ra mạng (`--secure`, `--cloudflare` hoặc `-H` không phải loopback), terminal hỏi một lần mật khẩu admin mới; Ctrl+C sẽ hủy khởi động |
| Cổng mặc định khi không có `-p` | Không ghi | Không ghi |
:::

Mục "Advanced Settings" của trang Studio còn mô tả một CLI `cli.py` với các lệnh `train`, `inference`, `export`, `list-checkpoints`, `ui`, `studio`:

```
Usage: cli.py [COMMAND]

Commands:
  train             Fine-tune a model
  inference         Run inference on a trained model
  export            Export a trained adapter
  list-checkpoints  List saved checkpoints
  ui                Launch the Unsloth Studio web UI
  studio            Launch the studio (alias)
```

::: warning Docs chưa thống nhất
| Tác vụ | Nguồn A: `cli.py` trong [new/studio/start](https://unsloth.ai/docs/new/studio/start) | Nguồn B: lệnh `unsloth` trong [new/studio](https://unsloth.ai/docs/new/studio), [basics/api](https://unsloth.ai/docs/basics/api), [GitHub README](https://github.com/unslothai/unsloth) |
| --- | --- | --- |
| Mở giao diện Studio | `cli.py ui` hoặc `cli.py studio` | `unsloth studio` |
| Fine-tune | `cli.py train` | Nguồn B không ghi lệnh CLI; train qua giao diện Studio/Desktop hoặc code Core |
| Inference | `cli.py inference` | `unsloth run --model ...` |
| Export | `cli.py export` | Nguồn B không ghi lệnh CLI; export qua mục Export của Studio |
| Liệt kê checkpoint | `cli.py list-checkpoints` | Nguồn B không ghi |

Nguồn A đặt `cli.py` trong cây thư mục tên `new-ui-prototype/`. Không nguồn nào nói `cli.py` và lệnh `unsloth` có quan hệ gì với nhau.
:::

::: danger Mở server ra mạng
Người có API key hoặc mật khẩu và truy cập được server sẽ gửi được request tới model đang nạp. Nếu server-side tools đang bật, họ còn chạy được code tùy ý trên máy host.
:::

::: warning Docs chưa thống nhất
| Thông số | Nguồn A: [basics/api](https://unsloth.ai/docs/basics/api) | Nguồn B: [GitHub README](https://github.com/unslothai/unsloth) |
| --- | --- | --- |
| Server-side tools mặc định | Với `unsloth run`: bật khi bind `127.0.0.1`, **tắt** khi bind `0.0.0.0` hoặc địa chỉ không phải loopback | Mục "Remote HTTPS & LAN Access" của Studio: "Server-side tools are **on** by default - so be careful!" |
| Cách bật/tắt | `--enable-tools` / `--disable-tools`; khi bind `0.0.0.0` thì `--enable-tools` hỏi xác nhận y/N, `--yes` / `-y` bỏ qua câu hỏi | Giữ mật khẩu an toàn hoặc dùng `--disable-tools` khi mở Unsloth ra ngoài |
:::

**Nguồn:** https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs, https://github.com/unslothai/unsloth

## Đối chiếu sang API

Model đã nạp trong Unsloth (kể cả GGUF) được phơi ra thành API có xác thực thông qua `llama-server` (server của llama.cpp). Cùng một cổng phục vụ hai "phương ngữ":

| Endpoint | Tương thích với | Dùng từ |
| --- | --- | --- |
| `POST /v1/messages` | Anthropic Messages API | Claude Code, Anthropic SDK, OpenClaw |
| `POST /v1/chat/completions` | OpenAI Chat Completions API | OpenAI SDK, opencode, Cursor, Continue, Cline, Open WebUI, curl |
| `GET /v1/models` | Danh sách model của OpenAI | Liệt kê model đang nạp |

::: warning Docs chưa thống nhất
Danh sách endpoint phía OpenAI khác nhau ngay trong cùng một trang [basics/api](https://unsloth.ai/docs/basics/api):

- Đoạn giới thiệu: "OpenAI-compatible `/v1/chat/completions` **and `/v1/responses`**".
- Bảng Endpoints: chỉ có `POST /v1/messages`, `POST /v1/chat/completions`, `GET /v1/models`, không có `/v1/responses`.
:::

Các điểm chính:

- **API key:** tạo trong Settings → API, có tiền tố `sk-unsloth-`, chỉ hiện một lần. Gửi kèm header `Authorization: Bearer sk-unsloth-…` ở mọi request; sai hoặc thiếu key trả về `401 Unauthorized`.
- **Cổng:** API chạy trên cổng mà Unsloth được khởi động (xem hộp bên dưới).
- **Tool calling:** hỗ trợ `tools` / `tool_choice` theo cả định dạng OpenAI và Anthropic. Thêm `enable_tools` và `enabled_tools` để Unsloth tự chạy Python, web search, terminal phía server.
- **API monitor:** Studio hiển thị trực tiếp từng request (token, time-to-first-token, lỗi).

::: warning Docs chưa thống nhất
Cổng (port) của API ghi khác nhau giữa các chỗ:

- "typically `http://localhost:8000` or `http://localhost:8888`": bảng Endpoints trong [basics/api](https://unsloth.ai/docs/basics/api)
- `unsloth run` "starts the server on the default port" nhưng không ghi số cổng: [basics/api](https://unsloth.ai/docs/basics/api)
- Ví dụ `curl http://localhost:8888/v1/models` và base URL `http://127.0.0.1:8888` khi dùng `--secure`: [basics/api](https://unsloth.ai/docs/basics/api)
- Docker map cả hai cổng `-p 8000:8000 -p 8888:8888`. README gọi cổng 8888 trong container là của **JupyterLab**, và bỏ `-p 8000:8000` khi thay bằng `-e UNSLOTH_STUDIO_SECURE=1`: [GitHub README](https://github.com/unslothai/unsloth)

Base URL chính xác của máy bạn hiển thị ở đầu trang API monitor, hoặc được `unsloth run` in ra console ([basics/api](https://unsloth.ai/docs/basics/api)).
:::

Ví dụ lấy ID model (nguyên văn docs):

```bash
curl http://localhost:8888/v1/models \
  -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx"
```

**[Nhận định]** Với người quen FastAPI: có thể coi Unsloth là một server tương thích OpenAI chạy local. Code đang gọi OpenAI SDK chỉ cần đổi base URL và API key là chuyển sang model local.

::: details API nội bộ của Studio (khác với API /v1)
Trang Studio liệt kê các route backend FastAPI dạng `/api/...` (ví dụ `POST /api/train/start`, `GET /api/train/metrics`, `POST /api/inference/chat`), xác thực bằng JWT. Đây là API giao diện Studio dùng nội bộ, tách biệt với endpoint `/v1/...` dành cho client bên ngoài.
:::

::: tip Kiến thức nền
Tool calling, chat template và tham số sampling (temperature, top-p...): xem [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling).
:::

Chi tiết: [Inference & API](/inference).

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/new/studio/start, https://github.com/unslothai/unsloth

## Inference và Training

| | Inference (suy luận) | Training (huấn luyện) |
| --- | --- | --- |
| **Khái niệm** | Dùng model có sẵn để sinh câu trả lời; trọng số không đổi | Cập nhật trọng số model bằng dữ liệu của bạn |
| **Đầu vào** | Prompt, ảnh, tài liệu, audio | Dataset + model gốc + hyperparameter |
| **Đầu ra** | Văn bản, ảnh, audio, tool call | Model hoặc LoRA adapter đã train, checkpoint |

::: tip Kiến thức nền
Loss, learning rate, epoch: xem [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen). LoRA và QLoRA: xem [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora). SFT và RL: xem [RL & preference](/kien-thuc-nen/rl-va-preference).
:::

### Unsloth hỗ trợ inference

- **Desktop/Studio Chat:** tải và chạy GGUF, safetensors, adapter đã fine-tune; so sánh hai model song song; tải lên tài liệu, ảnh, audio; chỉnh temperature, top-p, top-k, system prompt. Docs ghi Unsloth dựa trên llama.cpp và Hugging Face, hỗ trợ inference multi-GPU và tự động offload.
- **Tự chọn tham số:** nếu không đặt flag sampling, `unsloth run` tự chọn thiết lập khuyến nghị cho model (context length, temperature...).
- **API và agent:** endpoint `/v1/...`, lệnh `unsloth start`.
- **Core:** docs ghi mọi đường inference QLoRA, LoRA và không LoRA đều nhanh hơn 2 lần, không cần đổi code:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 64)
```

::: info Vì sao inference đôi khi chậm hơn
Theo FAQ, web search, chạy code và self-healing tool calling đều tốn thời gian. Tắt các tính năng này thì tốc độ ngang các app dùng llama.cpp khác.
:::

### Unsloth hỗ trợ training

Ba phương pháp train trong Studio:

| Phương pháp | Mô tả | VRAM |
| --- | --- | --- |
| **QLoRA** | Model gốc lượng tử hóa 4-bit + LoRA adapter | Thấp nhất |
| **LoRA** | Model gốc full precision + LoRA adapter | Trung bình |
| **Full Fine-tuning** | Train toàn bộ trọng số | Cao nhất |

- Loại model: Text, Vision, Audio, Embeddings.
- Một số giá trị mặc định trong Studio: learning rate `2e-4`, context length `2048`, LoRA rank `16`, alpha `32`, epochs `3`, batch size `4`, gradient accumulation `8`, optimizer AdamW 8-bit.
- Docs còn liệt kê pretraining, RL, GRPO, DPO, FP8 trong danh sách hỗ trợ.
- Dữ liệu: nạp từ Hugging Face Hub hoặc file local (`PDF`, `DOCX`, `JSONL`, `JSON`, `CSV`, `Parquet`); Data Recipes tạo dataset từ tài liệu.
- Theo dõi: biểu đồ loss, learning rate, gradient norm, eval loss; GPU monitor (utilization, VRAM, nhiệt độ).
- Export: GGUF, safetensors, LoRA để chạy trong Unsloth, llama.cpp, Ollama, vLLM, LM Studio.

::: warning Docs chưa thống nhất
Định dạng file dữ liệu tải lên để train được liệt kê khác nhau:

- `PDF`, `DOCX`, `JSONL`, `JSON`, `CSV`, `Parquet` (tab Local của mục Dataset): [new/studio/start](https://unsloth.ai/docs/new/studio/start)
- PDF, CSV, JSON, DOCX, **TXT** (Auto-create datasets): [new/studio](https://unsloth.ai/docs/new/studio)
- PDF, CSV, JSON hoặc YAML config (mục No-code training): [new/studio](https://unsloth.ai/docs/new/studio)
- PDF, CSV hoặc JSONL (mục Workflow): [new/studio](https://unsloth.ai/docs/new/studio)
- PDF, CSV hoặc JSON (mục Train with no code): [desktop](https://unsloth.ai/docs/desktop)
- PDF, CSV, DOCX "and more" (Data Recipes): [docs](https://unsloth.ai/docs)
:::

Chi tiết: [Fine-tuning](/fine-tuning), [Reinforcement Learning](/reinforcement-learning), [Dữ liệu](/du-lieu), [Export & deploy](/export-deploy).

**Nguồn:** https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/basics/inference-and-deployment/unsloth-inference, https://unsloth.ai/docs/desktop, https://unsloth.ai/docs

## Sơ đồ tổng thể

```mermaid
flowchart LR
  U["Người dùng"] --> D["Unsloth Desktop (app native)"]
  U --> S["Unsloth Studio (web UI)"]
  U --> C["Unsloth Core (thư viện Python)"]
  D -. "cài kèm / cùng giao diện" .-> S
  CLI["CLI: unsloth studio / run / start"] --> S

  S --> CHAT["Chat GGUF, safetensors"]
  S --> API["API /v1/chat/completions /v1/messages"]
  S --> TRAIN["Train QLoRA, LoRA, Full FT"]
  C --> TRAIN
  C --> INF["Inference trong code FastLanguageModel"]

  DATA["Data Recipes PDF, CSV, DOCX"] --> TRAIN
  TRAIN --> EXP["Export GGUF, safetensors, LoRA"]
  EXP --> CHAT
  EXP --> OUT["llama.cpp, Ollama, vLLM, LM Studio"]
  API --> AG["Agent & SDK Claude Code, Codex, OpenAI SDK"]
```

**[Nhận định]** Đường nét đứt Desktop → Studio thể hiện cách đọc của người viết: docs ghi "cách dễ nhất để cài Unsloth Studio là dùng app Desktop", nên Desktop có thể chứa sẵn Studio. Quan hệ chính xác giữa hai sản phẩm cần kiểm tra lại.

**Nguồn:** https://unsloth.ai/docs/get-started/install, https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/basics/api
