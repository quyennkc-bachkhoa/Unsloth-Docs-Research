---
title: Tổng quan kiến trúc
description: Ba dạng sản phẩm của Unsloth (Desktop, Studio, Core), CLI, API và phân biệt inference với training.
---

# Tổng quan kiến trúc

## Ba dạng sản phẩm

Unsloth có ba cách dùng tách biệt. Bạn chọn theo việc mình muốn cài app, dùng giao diện web hay viết code. Docs cài đặt của Unsloth nêu ba dạng:

1. **Unsloth Desktop**: app native (ứng dụng cài trực tiếp lên hệ điều hành), xây trên Tauri. App miễn phí, mã nguồn mở, đang ở bản Beta. GitHub README ghi đây là cách được khuyến nghị.
2. **Unsloth Studio**: web UI (giao diện chạy trên trình duyệt), no-code (không cần viết code), bản Beta. Bạn cài thủ công bằng script, rồi mở qua trình duyệt.
3. **Unsloth Core**: gói Python gốc. Bạn dùng code để training và inference.

### Unsloth Desktop

Desktop là cách nhanh nhất để bắt đầu: tải bộ cài, mở app, chọn model rồi chat.

- **Cài đặt:** tải bộ cài cho macOS (`.dmg`), Windows (`.exe`) hoặc Linux (`.deb` x64/ARM64, AppImage). Nguồn tải là https://unsloth.ai/download hoặc GitHub Releases.
- **Bắt đầu dùng:** mở app, vào dropdown "Select model" hoặc tab "Model hub". Chọn model và mức quantization vừa với máy, tải về, rồi chat.
- **Tính năng để chat và dùng công cụ:**
  - chat với tool calling "tự sửa lỗi" (self-healing), docs ghi chính xác hơn 50%;
  - chạy Bash và Python trong môi trường cô lập;
  - web search và deep research.
- **Tính năng cho ảnh và âm thanh:** sinh và train model diffusion cho ảnh hoặc video; audio (TTS, Whisper, Qwen3-ASR).
- **Tính năng để train và phục vụ model:**
  - train không cần code;
  - phục vụ model qua LAN hoặc Cloudflare HTTPS;
  - nối nhà cung cấp cloud (OpenAI, Anthropic, Ollama, vLLM...).
- **Cấp quyền:** model không được đọc file, sửa file hay truy cập internet nếu bạn chưa cho phép.

### Unsloth Studio

Studio là giao diện trên trình duyệt để fine-tune LLM mà không cần viết code.

- Docs mô tả Studio là GUI local chạy trên trình duyệt. Studio lo các phần: nạp model, định dạng dataset, cấu hình hyperparameter (siêu tham số) và theo dõi training trực tiếp.
- Trang Studio ghi cách dễ nhất để có Studio là cài app Desktop. Lệnh cài thủ công chỉ dành cho ai muốn tự cài Studio.
- Trang chủ Studio có 4 khu vực: Model, Dataset, Parameters, Training/Config. Ngoài ra có Chat, Data Recipes, Export.
- Có notebook Google Colab miễn phí chạy Studio trên GPU T4. Docs ghi notebook này train và chạy được hầu hết model tới 22B tham số.
- Giấy phép: phần Studio UI theo AGPL-3.0, gói Unsloth core theo Apache 2.0.

### Unsloth Core

Core là thư viện Python, dành cho khi bạn muốn tự viết code train và inference.

- Tên gói là `unsloth`, cài qua uv hoặc pip. Hướng dẫn chi tiết nằm ở trang "uv, pip install & venv" của docs.
- Bạn dùng Core trong code hoặc notebook. Ví dụ: `FastLanguageModel.from_pretrained(...)` để nạp model, `FastLanguageModel.for_inference(model)` để bật inference nhanh.
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

Bảng này đặt ba dạng cạnh nhau để bạn chọn nhanh. Ô ghi "Cần kiểm tra lại" là chỗ các trang nguồn không nói tới.

| Tiêu chí | Desktop | Studio | Core |
| --- | --- | --- | --- |
| **Dạng** | App native (Tauri) | Web UI trên trình duyệt | Thư viện Python |
| **Ai dùng** | Người muốn cài nhanh, không code (README: khuyến nghị) | Người muốn UI no-code nhưng tự cài, chạy trên server, Colab hoặc Docker | Người viết code train và inference, dùng notebook |
| **Cài thế nào** | Tải `.dmg` / `.exe` / `.deb` / AppImage | Chạy script `install.sh` / `install.ps1`, sau đó chạy lệnh khởi động Studio (các trang ghi lệnh khác nhau, xem mục CLI). Hoặc dùng Docker `unsloth/unsloth`, hoặc Colab | `uv pip install unsloth --torch-backend=auto` |
| **Chạy (inference)** | Chat GGUF, MLX, safetensors, diffusion, audio; API | Chat GGUF, safetensors; so sánh model song song; API | Inference trong code (`FastLanguageModel`) |
| **Train** | Không code: LoRA, full fine-tuning, pretraining; diffusion LoRA | QLoRA, LoRA, full fine-tuning; Text, Vision, Audio, Embeddings | LoRA, QLoRA, full fine-tuning, RL (GRPO, DPO...) qua code hoặc notebook |
| **Export** | Có (qua giao diện) | GGUF, safetensors, LoRA | Cần kiểm tra lại (không có trong các trang nguồn đã dùng) |
| **Nền tảng** | macOS, Windows, Linux, WSL | macOS, Linux, WSL, Windows (PowerShell) | Linux, WSL, Windows theo README; macOS cần kiểm tra lại |
| **Giấy phép** | Cần kiểm tra lại | Studio UI: AGPL-3.0 | Apache 2.0 |

::: warning Docs chưa thống nhất
Mỗi trang nói một kiểu về việc **train** được trên loại phần cứng nào. Các câu gốc:

- Train no-code "start training instantly on **NVIDIA**" (mục No-code training): [new/studio](https://unsloth.ai/docs/new/studio)
- "**MacOS:** Training, MLX and GGUF inference all work inside of Unsloth": [new/studio](https://unsloth.ai/docs/new/studio)
- "**AMD** training: Train, run RL, chat and deploy on AMD GPUs across Windows, WSL and Linux": [GitHub README](https://github.com/unslothai/unsloth)
- Hỗ trợ "MacOS, Linux, Windows, NVIDIA, AMD, **Intel** and CPU setups", không tách riêng inference hay training: [docs](https://unsloth.ai/docs), [basics/api](https://unsloth.ai/docs/basics/api)
- Hỗ trợ "NVIDIA, Intel, AMD and Mac GPUs/CPUs", phần cứng cũ có thể không được hỗ trợ tốt: [desktop](https://unsloth.ai/docs/desktop)

Không trang nào trong các nguồn trên ghi rõ việc train trên GPU Intel hoặc chỉ dùng CPU.
:::

::: warning Model GGUF chỉ dùng để inference
Trong Studio, model định dạng GGUF không xuất hiện trong danh sách train, vì GGUF chỉ dùng cho inference.
:::

::: tip Kiến thức nền
GGUF, MLX, safetensors và các mức quantization: xem [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa). Ý nghĩa "22B tham số": xem [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho).
:::

**Nguồn:** https://unsloth.ai/docs/get-started/install, https://unsloth.ai/docs/desktop, https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://github.com/unslothai/unsloth

## Đối chiếu sang CLI

Nếu bạn quen làm việc trong terminal, lệnh `unsloth` cho phép mở Studio, nạp model và nối agent mà không cần bấm giao diện. Lệnh này có sẵn sau khi bạn cài Studio hoặc Desktop. Các lệnh xuất hiện trong nguồn:

| Lệnh | Tác dụng |
| --- | --- |
| `unsloth studio -H 0.0.0.0 -p 8888` | Mở Studio, bind mọi địa chỉ mạng, cổng 8888 (lệnh trong trang cài đặt và trang Studio) |
| `unsloth studio` | Mở Studio, không kèm flag (lệnh trong GitHub README) |
| `unsloth studio --secure` | Mở Studio qua HTTPS bằng tunnel Cloudflare miễn phí |
| `unsloth studio reset-password` | Đặt lại mật khẩu |
| `unsloth run --model ...` | Nạp một model GGUF, bật server API, in ra URL endpoint và API key |
| `unsloth start claude` | Nối agent (Claude Code, Codex, OpenCode, Hermes, OpenClaw...) với model local |

Ví dụ nạp model từ CLI (nguyên văn docs API):

```bash
unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
```

Để mở Studio theo trang Get started, chạy lệnh dưới đây rồi vào `http://127.0.0.1:8888` trên trình duyệt:

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

::: warning Docs chưa thống nhất
Lệnh khởi động Studio và cách tạo mật khẩu được ghi khác nhau giữa trang docs và GitHub README:

| Thông số | Nguồn A: [get-started/install](https://unsloth.ai/docs/get-started/install), [new/studio](https://unsloth.ai/docs/new/studio), [new/studio/start](https://unsloth.ai/docs/new/studio/start) | Nguồn B: [GitHub README](https://github.com/unslothai/unsloth) |
| --- | --- | --- |
| Lệnh khởi động Studio | `unsloth studio -H 0.0.0.0 -p 8888` | `unsloth studio` (mục Launch); `unsloth studio -p 8888` (mục cài bản developer) |
| Địa chỉ bind | `-H 0.0.0.0` ngay trong lệnh khởi động cơ bản | `-H 0.0.0.0` được xếp vào mục "Remote HTTPS & LAN Access"; muốn không ra mạng thì dùng `-H 127.0.0.1` |
| Tạo mật khẩu | Mở `http://127.0.0.1:8888` trên trình duyệt, tại đó tạo mật khẩu mới ([new/studio/start](https://unsloth.ai/docs/new/studio/start)) | Khi mở ra mạng (`--secure`, `--cloudflare` hoặc `-H` không phải loopback), terminal hỏi một lần mật khẩu admin mới; Ctrl+C sẽ hủy khởi động |
| Cổng mặc định khi không có `-p` | Không ghi | Không ghi |
:::

Mục "Advanced Settings" của trang Studio còn mô tả một CLI khác tên `cli.py`, với các lệnh `train`, `inference`, `export`, `list-checkpoints`, `ui`, `studio`:

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
Hai bộ lệnh `cli.py` và `unsloth` cùng làm những việc giống nhau, nhưng docs không nói chúng liên quan thế nào:

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
Ai có API key hoặc mật khẩu và truy cập được server thì gửi được request tới model đang nạp. Nếu server-side tools đang bật, người đó còn chạy được code tùy ý trên máy host.
:::

::: warning Docs chưa thống nhất
Hai nguồn nói ngược nhau về việc server-side tools có bật sẵn hay không:

| Thông số | Nguồn A: [basics/api](https://unsloth.ai/docs/basics/api) | Nguồn B: [GitHub README](https://github.com/unslothai/unsloth) |
| --- | --- | --- |
| Server-side tools mặc định | Với `unsloth run`: bật khi bind `127.0.0.1`, **tắt** khi bind `0.0.0.0` hoặc địa chỉ không phải loopback | Mục "Remote HTTPS & LAN Access" của Studio: "Server-side tools are **on** by default - so be careful!" |
| Cách bật/tắt | `--enable-tools` / `--disable-tools`; khi bind `0.0.0.0` thì `--enable-tools` hỏi xác nhận y/N, `--yes` / `-y` bỏ qua câu hỏi | Giữ mật khẩu an toàn hoặc dùng `--disable-tools` khi mở Unsloth ra ngoài |
:::

**Nguồn:** https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs, https://github.com/unslothai/unsloth

## Đối chiếu sang API

Unsloth biến model đang nạp thành một API để code hoặc agent của bạn gọi tới. Mọi model đã nạp, kể cả GGUF, đều được phơi ra thành API có xác thực thông qua `llama-server` (server của llama.cpp). Cùng một cổng phục vụ hai "phương ngữ" API:

| Endpoint | Tương thích với | Dùng từ |
| --- | --- | --- |
| `POST /v1/messages` | Anthropic Messages API | Claude Code, Anthropic SDK, OpenClaw |
| `POST /v1/chat/completions` | OpenAI Chat Completions API | OpenAI SDK, opencode, Cursor, Continue, Cline, Open WebUI, curl |
| `GET /v1/models` | Danh sách model của OpenAI | Liệt kê model đang nạp |

::: warning Docs chưa thống nhất
Ngay trong cùng trang [basics/api](https://unsloth.ai/docs/basics/api), danh sách endpoint phía OpenAI đã khác nhau:

- Đoạn giới thiệu: "OpenAI-compatible `/v1/chat/completions` **and `/v1/responses`**".
- Bảng Endpoints: chỉ có `POST /v1/messages`, `POST /v1/chat/completions`, `GET /v1/models`, không có `/v1/responses`.
:::

Những điều cần biết khi gọi API:

- **API key:** tạo trong Settings → API. Key có tiền tố `sk-unsloth-` và chỉ hiện một lần. Mọi request phải gửi kèm header `Authorization: Bearer sk-unsloth-…`. Sai hoặc thiếu key thì server trả về `401 Unauthorized`.
- **Cổng:** API chạy trên cổng mà Unsloth được khởi động (xem hộp bên dưới).
- **Tool calling:** hỗ trợ `tools` / `tool_choice` theo cả định dạng OpenAI và Anthropic. Nếu thêm `enable_tools` và `enabled_tools`, Unsloth tự chạy Python, web search, terminal phía server.
- **API monitor:** Studio hiển thị trực tiếp từng request, gồm token, time-to-first-token và lỗi.

::: warning Docs chưa thống nhất
Số cổng (port) của API được ghi khác nhau giữa các chỗ:

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

**[Nhận định]** Nếu bạn quen FastAPI, có thể coi Unsloth là một server tương thích OpenAI chạy local. Code đang gọi OpenAI SDK chỉ cần đổi base URL và API key là chuyển sang model local.

::: details API nội bộ của Studio (khác với API /v1)
Trang Studio liệt kê các route backend FastAPI dạng `/api/...`, ví dụ `POST /api/train/start`, `GET /api/train/metrics`, `POST /api/inference/chat`. Các route này xác thực bằng JWT. Đây là API mà giao diện Studio dùng nội bộ, tách biệt với endpoint `/v1/...` dành cho client bên ngoài.
:::

::: tip Kiến thức nền
Tool calling, chat template và tham số sampling (temperature, top-p...): xem [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling).
:::

Chi tiết: [Inference & API](/inference/).

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/new/studio/start, https://github.com/unslothai/unsloth

## Inference và Training

Unsloth làm hai việc chính: inference (dùng model để trả lời) và training (dạy thêm cho model bằng dữ liệu của bạn). Bảng sau cho thấy hai việc này khác nhau ở đâu:

| | Inference (suy luận) | Training (huấn luyện) |
| --- | --- | --- |
| **Khái niệm** | Dùng model có sẵn để sinh câu trả lời; trọng số không đổi | Cập nhật trọng số model bằng dữ liệu của bạn |
| **Đầu vào** | Prompt, ảnh, tài liệu, audio | Dataset + model gốc + hyperparameter |
| **Đầu ra** | Văn bản, ảnh, audio, tool call | Model hoặc LoRA adapter đã train, checkpoint |

::: tip Kiến thức nền
Loss, learning rate, epoch: xem [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen). LoRA và QLoRA: xem [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora). SFT và RL: xem [RL & preference](/kien-thuc-nen/rl-va-preference).
:::

### Unsloth hỗ trợ inference

Bạn có thể chạy model qua giao diện chat, qua API hoặc trong code:

- **Desktop/Studio Chat:** tải và chạy GGUF, safetensors hoặc adapter đã fine-tune. Bạn có thể so sánh hai model song song, tải lên tài liệu, ảnh, audio, và chỉnh temperature, top-p, top-k, system prompt. Docs ghi Unsloth dựa trên llama.cpp và Hugging Face, hỗ trợ inference multi-GPU và tự động offload.
- **Tự chọn tham số:** nếu bạn không đặt flag sampling, `unsloth run` tự chọn thiết lập khuyến nghị cho model (context length, temperature...).
- **API và agent:** endpoint `/v1/...`, lệnh `unsloth start`.
- **Core:** docs Unsloth ghi mọi đường inference QLoRA, LoRA và không LoRA đều nhanh hơn 2 lần, không cần đổi code:

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
Theo FAQ, web search, chạy code và self-healing tool calling đều tốn thời gian. Khi tắt các tính năng này, tốc độ ngang các app dùng llama.cpp khác.
:::

### Unsloth hỗ trợ training

Studio có ba phương pháp train. Chúng khác nhau chủ yếu ở lượng VRAM cần dùng:

| Phương pháp | Mô tả | VRAM |
| --- | --- | --- |
| **QLoRA** | Model gốc lượng tử hóa 4-bit + LoRA adapter | Thấp nhất |
| **LoRA** | Model gốc full precision + LoRA adapter | Trung bình |
| **Full Fine-tuning** | Train toàn bộ trọng số | Cao nhất |

Những gì Studio hỗ trợ khi train:

- **Loại model:** Text, Vision, Audio, Embeddings.
- **Giá trị mặc định trong Studio:** learning rate `2e-4`, context length `2048`, LoRA rank `16`, alpha `32`, epochs `3`, batch size `4`, gradient accumulation `8`, optimizer AdamW 8-bit.
- **Phương pháp khác:** docs còn liệt kê pretraining, RL, GRPO, DPO, FP8 trong danh sách hỗ trợ.
- **Dữ liệu:** nạp từ Hugging Face Hub hoặc file local (`PDF`, `DOCX`, `JSONL`, `JSON`, `CSV`, `Parquet`). Data Recipes tạo dataset từ tài liệu.
- **Theo dõi:** biểu đồ loss, learning rate, gradient norm, eval loss. GPU monitor hiển thị utilization, VRAM, nhiệt độ.
- **Export:** GGUF, safetensors, LoRA, để chạy trong Unsloth, llama.cpp, Ollama, vLLM, LM Studio.

::: warning Docs chưa thống nhất
Mỗi chỗ trong docs liệt kê một bộ định dạng file dữ liệu khác nhau khi tải lên để train:

- `PDF`, `DOCX`, `JSONL`, `JSON`, `CSV`, `Parquet` (tab Local của mục Dataset): [new/studio/start](https://unsloth.ai/docs/new/studio/start)
- PDF, CSV, JSON, DOCX, **TXT** (Auto-create datasets): [new/studio](https://unsloth.ai/docs/new/studio)
- PDF, CSV, JSON hoặc YAML config (mục No-code training): [new/studio](https://unsloth.ai/docs/new/studio)
- PDF, CSV hoặc JSONL (mục Workflow): [new/studio](https://unsloth.ai/docs/new/studio)
- PDF, CSV hoặc JSON (mục Train with no code): [desktop](https://unsloth.ai/docs/desktop)
- PDF, CSV, DOCX "and more" (Data Recipes): [docs](https://unsloth.ai/docs)
:::

Chi tiết: [Fine-tuning](/fine-tuning/), [Reinforcement Learning](/reinforcement-learning/), [Dữ liệu](/du-lieu/), [Export & deploy](/export-deploy).

**Nguồn:** https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/basics/inference-and-deployment/unsloth-inference, https://unsloth.ai/docs/desktop, https://unsloth.ai/docs

## Sơ đồ tổng thể

Sơ đồ dưới gom mọi phần ở trên vào một hình: bạn vào từ Desktop, Studio hoặc Core; dữ liệu đi vào train; model đã train được export để chat hoặc chạy ở công cụ khác.

<div class="dg">
<div class="dg-layer">
<div class="dg-title">Lối vào</div>
<div class="dg-grid" style="--cols: 3; --dg-grid-gap: 48px">
<div class="dg-node">Unsloth Desktop<small>app native</small></div>
<div class="dg-node is-main dg-dash" data-e="cài kèm / cùng giao diện">Unsloth Studio<small>web UI; mở bằng CLI <code>unsloth studio / run / start</code></small></div>
<div class="dg-node is-main">Unsloth Core<small>thư viện Python</small></div>
</div>
</div>
<div class="dg-layer">
<div class="dg-title">Làm được gì</div>
<div class="dg-grid" style="--cols: 4">
<div>
<div class="dg-tags"><span class="dg-tag">Studio</span></div>
<div class="dg-col">
<div class="dg-node">Chat<small>GGUF, safetensors</small></div>
</div>
</div>
<div>
<div class="dg-tags"><span class="dg-tag">Studio</span></div>
<div class="dg-col">
<div class="dg-node">API<small><code>/v1/chat/completions</code> <code>/v1/messages</code></small></div>
<div class="dg-node is-end">Agent &amp; SDK<small>Claude Code, Codex, OpenAI SDK</small></div>
</div>
</div>
<div>
<div class="dg-tags"><span class="dg-tag">Studio</span><span class="dg-tag">Core</span></div>
<div class="dg-col">
<div class="dg-node">Data Recipes<small>PDF, CSV, DOCX</small></div>
<div class="dg-node is-main">Train<small>QLoRA, LoRA, Full FT</small></div>
<div class="dg-node">Export<small>GGUF, safetensors, LoRA</small></div>
<div class="dg-node is-end">llama.cpp, Ollama, vLLM, LM Studio</div>
<div class="dg-note">Model export cũng nạp lại vào Chat</div>
</div>
</div>
<div>
<div class="dg-tags"><span class="dg-tag">Core</span></div>
<div class="dg-col">
<div class="dg-node">Inference trong code<small><code>FastLanguageModel</code></small></div>
</div>
</div>
</div>
</div>
<div class="dg-cap">Nhãn trên mỗi cột cho biết sản phẩm nào có tính năng đó. Nét đứt Desktop → Studio là cách hiểu của người viết (xem Nhận định bên dưới).</div>
</div>

**[Nhận định]** Đường nét đứt Desktop → Studio là cách hiểu của người viết. Docs ghi "cách dễ nhất để cài Unsloth Studio là dùng app Desktop", nên Desktop có thể chứa sẵn Studio. Quan hệ chính xác giữa hai sản phẩm cần kiểm tra lại.

**Nguồn:** https://unsloth.ai/docs/get-started/install, https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/basics/api
