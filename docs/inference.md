---
title: Inference & API
description: Chạy model local trong Unsloth Studio, gọi qua API tương thích OpenAI/Anthropic, nối coding agent bằng unsloth start, Connections, MCP và tool calling.
---

# Inference & API

## Inference trong Unsloth là gì

Inference (suy luận, tức là dùng model đã có để sinh câu trả lời, không huấn luyện thêm) trong Unsloth có ba cách dùng:

1. **Chat trực tiếp** trong giao diện Unsloth Studio / Unsloth Desktop, chạy offline 100% trên máy. Studio chạy được cả trên macOS, CPU, Windows, Linux, WSL và **không bắt buộc có GPU**.
2. **Gọi qua API**: model đang nạp trong Unsloth được mở ra dưới dạng endpoint (địa chỉ HTTP nhận request) có xác thực, dùng chung định dạng với OpenAI và Anthropic. App Python của bạn hoặc coding agent (tác tử lập trình như Claude Code, Codex) gọi vào đó thay vì gọi cloud.
3. **Connections**: Unsloth đóng vai giao diện chung, nối tới nhà cung cấp cloud (OpenAI, Anthropic, OpenRouter) hoặc model server bạn tự chạy (llama.cpp, vLLM, Ollama).

Bên dưới, Studio được xây trên llama.cpp và Hugging Face; model nạp trong Unsloth (kể cả GGUF) được phục vụ qua `llama-server`.

::: tip Kiến thức nền
Chưa rõ sampling (temperature/top-p/top-k/min-p), chat template, KV cache hay tool calling là gì? Xem [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling).
:::

```mermaid
flowchart LR
  A["App Python / curl"] --> C["API Unsloth trên localhost (vd :8888)"]
  B["Coding agent: Claude Code, Codex, OpenCode"] --> C
  C -->|"/v1/chat/completions, /v1/responses"| D["llama-server trong Unsloth"]
  C -->|"/v1/messages"| D
  D --> E["Model local (GGUF) trên CPU/GPU"]
  C -.->|"server-side tools"| F["Python, bash, web search"]
```

**Nguồn:** https://unsloth.ai/docs/new/studio/chat, https://unsloth.ai/docs/basics/api

## Chạy model local trong Studio Chat

### Tải và chọn model

- Mở dropdown **Select model** (góc trên bên trái trang Chat) hoặc tab **Model hub**, chọn model và mức quantization (lượng tử hóa, nén trọng số để model nhẹ hơn) vừa với máy, tải về rồi chat ngay.
- Studio tìm và tải model từ Hugging Face hoặc dùng file local. Định dạng hỗ trợ: GGUF (định dạng file model của llama.cpp, thường đã lượng tử hóa), safetensors (định dạng lưu trọng số chuẩn của Hugging Face), LoRA adapter, model vision-language và text-to-speech.
- Ví dụ trong tài liệu: `unsloth/gemma-4-26B-A4B-it-GGUF` với quant khuyến nghị `UD-Q4_K_XL`.

::: tip Kiến thức nền
Đọc tên model/GGUF như `26B-A4B`, `UD-Q4_K_XL`: xem [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) và [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).
:::

**Model GGUF đã tải sẵn từ trước:** Studio tự phát hiện model cũ tải qua Hugging Face, LM Studio… (cập nhật 27/3) và cho chọn một thư mục có sẵn để quét (cập nhật 1/4). Studio đọc từ Hugging Face Hub cache; file GGUF của LM Studio nằm ở thư mục riêng, không hiển thị với llama.cpp mặc định, nên cần chép/chuyển các file `.gguf` sang HF cache (hoặc đường dẫn llama.cpp đọc được).

::: warning Docs chưa thống nhất
Đường dẫn cache trên Windows được viết khác nhau, ngay trong cùng một trang [new/studio/chat](https://unsloth.ai/docs/new/studio/chat):

| Thư mục | Cách viết 1 | Cách viết 2 |
| --- | --- | --- |
| Hugging Face Hub cache | `C:\Users{your_username}.cache\huggingface\hub` (mục "Using old / existing GGUF models") | `C:\Users\<username>\.cache\huggingface\hub\` và `%USERPROFILE%\.cache\huggingface\hub\` (mục "Deleting model files") |
| Model của LM Studio | `C:\Users\{your_username}.cache\lm-studio\models` | `C:\Users{your_username}\lm-studio\models` |
:::

**Xóa model:** bấm biểu tượng thùng rác trong model search, hoặc xóa thư mục trong HF cache:

- macOS, Linux, WSL: `~/.cache/huggingface/hub/`
- Windows: `%USERPROFILE%\.cache\huggingface\hub\`

Nếu đặt biến `HF_HUB_CACHE` hoặc `HF_HOME` thì dùng thư mục đó; trên Linux/WSL, `XDG_CACHE_HOME` cũng đổi được gốc cache.

### Tính năng của Chat

| Tính năng | Mô tả ngắn |
| --- | --- |
| Code execution | Model chạy được Bash và Python (không chỉ JavaScript), trong sandbox, để thử code, tạo file, kiểm chứng đáp án bằng tính toán thật |
| Auto-healing tool calling | Tự sửa tool call (lời gọi công cụ) sai định dạng; nguồn ghi giảm lỗi 50% |
| Advanced web search | Truy cập trực tiếp trang web để đọc nội dung thay vì chỉ đọc tóm tắt; dùng API của DuckDuckGo |
| Chat Workspace | Nhập prompt, đính kèm tài liệu, ảnh (webp, png), file code, txt, audio; bật/tắt **Thinking** và **Web search** |
| Thêm file làm context | Đính kèm PDF, ảnh chụp màn hình, DOCX…; file được xử lý local |
| Model Arena | So sánh 2 model cạnh nhau với cùng prompt (vd base model và LoRA adapter sau fine-tune); hiện nạp tuần tự từng model, chạy song song "đang được phát triển" |
| Multi-GPU | Chat tự chạy trên hệ nhiều GPU khi inference |
| Connect Providers | Dùng model cloud/model server trong cùng giao diện (xem mục Connections) |

Kết quả thử nghiệm của Unsloth với `unsloth/Qwen3.5-4B-GGUF (UD-Q4_K_XL)` (bật web search, code execution, thinking):

| Chỉ số | Tool calling thường | Tool calling của Unsloth |
| --- | --- | --- |
| XML lọt vào câu trả lời | 10/10 | 0/10 |
| Số lần fetch URL | 0 | 4/10 lần chạy |
| Lần chạy ra đúng tên bài hát | 0/10 | 2/10 |
| Số tool call trung bình | 5.5 | 3.8 |
| Thời gian phản hồi trung bình | 12.3s | 9.8s |

### Tham số sampling

- Với model mới như Qwen3.5, Studio **tự đặt sẵn** temperature, top-p, top-k, MTP để ra kết quả tốt; vẫn chỉnh tay được, và sửa được system prompt lẫn chat template.
- Không cần chỉnh context length (độ dài ngữ cảnh) nhờ "smart auto context" của llama.cpp: chỉ dùng phần context cần thiết. Context length là gì và vì sao nó quyết định KV cache: xem [Token & context](/kien-thuc-nen/token-va-context), [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho).
- Giá trị mặc định cụ thể của từng tham số trong giao diện Studio: **cần kiểm tra lại** (trang nguồn không ghi số).

::: warning Docs chưa thống nhất
Về việc có cần chỉnh context length hay không, các trang ghi khác nhau:

- [new/studio/chat](https://unsloth.ai/docs/new/studio/chat): "Context length adjustment is no longer necessary" nhờ smart auto context của llama.cpp.
- [integrations/unsloth-start](https://unsloth.ai/docs/integrations/unsloth-start): có cờ `--context-length` / `--max-seq-length`, ví dụ `--context-length 32768`; không truyền cờ thì Unsloth tự chọn context length.
- [basics/api](https://unsloth.ai/docs/basics/api): ví dụ tăng context bằng `-c 131072`.
- [basics/codex](https://unsloth.ai/docs/basics/codex): khi OOM giữa chừng, "Reduce context in Unsloth **Settings → Inference**".
:::

Khi chạy bằng dòng lệnh `unsloth run`, nếu không truyền cờ sampling nào thì Unsloth tự chọn cấu hình khuyến nghị cho model (context length, temperature…). Muốn ghi đè thì truyền cờ, các cờ này được chuyển thẳng xuống `llama-server`:

```bash
# Lower randomness and improve reproducibility
unsloth run \
  --model unsloth/Qwen3-1.7B-GGUF \
  --temp 0.6 \
  --seed 42
```

```bash
# Tune token selection and repetition behavior
unsloth run \
  --model unsloth/Qwen3-1.7B-GGUF \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.05 \
  --repeat-penalty 1.1
```

```bash
# Use a larger context window and more CPU threads
unsloth run \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  -c 131072 \
  --threads 32
```

::: details Unsloth không dùng GPU (Docker)
Kéo image mới nhất bằng `docker pull unsloth/unsloth:latest`; chạy container với `--gpus all` (`docker run`) hoặc `capabilities: [gpu]` (Docker Compose). Trên Linux cần NVIDIA Container Toolkit. Trên Windows, kiểm tra `nvcc --version` khớp với CUDA version trong `nvidia-smi`.
:::

**Nguồn:** https://unsloth.ai/docs/new/studio/chat, https://unsloth.ai/docs/basics/api

## API tương thích OpenAI và Anthropic

### Endpoint, port và API key

Unsloth nói "hai phương ngữ" trên **cùng một port**, đều hỗ trợ streaming (trả kết quả từng phần), tool calling và ảnh đầu vào:

| Endpoint | Tương thích với | Dùng từ |
| --- | --- | --- |
| `POST /v1/messages` | Anthropic Messages API | Claude Code, Anthropic SDK, OpenClaw |
| `POST /v1/chat/completions` | OpenAI Chat Completions API | OpenAI SDK, opencode, Cursor, Continue, Cline, Open WebUI, curl |
| `POST /v1/responses` | OpenAI Responses API | Codex và các client OpenAI đời mới |
| `GET /v1/models` | OpenAI models list | Liệt kê model đang nạp (lấy `id` để điền vào trường model) |

::: warning Docs chưa thống nhất
Danh sách endpoint khác nhau giữa các trang:

- Bảng "Endpoints" của [basics/api](https://unsloth.ai/docs/basics/api) chỉ liệt kê `POST /v1/messages`, `POST /v1/chat/completions`, `GET /v1/models`, không có `/v1/responses`. Phần mở đầu của cùng trang lại ghi "OpenAI-compatible `/v1/chat/completions` and **`/v1/responses`**".
- [integrations/connect-curl-and-http-to-unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth) có ví dụ riêng cho `/v1/responses`; [basics/codex](https://unsloth.ai/docs/basics/codex) dùng endpoint `/v1/responses`.

Dòng `/v1/responses` trong bảng trên được tổng hợp từ hai trang sau.
:::

- **Port:** xem hộp bên dưới; các ví dụ trên trang này dùng `8888` như trong docs.
- **API key:** mọi request cần header `Authorization: Bearer sk-unsloth-…`.

::: warning Docs chưa thống nhất
Port và base URL mặc định của API Unsloth được ghi khác nhau:

| Trang | Port / base URL ghi trong docs |
| --- | --- |
| [basics/api](https://unsloth.ai/docs/basics/api) | "typically `http://localhost:8000` or `http://localhost:8888`"; mục `--secure` nhắc `http://127.0.0.1:8888`; ví dụ curl dùng `8888` |
| [integrations/connect-python-sdk-to-unsloth](https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth) | "typically `8000` or `8888`"; code dùng `http://localhost:8888/v1` (OpenAI SDK) và `http://localhost:8888` (Anthropic SDK) |
| [integrations/connect-curl-and-http-to-unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth) | "on the port Unsloth started on"; ví dụ dùng `8888` |
| [basics/claude-code](https://unsloth.ai/docs/basics/claude-code) | `ANTHROPIC_BASE_URL="http://localhost:8888"` |
| [basics/codex](https://unsloth.ai/docs/basics/codex) | `base_url = "http://localhost:8888/v1"`, và ghi `8888` là "Unsloth Studio's default" |
| [integrations/opencode](https://unsloth.ai/docs/integrations/opencode) | `http://localhost:8888/v1/` (có dấu `/` cuối, "keep the trailing `/v1/`") |

Muốn biết chắc port: `unsloth run` in endpoint URL ra console ([basics/api](https://unsloth.ai/docs/basics/api)).
:::

Tạo API key trong giao diện:

1. Bấm avatar **Unsloth** ở góc dưới bên trái sidebar.
2. Vào **Settings** → **API**.
3. Nhập tên dễ nhớ (vd `claude-code-macbook`), đặt hạn dùng nếu muốn.
4. Bấm **Create** và **chép key ngay**: Unsloth chỉ lưu hash, không hiển thị lại.

Key bị thu hồi sẽ nhận `401 Unauthorized`.

Cách khác: `unsloth run` nạp model, tự tạo API key và in endpoint + key ra console:

```bash
unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
```

Ba cách viết tên model tương đương:

```bash
# Combined: repo and quantization variant in one string (recommended — shortest)
unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL

# Separate: repo and variant as two flags (the older style, still works)
unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF --gguf-variant UD-Q4_K_XL

# Using -hf / --hf-repo (matches llama.cpp's spelling, handy if you're coming from there)
unsloth run -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
```

**[Nhận định]** Nếu quen FastAPI: hình dung Unsloth như một app có sẵn các route `POST /v1/...` với dependency kiểm tra Bearer token; client chỉ cần đổi `base_url` và `api_key`, còn schema request/response giữ nguyên như khi gọi OpenAI/Anthropic thật. Bạn không phải tự viết server bọc model.

### Ví dụ curl

Liệt kê model đang nạp:

```bash
curl http://localhost:8888/v1/models \
  -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx"
```

```json
{
  "object": "list",
  "data": [
    {"id": "unsloth/gemma-3-27b-it-GGUF", "object": "model", "owned_by": "local"}
  ]
}
```

::: warning Docs chưa thống nhất
Model ID mà Unsloth trả về (và cần điền vào trường `model`) có hai dạng, có và không có tiền tố `unsloth/`:

| Trang | Model ID ghi trong docs |
| --- | --- |
| [integrations/connect-curl-and-http-to-unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth) | response mẫu: `unsloth/gemma-3-27b-it-GGUF` |
| [basics/api](https://unsloth.ai/docs/basics/api) (Troubleshooting) | response mẫu: `"id": "gemma-4-26B-A4B-it-GGUF"` |
| [basics/codex](https://unsloth.ai/docs/basics/codex) | "Unsloth Studio exposes the full repo id (for example `unsloth/gemma-4-26B-A4B-it-GGUF`)" |
| [basics/claude-code](https://unsloth.ai/docs/basics/claude-code) | bash: `ANTHROPIC_MODEL="unsloth/gemma-4-26B-A4B-it-GGUF"`; PowerShell: `$env:ANTHROPIC_MODEL = "gemma-4-26B-A4B-it-GGUF"` |
| [integrations/connect-python-sdk-to-unsloth](https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth) | `model="default"` ("the name you gave the model in unsloth or default"), cũng dùng `qwen-local`, `unsloth/Qwen3.6-27B-GGUF` |

Các trang đều khuyên gọi `GET /v1/models` và chép nguyên trường `id`.
:::

Chat Completions (OpenAI):

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Streaming: thêm `"stream": true`, response chuyển sang SSE (Server-Sent Events, luồng sự kiện `text/event-stream`); dùng `-N` để curl không gom bộ đệm:

```bash
curl -N http://localhost:8888/v1/chat/completions \
  -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-local",
    "messages": [{"role": "user", "content": "Write a haiku about locally-run LLMs."}],
    "stream": true
  }'
```

Anthropic Messages (`max_tokens` là **bắt buộc** ở endpoint này, tùy chọn ở `/v1/chat/completions`):

```bash
curl http://localhost:8888/v1/messages \
  -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Responses (OpenAI Responses API):

```bash
curl http://localhost:8888/v1/responses \
  -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-local",
    "input": "Write a one-sentence greeting."
  }'
```

Ghi đè tham số sinh cho từng request (giá trị trong request thắng giá trị mặc định của server):

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Authorization: Bearer $UNSLOTH_STUDIO_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {
        "role": "user",
        "content": "Write a short poem about local AI."
      }
    ],
    "temperature": 0.8,
    "top_p": 0.9,
    "max_tokens": 512
  }'
```

Thinking (model "suy nghĩ" trước khi trả lời) bật mặc định; gửi `enable_thinking: false` trong request để tắt.

### Python: OpenAI SDK

Đặt key vào biến môi trường để không dán thẳng vào code:

```bash
export UNSLOTH_STUDIO_AUTH_TOKEN=sk-unsloth-xxxxxxxxxxxx
```

```bash
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8888/v1",              # your unsloth port + /v1
    api_key=os.environ["UNSLOTH_STUDIO_AUTH_TOKEN"],     # your sk-unsloth-… key
)
```

```python
response = client.chat.completions.create(
    model="default",                               # the name you gave the model in unsloth or default
    messages=[
        {"role": "user", "content": "Give me two facts about Paris"}
    ],
)
print(response.choices[0].message.content)
```

Streaming:

```python
stream = client.chat.completions.create(
    model="qwen-local",
    messages=[{"role": "user", "content": "Write a haiku about locally-run LLMs."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
```

### Python: Anthropic SDK

```bash
pip install anthropic
```

```python
import os
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8888", # your unsloth port (no /v1 here - the SDK adds it)
    api_key="dummy", # a random none empty value
    default_headers={"Authorization": f"Bearer {os.environ['UNSLOTH_STUDIO_AUTH_TOKEN']}"} # your sk-unsloth-… key
)
```

```python
message = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Say hello in three languages."}
    ],
)
print(message.content[0].text)
```

Chọn SDK nào: dùng OpenAI SDK nếu code đã phụ thuộc gói `openai`, muốn `tools`/`tool_choice` kiểu OpenAI hoặc Responses API; dùng Anthropic SDK nếu đã dùng gói `anthropic`, thích định dạng tool `input_schema` hoặc kiểu sự kiện streaming của Anthropic. Có thể dùng cả hai trong một project với cùng một key.

Ngoài ra API còn hỗ trợ ảnh (vision, model phải là multimodal) và structured output qua `response_format` với JSON Schema (`strict: True` để ép schema lúc decode). Xem chi tiết tại trang nguồn Python SDK.

### Cấu hình server và truy cập từ máy khác

- Mặc định Unsloth chỉ nghe trên máy local. Bind `0.0.0.0` để máy khác trong LAN kết nối:

```bash
# Allow LAN devices to connect
unsloth run \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  -H 0.0.0.0 \
  -p 8888
```

- **Chính sách server-side tools** (công cụ do Unsloth tự chạy: web search, code execution…): trên `127.0.0.1` mặc định **bật**; trên `0.0.0.0` hoặc địa chỉ không phải loopback mặc định **tắt**, vì lộ API key trên server mở mạng đồng nghĩa với cho người khác chạy code tùy ý trên máy. Ép bật/tắt bằng `--enable-tools` / `--disable-tools` (trên `0.0.0.0`, `--enable-tools` hỏi xác nhận y/N; `--yes`/`-y` bỏ qua câu hỏi). Request không thể vượt chính sách này bằng `enable_tools=true`.
- Truy cập từ xa qua Internet: chạy `unsloth studio --secure`, Unsloth vẫn bind localhost và mở một URL HTTPS Cloudflare miễn phí. SSE không đi qua được Cloudflare quick tunnel, nên đặt `stream: false`. Docs không nói chính sách server-side tools áp dụng thế nào trong trường hợp này (server bind localhost nhưng truy cập được từ Internet): **cần kiểm tra lại**.

::: warning Docs chưa thống nhất
Server-side tools "bật mặc định" hay "phải tự bật" được mô tả khác nhau:

| Trang | Mô tả trong docs |
| --- | --- |
| [basics/api](https://unsloth.ai/docs/basics/api) (Server-side tool policy) | `127.0.0.1`: tools **on** by default; `0.0.0.0` hoặc địa chỉ non-loopback: tools **off** by default |
| [basics/api](https://unsloth.ai/docs/basics/api) (Tool calling, Troubleshooting) | "Opt in by adding these extra fields"; "remember to set `enable_tools: true` **and** list the ones you want in `enabled_tools`" |
| [integrations/connect-curl-and-http-to-unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth), [integrations/connect-python-sdk-to-unsloth](https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth) | Opt in bằng `enable_tools: true` trong request |
| [basics/claude-code](https://unsloth.ai/docs/basics/claude-code), [integrations/opencode](https://unsloth.ai/docs/integrations/opencode) | "By default Unsloth Studio runs its own server-side tools, which swallows the agent's tool calls"; khuyên thêm `--disable-tools` |
| [basics/codex](https://unsloth.ai/docs/basics/codex) | "When driving an external coding agent, add `--disable-tools`" |

Ví dụ LAN trong [basics/claude-code](https://unsloth.ai/docs/basics/claude-code) dùng `-H 0.0.0.0` mà không có `--disable-tools`; ví dụ LAN trong [integrations/opencode](https://unsloth.ai/docs/integrations/opencode) có `--disable-tools`.
:::
- **API monitor**: mọi request dùng API key hiện trong Studio (panel góc màn hình và trang **Settings → API Monitor**), gồm prompt, response, số token, time-to-first-token, throughput, lỗi, và **Context used**.

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth

## Kết nối coding agent với `unsloth start`

`unsloth start <agent>` tự cấu hình endpoint, API key, provider, model và context length cho từng lần chạy agent. Cấu hình là tạm thời/theo phiên, **không** ghi provider Unsloth vào file cấu hình thường ngày của agent. Toàn bộ có thể chạy offline.

Cách dùng nhanh: mở Unsloth, nạp model, `cd` vào thư mục project rồi chạy:

```bash
unsloth start claude
```

| Agent | Lệnh | Ghi chú |
| --- | --- | --- |
| Claude Code | `unsloth start claude` | Dùng session store sẵn có của Claude, không cần `--persist`; tiếp tục phiên bằng `unsloth start claude --continue` |
| OpenAI Codex | `unsloth start codex` | Cần model **GGUF** chạy qua backend `llama-server`; home của Codex là tạm, cần `--persist` để giữ phiên |
| DeepSeek Harness | `unsloth start dsh` | Nguồn không ghi thêm |
| OpenCode | `unsloth start opencode` | Dùng session store sẵn có, không cần `--persist` |
| Hermes Agent | `unsloth start hermes` | Cần `--persist` để giữ phiên |
| OpenClaw | `unsloth start openclaw` | Cần `--persist` (config, workspace, session mặc định là tạm) |
| Pi Agent | `unsloth start pi` | Cần `--persist` để giữ phiên |

Nạp model ngay khi khởi động agent (hậu tố `:UD-Q4_K_XL` chọn quant GGUF). Nếu Unsloth chưa chạy, `--model` sẽ khởi động một server tạm và tắt nó khi agent thoát; nếu Unsloth đang chạy thì chỉ kết nối vào:

```bash
unsloth start claude \
  --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \
  --context-length 32768 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --reasoning-effort medium
```

::: warning Docs chưa thống nhất
Lệnh nạp model kèm agent/server được viết khác nhau (tên model, tên agent, dấu `\` xuống dòng). Lệnh ở trên lấy từ [integrations/unsloth-start](https://unsloth.ai/docs/integrations/unsloth-start). Các biến thể khác trong docs, chép nguyên văn:

[basics/claude-code](https://unsloth.ai/docs/basics/claude-code):

```bash
unsloth start claude \
    --model unsloth/qwen3.8-27B-GGUF-GGUF:UD-Q4_K_XL
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --reasoning-effort medium
```

[basics/codex](https://unsloth.ai/docs/basics/codex) (mục "Run OpenAI Codex with `unsloth start`"; cùng trang ở dưới dùng `unsloth start codex`):

```bash
unsloth start Code \
    --model unsloth/qwen3.8-27B-GGUF-GGUF:UD-Q4_K_XL
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --reasoning-effort medium
```

[integrations/opencode](https://unsloth.ai/docs/integrations/opencode):

```bash
unsloth start opencode \
    --model unsloth/qwen3.8-27B-GGUF-GGUF:UD-Q4_K_XL
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --reasoning-effort medium
```

[basics/api](https://unsloth.ai/docs/basics/api) (mục "Unsloth run command"):

```bash
unsloth run --model unsloth/qwen3.8-27B-GGUF-GGUF:UD-Q4_K_XL
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

| Thông số | [integrations/unsloth-start](https://unsloth.ai/docs/integrations/unsloth-start) | [basics/claude-code](https://unsloth.ai/docs/basics/claude-code), [basics/codex](https://unsloth.ai/docs/basics/codex), [integrations/opencode](https://unsloth.ai/docs/integrations/opencode), [basics/api](https://unsloth.ai/docs/basics/api) |
| --- | --- | --- |
| Tên model | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` | `unsloth/qwen3.8-27B-GGUF-GGUF:UD-Q4_K_XL` |
| `\` sau dòng `--model` | Có | Không |
| Tên agent Codex | `codex` | `Code` (chỉ ở một khối của basics/codex) |
| Reasoning effort | `--reasoning-effort medium` | `--reasoning-effort medium`; riêng basics/api dùng `--chat-template-kwargs '{"reasoning_effort":"medium"}'` |
:::

Tham số không thuộc Unsloth được chuyển nguyên cho agent:

```bash
unsloth start claude --continue
unsloth start codex --persist resume --last
unsloth start opencode run --continue "Continue the previous task"
unsloth start pi --persist --continue
```

Các tùy chọn đáng chú ý:

| Tùy chọn | Tác dụng |
| --- | --- |
| `--model`, `-m` | Chọn model; bỏ trống thì dùng model đầu tiên Unsloth báo |
| `--api-key` | Truyền API key (hoặc biến `UNSLOTH_API_KEY`) |
| `--launch` / `--no-launch` | Chạy agent, hoặc chỉ in môi trường + lệnh sinh ra |
| `--serve` / `--no-serve` | Cho/không cho tự khởi động server local |
| `--reasoning` | `on`, `off` hoặc `auto` |
| `--context-length`, `--max-seq-length` | Context length khi nạp model |
| `--persist` / `--no-persist` | Giữ thư mục home do Unsloth quản lý |
| `--yolo` | Chế độ không hỏi xác nhận của agent |

::: warning Docs chưa thống nhất
Ví dụ về cờ reasoning trong [basics/api](https://unsloth.ai/docs/basics/api) (mục "Control reasoning behavior") có comment và giá trị cờ ngược nhau, chép nguyên văn:

```bash
# Disable reasoning / thinking output
unsloth run \
  --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \
  --reasoning on  \
  --reasoning-effort medium
```

Trong khi đó [integrations/connect-python-sdk-to-unsloth](https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth) và [basics/claude-code](https://unsloth.ai/docs/basics/claude-code) ghi: "Use `--reasoning off` to turn thinking off, or `--reasoning on` to turn it on". [integrations/unsloth-start](https://unsloth.ai/docs/integrations/unsloth-start) ghi `--reasoning` nhận `on`, `off` hoặc `auto`.
:::

Kết nối tới Unsloth server ở xa:

```bash
export UNSLOTH_STUDIO_URL=https://studio.example.com
export UNSLOTH_API_KEY=sk-unsloth-...
unsloth start claude
```

### Kết nối thủ công (khi không dùng `unsloth start`)

**Claude Code** đọc biến môi trường (chỉ có hiệu lực trong phiên shell hiện tại):

```bash
export ANTHROPIC_BASE_URL="http://localhost:8888"
```

```bash
export ANTHROPIC_AUTH_TOKEN="sk-unsloth-xxxxxxxxxxxx"
```

```bash
export ANTHROPIC_API_KEY=""
```

```bash
export ANTHROPIC_MODEL="unsloth/gemma-4-26B-A4B-it-GGUF"
```

```shellscript
claude --model unsloth/gemma-4-26B-A4B-it-GGUF
```

Claude Code gắn một attribution header vào đầu system prompt, giá trị đổi theo mỗi request, làm KV cache mất hiệu lực và inference với model local **chậm hơn 90%**. Cách tắt ngay khi chạy:

```bash
claude --settings '{"env":{"CLAUDE_CODE_ATTRIBUTION_HEADER":"0","CLAUDE_CODE_ENABLE_TELEMETRY":"0"}}' --model unsloth/gemma-4-26B-A4B-it-GGUF
```

Tùy chọn thu gọn system prompt để tái dùng KV cache tốt hơn:

```shellscript
claude --model unsloth/gemma-4-26B-A4B-it-GGUF --bare --exclude-dynamic-system-prompt-sections
```

**Codex** chỉ dùng OpenAI Responses API (bỏ hỗ trợ Chat Completions), cấu hình trong `~/.codex/config.toml` (Windows: `%USERPROFILE%\.codex\config.toml`):

```toml
# Default local provider used with `codex --oss`
oss_provider = "unsloth_api"

[model_providers.unsloth_api]
name                  = "Unsloth Studio"
base_url              = "http://localhost:8888/v1"
env_key               = "UNSLOTH_STUDIO_AUTH_TOKEN"
wire_api              = "responses"
requires_openai_auth  = false
```

```toml
model_provider = "unsloth_api"
model = "unsloth/gemma-4-26B-A4B-it-GGUF"
```

```bash
mkdir my-project && cd my-project
codex --oss --profile unsloth_api
```

`env_key` là **tên** biến môi trường chứa key, không phải key. Đừng chạy `codex` trần trước khi cấu hình: nó mở màn hình "Sign in with ChatGPT" không thoát được.

::: warning Docs chưa thống nhất
Cấu hình Codex trong [basics/codex](https://unsloth.ai/docs/basics/codex) mô tả vị trí profile và cách chọn profile theo nhiều cách khác nhau:

| Thông số | Cách ghi 1 | Cách ghi 2 |
| --- | --- | --- |
| Nơi đặt profile `unsloth_api` | Khối `model_provider = "unsloth_api"` / `model = ...` "create a Codex profile", không có tiêu đề bảng TOML; mục "Disconnect" nhắc khối `[profiles.unsloth_api]` trong `~/.codex/config.toml` | "The `--profile unsloth_api` flag tells Codex to load `~/.codex/unsloth_api.config.toml`" |
| Lệnh chạy | `codex --oss --profile unsloth_api` | Mục Disconnect: "Launch Codex without `-p unsloth_api`" |
| Lệnh chạy với llama.cpp | `codex --oss llama_cpp` (mục llama.cpp) | "use the same shape with the `llama_cpp` profile"; mục Setup: `codex --oss --profile llama_cpp` |
| `requires_openai_auth` trong provider llama.cpp | Provider `unsloth_api` có `requires_openai_auth = false` | Provider `llama_cpp` không ghi dòng này (docs nói mặc định đã là `false`) |
| Port | Unsloth Studio: `http://localhost:8888/v1` | llama-server: `http://localhost:8001/v1` |

Ngoài ra `unsloth start codex` ([integrations/unsloth-start](https://unsloth.ai/docs/integrations/unsloth-start)) tạo Codex home riêng, không dùng `~/.codex`.
:::

**OpenCode Desktop:** `/model` → **Connect provider** → **Custom**, điền Provider ID `unsloth-studio`, Base URL `http://localhost:8888/v1/` (giữ đuôi `/v1/`), API key `sk-unsloth-…`, thêm model ID đúng như Unsloth phục vụ (vd `unsloth/Qwen3.6-27B-GGUF`). Khởi động lại opencode thì provider mới xuất hiện.

**Khi tự chạy `unsloth run` cho agent**, thêm `--disable-tools`:

```bash
# Serve for a coding agent: --disable-tools passes the agent's own tools through
unsloth run \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  --disable-tools \
  --reasoning off \
  -p 8888
```

Mặc định Unsloth chạy server-side tools của chính nó và "nuốt" tool call của agent, khiến agent trả lời nhưng không bao giờ sửa file. `--disable-tools` chuyển sang passthrough (chuyển tiếp) để agent dùng tool Write/Edit/Bash của nó.

**Nguồn:** https://unsloth.ai/docs/integrations/unsloth-start, https://unsloth.ai/docs/basics/claude-code, https://unsloth.ai/docs/basics/codex, https://unsloth.ai/docs/integrations/opencode

## Connections: nhà cung cấp API và model server

Connections cho phép dùng model bên ngoài trong cùng giao diện chat của Unsloth, và vẫn gắn được bộ công cụ của Unsloth (web search, code execution, deep research). Có hai nhóm:

| Kết nối | Loại | Cần gì | Base URL mẫu | Model xuất hiện ở |
| --- | --- | --- | --- | --- |
| OpenAI (cả ChatGPT/Codex subscription) | Cloud | API key từ OpenAI dashboard | (không cần) | **Connected** |
| Anthropic | Cloud | API key từ Anthropic Console | (không cần) | **Connected** |
| OpenRouter | Cloud, nhiều model qua 1 key | API key OpenRouter | (không cần) | **Connected** |
| llama.cpp (`llama-server`) | Model server | Key chỉ khi server chạy với `--api-key` | `http://localhost:8080/v1` | **Connected** / **External** (nguồn ghi không thống nhất) |
| vLLM | Model server, throughput cao | Key chỉ khi chạy với `--api-key` | `http://localhost:8000/v1` | **Connected** |
| Ollama | Model server đơn giản | Thường không cần key | `http://localhost:11434` hoặc `http://localhost:11434/v1` | **Connected** |

### Các bước với nhà cung cấp cloud

1. Tạo API key trên dashboard của nhà cung cấp.
2. Trong Unsloth: **Settings** → **Connections** → **Add Connection**.
3. Chọn nhà cung cấp, dán API key.
4. Bấm **Reload Models** để lấy danh sách model tài khoản được dùng; chọn model muốn bật rồi lưu.
5. Chọn model trong dropdown **Select Model**, nhóm **Connected**.

Với OpenRouter, nếu **Load Models** không trả về model cần, nhập tay model ID, ví dụ:

```
openai/gpt-5.5 
anthropic/claude-sonnet-4.6 
google/gemini-3-pro
```

### Các bước với model server

Khởi động server trước. llama.cpp:

```bash
llama-server \
  --model /path/to/model.gguf \
  --host 0.0.0.0 \
  --port 8080
```

Hoặc lấy GGUF trực tiếp từ Hugging Face:

`llama-server -hf unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL`

vLLM:

```bash
vllm serve unsloth/gemma-4-26B-A4B-it \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key token-abc123 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

::: warning Docs chưa thống nhất
Lệnh vLLM ở trên lấy từ mục "Common vLLM arguments" của [integrations/connections/vllm](https://unsloth.ai/docs/integrations/connections/vllm). Các biến thể lệnh tối thiểu trong docs, chép nguyên văn:

[integrations/connections](https://unsloth.ai/docs/integrations/connections):

```bash
  vllm serve unsloth/gemma-4-26B-A4B-it \
  --dtype auto \
```

[integrations/connections/vllm](https://unsloth.ai/docs/integrations/connections/vllm) (mục "Choose a model"):

```bash
vllm serve unsloth/gemma-4-26B-A4B-it 
\ --dtype auto
```
:::

Ollama:

```bash
ollama pull qwen3.6:35b-a3b
ollama run qwen3.6:35b-a3b
```

Sau đó: **Settings → Connections** → thêm kết nối → chọn llama.cpp / vLLM / Ollama → dán **Base URL** → bấm **Load Models** (hoặc nhập model ID tay nếu server không có `/models`) → lưu. Với vLLM, bật tùy chọn **Reasoning model** nếu model hỗ trợ thinking.

::: warning Docs chưa thống nhất
Base URL của Ollama khi thêm vào Connections:

| Thông số | [integrations/connections](https://unsloth.ai/docs/integrations/connections) | [integrations/connections/ollama](https://unsloth.ai/docs/integrations/connections/ollama) |
| --- | --- | --- |
| Base URL | `http://localhost:11434/v1` | "In most local setups": `http://localhost:11434`; "If Unsloth asks for an OpenAI-compatible base URL": `http://localhost:11434/v1` |
| Model mẫu | `qwen3:14b` | `qwen3.6:35b-a3b`; model ID nhập tay ví dụ `qwen3.6` |
:::

### Tính năng đi kèm

- **Prompt caching** (tái dùng phần đầu prompt dài giống nhau để giảm độ trễ và chi phí): hỗ trợ OpenAI, Anthropic, llama.cpp; chỉnh ở mục **Prompt caching** trong side panel. Với llama.cpp, caching bật mặc định, tắt bằng `--no-cache-prompt` khi khởi động `llama-server`.
- **Code execution phía nhà cung cấp:** Anthropic dùng Code execution tool của Claude; OpenAI dùng container tái sử dụng (tạo/xóa/chọn trong **Code Execution** settings; chọn lại cùng container ở thread mới để giữ file và trạng thái).
- **Web search & Thinking** phía nhà cung cấp: hỗ trợ model của OpenAI, Anthropic, OpenRouter, Mistral, Gemini, Kimi. Nút **Think** đổi theo model (bật/tắt hoặc các mức reasoning effort).
- **Image generation**: có nút "Edit Image" và tải ảnh độ phân giải gốc.

**Nguồn:** https://unsloth.ai/docs/integrations/connections, https://unsloth.ai/docs/integrations/connections/openai, https://unsloth.ai/docs/integrations/connections/anthropic-claude, https://unsloth.ai/docs/integrations/connections/openrouter, https://unsloth.ai/docs/integrations/connections/connect-llama.cpp-to-unsloth-run-ggufs-with-llama-server, https://unsloth.ai/docs/integrations/connections/vllm, https://unsloth.ai/docs/integrations/connections/ollama

## MCP

MCP (Model Context Protocol, giao thức chuẩn để model gọi công cụ/dịch vụ bên ngoài) cho phép model local như Qwen hay Gemma dùng file, app, cơ sở dữ liệu, dịch vụ của bạn thay vì chỉ trả lời từ trí nhớ. MCP chạy được với cả GGUF local lẫn model của nhà cung cấp cloud đã kết nối, và dùng song song với code execution, web search trong cùng một thread.

### Bật MCP trong Unsloth Studio

1. Bấm **MCP** trên thanh công cụ chat. Studio có sẵn MCP server của **Context7, Exa và Hugging Face**. Bật Exa sẽ tắt công cụ web search mặc định của Unsloth.
2. Thêm server riêng: **Add custom MCP**, điền **Display name** (vd `Vercel`) và **URL** là endpoint gốc của server (vd `https://mcp.vercel.com`).
3. Chọn cách xác thực: server dùng đăng nhập qua trình duyệt (GitHub, Linear, Vercel…) thì bật **Use OAuth sign-in**; server dùng token thì để OAuth tắt, bấm **Add header** và thêm:

```
Authorization: Bearer <your-token>
```

4. Bấm **Test connection**, thành công thì **Add server**. Unsloth tự lấy danh sách tool, vd *Refreshed "Vercel" (18 tools)*.
5. Bật cả công tắc của từng server lẫn công tắc tổng **Use MCP Servers**, rồi chọn model và chat; model tự gọi tool khi cần.

Ví dụ trong tài liệu dùng cùng lúc 3 server mặc định: hỏi về fine-tune Qwen (Exa trả lời), tìm trong docs Unsloth (Context7), tìm model `unsloth/Qwen` trên Hugging Face (MCP của Hugging Face).

### Ví dụ cấu hình với llama.cpp

Khi không dùng Studio, tài liệu dùng `llama-server` + `mcp-cli` (MCP client dòng lệnh của IBM). File `server_config.json` cho MCP server filesystem:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/ABSOLUTE/PATH/TO/mcp-workspace"
      ],
      "env": {}
    }
  }
}
```

```bash
# terminal 1
llama-server -hf unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL \
  --alias local --host 127.0.0.1 --port 8080 --no-ui \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --chat-template-kwargs '{"enable_thinking":false}'

# terminal 2, in the folder with server_config.json
uvx mcp-cli \
  --provider llamacpp \
  --api-base http://127.0.0.1:8080/v1 \
  --api-key none \
  --model local \
  --server filesystem \
  --config-file server_config.json
```

`mcp-cli` cần thêm file cấu hình `~/.chuk_llm/config.yaml` trước khi chạy (xem trang nguồn), và mặc định hỏi xác nhận trước mỗi lần chạy tool.

::: danger Bảo mật MCP
Chỉ kết nối MCP server tin cậy. Giữ bước xác nhận của người dùng cho hành động đọc dữ liệu riêng tư, đổi deployment, mua domain, sửa project. Cẩn thận khi kết hợp MCP với web search: nội dung web bị prompt injection (chèn lệnh độc hại vào nội dung) có thể kích hoạt tool call ngoài ý muốn.
:::

**Nguồn:** https://unsloth.ai/docs/basics/mcp

## Tool calling

Tool calling (gọi công cụ) là khi LLM được phép kích hoạt hàm cụ thể ("tìm file", "chạy máy tính", "gọi API") bằng cách xuất ra một request có cấu trúc thay vì đoán câu trả lời bằng văn bản. Model chỉ **đề xuất** lời gọi (tên hàm + tham số JSON); chương trình của bạn chạy hàm rồi gửi kết quả lại cho model ở lượt sau.

::: tip Kiến thức nền
Cơ chế tool calling và vai trò của chat template: xem [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling).
:::

### Unsloth hỗ trợ thế nào

- **Trong Studio:** tool calling được cài sẵn, chỉ cần chọn model và bật/tắt. Có self-healing (tự sửa tool call hỏng), khử trùng lặp, chống XML lọt vào output, cho phép hơn 25 tool call, và dừng vòng gọi ổn định hơn. Nguồn ghi tool call "chính xác hơn 30% đến 80%" trên mọi model.
- **Qua API, client-side tools:** gửi `tools` + `tool_choice` kiểu OpenAI tới `/v1/chat/completions`, hoặc `tools` (có `input_schema`) kiểu Anthropic tới `/v1/messages`. Client tự chạy hàm và trả kết quả. Ánh xạ `tool_choice`: Anthropic `auto` → OpenAI `auto`, `any` → `required`, `none` → `none`.
- **Qua API, server-side tools:** Unsloth tự chạy Python, bash, web search và stream kết quả về dạng sự kiện `tool_result`. Bật bằng các trường thêm:

| Trường | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `enable_thinking` | `boolean` | `false` để tắt thinking; mặc định `true` |
| `enable_tools` | `boolean` | `true` để bật chạy tool phía server |
| `enabled_tools` | `array<string>` | Tool model được gọi: `python`, `bash`, `web_search` |
| `session_id` | `string` | Tùy chọn; giữ trạng thái tool (vd Python kernel) giữa các lần gọi |

::: warning Docs chưa thống nhất
Tên tool hợp lệ trong `enabled_tools`:

| Thông số | [basics/api](https://unsloth.ai/docs/basics/api) | [integrations/connect-python-sdk-to-unsloth](https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth), [integrations/connect-curl-and-http-to-unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth) |
| --- | --- | --- |
| Ví dụ JSON | `["python", "web_search","terminal"]` | `["python", "web_search"]`, `["python"]`, `["web_search", "python"]` |
| Danh sách hỗ trợ | Văn bản: "execute Python, web search, and bash"; Troubleshooting ví dụ `["python", "web_search"]` | "supports `python`, `bash`, and `web_search`" |
:::

Ví dụ function calling với OpenAI SDK (client-side):

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
                },
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Perth right now?"}],
    tools=tools,
    tool_choice="auto",
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name, tool_call.function.arguments)
```

Server-side tools qua OpenAI SDK (dùng `extra_body` để các trường riêng của Unsloth đi thẳng tới server):

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 123 * 456? Use Python to compute it."}],
    stream=True,
    extra_body={
        "enable_tools": True,
        "enabled_tools": ["python", "web_search"],
        "session_id": "my-session",
    },
)
for chunk in stream:
    if chunk.choices:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
```

### Vòng lặp tool calling tự viết với llama-server

Tool Calling Guide chạy model bằng `llama-server` (port `8001`), định nghĩa sẵn các hàm Python (`add_number`, `multiply_number`, `subtract_number`, `write_a_story`, `terminal`, `python`), bảng `MAP_FN` ánh xạ tên → hàm và danh sách `tools` theo schema OpenAI (xem đầy đủ ở trang nguồn). Hàm dưới đây gọi endpoint OpenAI-compatible, đọc `tool_calls`, chạy hàm tương ứng và nối kết quả vào `messages` với `role: "tool"`:

```python
from openai import OpenAI
def unsloth_inference(
    messages,
    temperature = 0.7,
    top_p = 0.95,
    top_k = 40,
    min_p = 0.01,
    repetition_penalty = 1.0,
):
    messages = messages.copy()
    openai_client = OpenAI(
        base_url = "http://127.0.0.1:8001/v1",
        api_key = "sk-no-key-required",
    )
    model_name = next(iter(openai_client.models.list())).id
    print(f"Using model = {model_name}")
    has_tool_calls = True
    original_messages_len = len(messages)
    while has_tool_calls:
        print(f"Current messages = {messages}")
        response = openai_client.chat.completions.create(
            model = model_name,
            messages = messages,
            temperature = temperature,
            top_p = top_p,
            tools = tools if tools else None,
            tool_choice = "auto" if tools else None,
            extra_body = {"top_k": top_k, "min_p": min_p, "repetition_penalty" :repetition_penalty,}
        )
        tool_calls = response.choices[0].message.tool_calls or []
        content = response.choices[0].message.content or ""
        tool_calls_dict = [tc.to_dict() for tc in tool_calls] if tool_calls else tool_calls
        messages.append({"role": "assistant", "tool_calls": tool_calls_dict, "content": content,})
        for tool_call in tool_calls:
            fx, args, _id = tool_call.function.name, tool_call.function.arguments, tool_call.id
            out = MAP_FN[fx](**json.loads(args))
            messages.append({"role": "tool", "tool_call_id": _id, "name": fx, "content": str(out),})
        else:
            has_tool_calls = False
    return messages
```

Gọi thử:

```python
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "What is today's date plus 3 days?"}],
}]
unsloth_inference(messages, temperature = 0.15, top_p = 1.0, top_k = -1, min_p = 0.00)
```

Ví dụ này dùng Devstral 2; đổi model thì phải dùng đúng sampling parameters khuyến nghị cho model đó. Server llama.cpp cần cờ `--jinja` để tool calling dùng đúng chat template.

**[Nhận định]** Trong Python, `for ... else` chạy nhánh `else` khi vòng `for` kết thúc không có `break`, nên đoạn code trên luôn đặt `has_tool_calls = False` sau lượt đầu: nó chạy tool đúng một vòng rồi dừng, không tự gửi kết quả tool lại cho model. Nếu cần vòng lặp nhiều bước, phải tự sửa điều kiện dừng.

**Nguồn:** https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms, https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/new/studio/chat

## Cạm bẫy thường gặp

::: warning Lỗi hay gặp khi dùng API và agent
- **`401 Unauthorized`**: thiếu header hoặc key sai. Header phải là `Authorization: Bearer sk-unsloth-…`. Mất key thì tạo key mới ở **Settings → API**; key cũ không xem lại được.
- **`404 Not Found` với SDK**: OpenAI SDK cần `base_url` **có** đuôi `/v1`; Anthropic SDK cần `base_url` **không có** `/v1` (SDK tự thêm `/v1/messages`).
- **Quên `max_tokens` ở `/v1/messages`**: endpoint Anthropic bắt buộc trường này.
- **Streaming ra một cục JSON / curl im lặng**: dùng `curl -N`; trong script dùng `print(..., flush=True)`; sau proxy thì tắt buffering. Qua Cloudflare tunnel (`--secure`) phải đặt `stream: false`.
- **Agent trả lời nhưng không sửa file**: server-side tools của Unsloth đang nuốt tool call của agent. Chạy `unsloth run` với `--disable-tools`.
- **Server-side tools không chạy**: phải có `enable_tools: true` **và** liệt kê tool trong `enabled_tools`; server bind `0.0.0.0` thì tools mặc định tắt.
- **Claude Code chậm hơn 90% với model local**: attribution header làm KV cache mất hiệu lực, cần đặt `CLAUDE_CODE_ATTRIBUTION_HEADER` = `0`.
- **Claude Code vẫn dùng model Anthropic mặc định**: kiểm tra `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` được export trong **cùng** shell chạy `claude`. Gặp `Unable to connect to API (ConnectionRefused)` sau khi thôi dùng Unsloth thì `unset ANTHROPIC_BASE_URL`.
- **Codex không kết nối**: phải là model GGUF qua `llama-server`; `wire_api` phải là `"responses"` (`"chat"` đã bị bỏ).
- **Sai model ID**: gọi `GET /v1/models` và chép đúng trường `id`.
- **Câu trả lời bị cắt ngắn**: xem **Context used** trong API monitor; gần 100% hoặc stop reason `length` nghĩa là context đã đầy.
- **`Lost connection to the model server`**: llama.cpp phía sau bị crash hoặc tab model đã đóng; nạp lại model từ **New Chat**.
- **Lộ API key**: ai có key và truy cập được mạng tới Unsloth đều gửi được request tới model. Output của `unsloth start ... --no-launch` cũng chứa thông tin đăng nhập.
:::

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/basics/claude-code, https://unsloth.ai/docs/basics/codex, https://unsloth.ai/docs/integrations/unsloth-start
