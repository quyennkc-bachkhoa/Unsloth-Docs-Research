---
title: "API tương thích OpenAI và Anthropic"
description: "Endpoint, port, API key, ví dụ curl, Python với OpenAI SDK và Anthropic SDK, cấu hình server và truy cập từ máy khác."
---

# API tương thích OpenAI và Anthropic

Unsloth mở model đang nạp thành một API giống hệt API của OpenAI và Anthropic. Code đã viết cho hai hãng đó chạy được với model local, chỉ cần đổi địa chỉ và key.

## Endpoint, port và API key

Unsloth nói "hai phương ngữ" trên **cùng một port**. Cả hai đều hỗ trợ streaming (trả kết quả từng phần), tool calling và ảnh đầu vào:

| Endpoint | Tương thích với | Dùng từ |
| --- | --- | --- |
| `POST /v1/messages` | Anthropic Messages API | Claude Code, Anthropic SDK, OpenClaw |
| `POST /v1/chat/completions` | OpenAI Chat Completions API | OpenAI SDK, opencode, Cursor, Continue, Cline, Open WebUI, curl |
| `POST /v1/responses` | OpenAI Responses API | Codex và các client OpenAI đời mới |
| `GET /v1/models` | OpenAI models list | Liệt kê model đang nạp (lấy `id` để điền vào trường model) |

::: warning Docs chưa thống nhất
Các trang docs liệt kê endpoint không giống nhau:

- Bảng "Endpoints" của [basics/api](https://unsloth.ai/docs/basics/api) chỉ có `POST /v1/messages`, `POST /v1/chat/completions`, `GET /v1/models`. Bảng này không có `/v1/responses`. Nhưng phần mở đầu của cùng trang lại ghi "OpenAI-compatible `/v1/chat/completions` and **`/v1/responses`**".
- [integrations/connect-curl-and-http-to-unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth) có ví dụ riêng cho `/v1/responses`. [basics/codex](https://unsloth.ai/docs/basics/codex) cũng dùng endpoint `/v1/responses`.

Dòng `/v1/responses` trong bảng trên được tổng hợp từ hai trang sau.
:::

- **Port:** xem hộp bên dưới. Các ví dụ trên trang này dùng `8888` như trong docs.
- **API key:** mọi request cần header `Authorization: Bearer sk-unsloth-…`.

::: warning Docs chưa thống nhất
Mỗi trang docs ghi port và base URL mặc định của API Unsloth một kiểu:

| Trang | Port / base URL ghi trong docs |
| --- | --- |
| [basics/api](https://unsloth.ai/docs/basics/api) | "typically `http://localhost:8000` or `http://localhost:8888`"; mục `--secure` nhắc `http://127.0.0.1:8888`; ví dụ curl dùng `8888` |
| [integrations/connect-python-sdk-to-unsloth](https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth) | "typically `8000` or `8888`"; code dùng `http://localhost:8888/v1` (OpenAI SDK) và `http://localhost:8888` (Anthropic SDK) |
| [integrations/connect-curl-and-http-to-unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth) | "on the port Unsloth started on"; ví dụ dùng `8888` |
| [basics/claude-code](https://unsloth.ai/docs/basics/claude-code) | `ANTHROPIC_BASE_URL="http://localhost:8888"` |
| [basics/codex](https://unsloth.ai/docs/basics/codex) | `base_url = "http://localhost:8888/v1"`, và ghi `8888` là "Unsloth Studio's default" |
| [integrations/opencode](https://unsloth.ai/docs/integrations/opencode) | `http://localhost:8888/v1/` (có dấu `/` cuối, "keep the trailing `/v1/`") |

Cách chắc chắn để biết port: `unsloth run` in endpoint URL ra console ([basics/api](https://unsloth.ai/docs/basics/api)).
:::

Có hai cách lấy API key. Cách thứ nhất là tạo trong giao diện:

1. Bấm avatar **Unsloth** ở góc dưới bên trái sidebar.
2. Vào **Settings** → **API**.
3. Nhập một tên dễ nhớ, ví dụ `claude-code-macbook`. Đặt hạn dùng nếu muốn.
4. Bấm **Create** rồi **chép key ngay**. Unsloth chỉ lưu hash của key nên sẽ không hiển thị lại.

Request dùng key đã bị thu hồi sẽ nhận `401 Unauthorized`.

Cách thứ hai là dùng `unsloth run`. Lệnh này nạp model, tự tạo API key, rồi in endpoint và key ra console:

```bash
unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
```

Bạn có thể viết tên model theo ba cách, kết quả như nhau:

```bash
# Combined: repo and quantization variant in one string (recommended — shortest)
unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL

# Separate: repo and variant as two flags (the older style, still works)
unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF --gguf-variant UD-Q4_K_XL

# Using -hf / --hf-repo (matches llama.cpp's spelling, handy if you're coming from there)
unsloth run -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
```

**[Nhận định]** Nếu bạn quen FastAPI: hãy hình dung Unsloth là một app có sẵn các route `POST /v1/...`, kèm dependency kiểm tra Bearer token. Client chỉ cần đổi `base_url` và `api_key`. Schema request và response giữ nguyên như khi gọi OpenAI hoặc Anthropic thật. Bạn không phải tự viết server bọc model.

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth

## Ví dụ curl

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
Model ID là giá trị Unsloth trả về và bạn cần điền vào trường `model`. Docs ghi ID này theo hai dạng: có và không có tiền tố `unsloth/`:

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

Streaming: thêm `"stream": true`. Response khi đó chuyển sang SSE (Server-Sent Events, luồng sự kiện `text/event-stream`). Thêm `-N` để curl in ra ngay, không gom vào bộ đệm:

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

Anthropic Messages. Ở endpoint này `max_tokens` là **bắt buộc**. Ở `/v1/chat/completions` thì trường này tùy chọn:

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

Bạn cũng ghi đè được tham số sinh cho từng request. Giá trị ghi trong request được ưu tiên hơn giá trị mặc định của server:

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

Thinking là chế độ model "suy nghĩ" trước khi trả lời. Chế độ này bật mặc định. Muốn tắt, gửi `enable_thinking: false` trong request.

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth

## Python: OpenAI SDK

Đặt key vào biến môi trường để không phải dán thẳng key vào code:

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

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth

## Python: Anthropic SDK

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

Chọn SDK nào:

- **OpenAI SDK** nếu code của bạn đã phụ thuộc gói `openai`, hoặc bạn muốn dùng `tools`/`tool_choice` kiểu OpenAI, hoặc cần Responses API.
- **Anthropic SDK** nếu bạn đã dùng gói `anthropic`, hoặc thích định dạng tool `input_schema` và kiểu sự kiện streaming của Anthropic.

Bạn có thể dùng cả hai trong một project với cùng một key.

API còn hỗ trợ ảnh đầu vào (vision). Khi đó model phải là multimodal. API cũng hỗ trợ structured output qua `response_format` với JSON Schema. Đặt `strict: True` để ép output đúng schema ngay lúc decode. Chi tiết xem ở trang nguồn Python SDK.

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth

## Cấu hình server và truy cập từ máy khác

- Mặc định Unsloth chỉ nhận kết nối từ chính máy đang chạy nó. Muốn máy khác trong LAN kết nối được, bind vào `0.0.0.0`:

```bash
# Allow LAN devices to connect
unsloth run \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  -H 0.0.0.0 \
  -p 8888
```

- **Chính sách server-side tools.** Server-side tools là công cụ do Unsloth tự chạy, như web search, code execution… Mặc định của chúng tùy địa chỉ bind:
  - Trên `127.0.0.1`: mặc định **bật**.
  - Trên `0.0.0.0` hoặc địa chỉ không phải loopback: mặc định **tắt**. Lý do: nếu API key bị lộ trên một server mở ra mạng, người khác có thể chạy code tùy ý trên máy bạn.
  - Ép bật hoặc tắt bằng `--enable-tools` / `--disable-tools`. Trên `0.0.0.0`, `--enable-tools` sẽ hỏi xác nhận y/N. Thêm `--yes`/`-y` để bỏ qua câu hỏi này.
  - Request không thể vượt chính sách này bằng `enable_tools=true`.
- **Truy cập từ xa qua Internet:** chạy `unsloth studio --secure`. Unsloth vẫn bind localhost, nhưng mở thêm một URL HTTPS Cloudflare miễn phí. SSE không đi qua được Cloudflare quick tunnel, nên bạn cần đặt `stream: false`. Trường hợp này server bind localhost nhưng vẫn truy cập được từ Internet. Docs không nói chính sách server-side tools áp dụng ra sao lúc đó: **cần kiểm tra lại**.

::: warning Docs chưa thống nhất
Server-side tools "bật mặc định" hay "phải tự bật"? Các trang mô tả khác nhau:

| Trang | Mô tả trong docs |
| --- | --- |
| [basics/api](https://unsloth.ai/docs/basics/api) (Server-side tool policy) | `127.0.0.1`: tools **on** by default; `0.0.0.0` hoặc địa chỉ non-loopback: tools **off** by default |
| [basics/api](https://unsloth.ai/docs/basics/api) (Tool calling, Troubleshooting) | "Opt in by adding these extra fields"; "remember to set `enable_tools: true` **and** list the ones you want in `enabled_tools`" |
| [integrations/connect-curl-and-http-to-unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth), [integrations/connect-python-sdk-to-unsloth](https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth) | Opt in bằng `enable_tools: true` trong request |
| [basics/claude-code](https://unsloth.ai/docs/basics/claude-code), [integrations/opencode](https://unsloth.ai/docs/integrations/opencode) | "By default Unsloth Studio runs its own server-side tools, which swallows the agent's tool calls"; khuyên thêm `--disable-tools` |
| [basics/codex](https://unsloth.ai/docs/basics/codex) | "When driving an external coding agent, add `--disable-tools`" |

Ví dụ LAN trong [basics/claude-code](https://unsloth.ai/docs/basics/claude-code) dùng `-H 0.0.0.0` mà không có `--disable-tools`; ví dụ LAN trong [integrations/opencode](https://unsloth.ai/docs/integrations/opencode) có `--disable-tools`.
:::
- **API monitor:** mọi request dùng API key đều hiện trong Studio, ở panel góc màn hình và trang **Settings → API Monitor**. Mỗi request có:
  - prompt và response;
  - số token, time-to-first-token, throughput;
  - lỗi, và **Context used**.

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth
