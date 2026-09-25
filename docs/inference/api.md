---
title: "API tương thích OpenAI và Anthropic"
description: "Endpoint, port, API key, ví dụ curl, Python với OpenAI SDK và Anthropic SDK, cấu hình server và truy cập từ máy khác."
---

# API tương thích OpenAI và Anthropic

Khi bạn nạp một model trong Unsloth, Unsloth có thể mở model đó thành một API. Từ đó, code Python, lệnh `curl` hay coding agent như Claude Code gửi câu hỏi tới model trên máy bạn, giống như gửi tới OpenAI hoặc Anthropic.

Điểm tiện nhất: API này dùng **đúng định dạng** request và response của OpenAI và Anthropic. Code bạn đã viết cho hai hãng đó chạy được với model local. Bạn chỉ cần đổi địa chỉ và key.

**[Nhận định]** Nếu bạn quen FastAPI: hãy hình dung Unsloth là một app có sẵn các route `POST /v1/...`, kèm dependency kiểm tra Bearer token. Client chỉ cần đổi `base_url` và `api_key`. Schema request và response giữ nguyên như khi gọi OpenAI hoặc Anthropic thật. Bạn không phải tự viết server bọc model.

Để gọi API, bạn cần ba thứ: **API key**, **địa chỉ** (base URL, gồm port) và **tên model**. Các mục dưới đi lần lượt từng thứ, rồi đến ví dụ gọi thật.

## Lấy API key

API key là chuỗi bí mật dạng `sk-unsloth-…`. Mọi request phải gửi kèm key này trong header:

```
Authorization: Bearer sk-unsloth-…
```

Thiếu header hoặc sai key, Unsloth trả về `401 Unauthorized`. Request dùng key đã bị thu hồi cũng nhận `401 Unauthorized`.

Có hai cách lấy key.

**Cách 1: tạo trong giao diện Studio.**

1. Bấm avatar **Unsloth** ở góc dưới bên trái sidebar.
2. Vào **Settings** → **API**.
3. Nhập một tên dễ nhớ, ví dụ `claude-code-macbook`. Đặt hạn dùng nếu muốn.
4. Bấm **Create** rồi **chép key ngay**. Unsloth chỉ lưu hash của key nên sẽ không hiển thị lại.

**Cách 2: dùng lệnh `unsloth run`.** Lệnh này nạp model, tự tạo API key, rồi in địa chỉ endpoint và key ra console:

```bash
unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
```

Tên model trong lệnh viết được theo ba cách, kết quả như nhau:

```bash
# Combined: repo and quantization variant in one string (recommended — shortest)
unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL

# Separate: repo and variant as two flags (the older style, still works)
unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF --gguf-variant UD-Q4_K_XL

# Using -hf / --hf-repo (matches llama.cpp's spelling, handy if you're coming from there)
unsloth run -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
```

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth

## Tìm địa chỉ API

Địa chỉ API có dạng `http://localhost:<port>`. `localhost` là chính máy bạn; port là "cổng" mà Unsloth mở ra để nhận request. Các ví dụ trên trang này dùng port `8888` như trong docs.

Docs không thống nhất port mặc định là bao nhiêu (xem hộp bên dưới). Cách chắc chắn nhất: nhìn dòng endpoint URL mà `unsloth run` in ra console.

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

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth

## Chọn endpoint

Endpoint là đường dẫn cụ thể trên địa chỉ API, mỗi đường dẫn nhận một kiểu request. Unsloth có hai "phương ngữ": một theo định dạng của Anthropic, một theo định dạng của OpenAI. Cả hai chạy trên **cùng một port**. Bạn chọn endpoint theo công cụ đang dùng:

| Endpoint | Tương thích với | Dùng từ |
| --- | --- | --- |
| `POST /v1/messages` | Anthropic Messages API | Claude Code, Anthropic SDK, OpenClaw |
| `POST /v1/chat/completions` | OpenAI Chat Completions API | OpenAI SDK, opencode, Cursor, Continue, Cline, Open WebUI, curl |
| `POST /v1/responses` | OpenAI Responses API | Codex và các client OpenAI đời mới |
| `GET /v1/models` | OpenAI models list | Liệt kê model đang nạp (lấy `id` để điền vào trường model) |

Cả hai phương ngữ đều hỗ trợ streaming (trả kết quả từng phần trong lúc model đang sinh), tool calling và ảnh đầu vào.

::: warning Docs chưa thống nhất
Các trang docs liệt kê endpoint không giống nhau:

- Bảng "Endpoints" của [basics/api](https://unsloth.ai/docs/basics/api) chỉ có `POST /v1/messages`, `POST /v1/chat/completions`, `GET /v1/models`. Bảng này không có `/v1/responses`. Nhưng phần mở đầu của cùng trang lại ghi "OpenAI-compatible `/v1/chat/completions` and **`/v1/responses`**".
- [integrations/connect-curl-and-http-to-unsloth](https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth) có ví dụ riêng cho `/v1/responses`. [basics/codex](https://unsloth.ai/docs/basics/codex) cũng dùng endpoint `/v1/responses`.

Dòng `/v1/responses` trong bảng trên được tổng hợp từ hai trang sau.
:::

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth

## Gửi request thử bằng curl

Có key và địa chỉ rồi, bạn thử gọi API bằng `curl` trong terminal. Trong các lệnh dưới, thay `sk-unsloth-xxxxxxxxxxxx` bằng key của bạn.

**Bước 1: hỏi xem model nào đang nạp.** Kết quả trả về trường `id`. Đó chính là tên model bạn điền vào trường `model` ở các request sau.

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

**Bước 2: gửi một câu hỏi.** Chọn một trong các endpoint dưới đây.

Theo định dạng OpenAI (Chat Completions):

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Theo định dạng Anthropic (Messages). Lưu ý: endpoint này **bắt buộc** có `max_tokens`, còn `/v1/chat/completions` thì không.

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

Theo định dạng OpenAI Responses API:

```bash
curl http://localhost:8888/v1/responses \
  -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-local",
    "input": "Write a one-sentence greeting."
  }'
```

**Nhận câu trả lời từng phần (streaming).** Mặc định, API đợi model sinh xong mới trả toàn bộ câu trả lời. Muốn thấy chữ hiện dần như khi chat, thêm `"stream": true`. Response khi đó chuyển sang SSE (Server-Sent Events, luồng sự kiện `text/event-stream`). Thêm `-N` để curl in ra ngay, không gom vào bộ đệm:

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

**Chỉnh tham số sinh cho một request.** Bạn đặt `temperature`, `top_p`, `max_tokens` ngay trong request. Giá trị trong request được ưu tiên hơn giá trị mặc định của server:

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

**Tắt chế độ thinking.** Thinking là chế độ model "suy nghĩ" trước khi trả lời, và được bật mặc định. Muốn tắt, gửi `enable_thinking: false` trong request.

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth

## Gọi từ Python

Trong Python, bạn không cần tự viết request HTTP. Hãy dùng thư viện chính thức của OpenAI hoặc Anthropic, chỉ đổi địa chỉ và key sang Unsloth.

Trước hết, đặt key vào biến môi trường để không phải dán thẳng key vào code:

```bash
export UNSLOTH_STUDIO_AUTH_TOKEN=sk-unsloth-xxxxxxxxxxxx
```

### Với OpenAI SDK

```bash
pip install openai
```

Tạo client. Chú ý `base_url` **có** đuôi `/v1`:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8888/v1",              # your unsloth port + /v1
    api_key=os.environ["UNSLOTH_STUDIO_AUTH_TOKEN"],     # your sk-unsloth-… key
)
```

Gửi câu hỏi và in câu trả lời:

```python
response = client.chat.completions.create(
    model="default",                               # the name you gave the model in unsloth or default
    messages=[
        {"role": "user", "content": "Give me two facts about Paris"}
    ],
)
print(response.choices[0].message.content)
```

Streaming, in từng phần câu trả lời ngay khi nhận được:

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

### Với Anthropic SDK

```bash
pip install anthropic
```

Tạo client. Có hai điểm khác OpenAI SDK:

- `base_url` **không có** `/v1`, vì SDK tự thêm vào.
- Key đi qua `default_headers`; còn `api_key` chỉ cần một giá trị bất kỳ, miễn không rỗng.

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

### Chọn SDK nào

- **OpenAI SDK** nếu code của bạn đã dùng gói `openai`, nếu bạn muốn dùng `tools`/`tool_choice` kiểu OpenAI, hoặc cần Responses API.
- **Anthropic SDK** nếu bạn đã dùng gói `anthropic`, hoặc thích định dạng tool `input_schema` và kiểu sự kiện streaming của Anthropic.

Bạn có thể dùng cả hai trong một project với cùng một key.

Ngoài chat bằng chữ, API còn nhận ảnh đầu vào (vision), nếu model là multimodal. API cũng hỗ trợ structured output: bạn gửi `response_format` kèm JSON Schema, đặt `strict: True`, và model buộc phải trả về đúng schema ngay lúc decode. Chi tiết xem ở trang nguồn Python SDK.

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth

## Mở API cho máy khác

Mặc định Unsloth chỉ nhận kết nối từ chính máy đang chạy nó. Mục này dành cho khi bạn muốn laptop, điện thoại hay server khác gọi được vào model.

### Trong cùng mạng LAN

Chạy Unsloth với `-H 0.0.0.0` (gọi là "bind vào `0.0.0.0`"). Khi đó mọi thiết bị trong mạng LAN đều kết nối được:

```bash
# Allow LAN devices to connect
unsloth run \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  -H 0.0.0.0 \
  -p 8888
```

### Qua Internet

Chạy `unsloth studio --secure`. Unsloth vẫn chỉ nghe trên máy bạn (localhost), nhưng mở thêm một URL HTTPS miễn phí qua Cloudflare để truy cập từ bất kỳ đâu.

Lưu ý: streaming (SSE) không đi qua được Cloudflare quick tunnel, nên bạn cần đặt `stream: false`.

### Server-side tools khi mở ra mạng

Server-side tools là các công cụ do chính Unsloth chạy trên máy bạn, như web search, code execution… Chúng hữu ích khi dùng một mình, nhưng nguy hiểm khi mở ra mạng: nếu API key bị lộ, người khác có thể chạy code tùy ý trên máy bạn. Vì vậy mặc định của chúng phụ thuộc địa chỉ bind:

- Bind `127.0.0.1` (chỉ máy bạn): mặc định **bật**.
- Bind `0.0.0.0` hoặc địa chỉ không phải loopback (mở ra mạng): mặc định **tắt**.

Bạn ép bật hoặc tắt bằng `--enable-tools` / `--disable-tools`. Trên `0.0.0.0`, `--enable-tools` sẽ hỏi xác nhận y/N; thêm `--yes`/`-y` để bỏ qua câu hỏi. Một request không thể tự vượt chính sách này bằng `enable_tools=true`.

Với `--secure`, server bind localhost nhưng vẫn truy cập được từ Internet. Docs không nói chính sách server-side tools áp dụng ra sao lúc đó: **cần kiểm tra lại**.

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

### Theo dõi request

Mọi request dùng API key đều hiện trong Studio: ở panel góc màn hình, và ở trang **Settings → API Monitor**. Mỗi request có:

- prompt và response;
- số token, time-to-first-token, throughput;
- lỗi, và **Context used**.

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth
