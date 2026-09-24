---
title: "Kết nối coding agent"
description: "Dùng unsloth start để nối Claude Code, Codex, OpenCode với model local, và cách cấu hình thủ công."
---

# Kết nối coding agent với `unsloth start`

`unsloth start <agent>` cho coding agent (Claude Code, Codex, OpenCode…) dùng model đang chạy trong Unsloth, chỉ bằng một lệnh. Lệnh này tự cấu hình endpoint, API key, provider, model và context length cho từng lần chạy agent. Cấu hình chỉ tồn tại trong phiên đó. Nó **không** ghi provider Unsloth vào file cấu hình thường ngày của agent. Toàn bộ có thể chạy offline.

Cách dùng nhanh: mở Unsloth, nạp model, `cd` vào thư mục project, rồi chạy:

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

Bạn cũng nạp được model ngay khi khởi động agent. Hậu tố `:UD-Q4_K_XL` chọn bản quant GGUF. Cờ `--model` hoạt động như sau:

- Nếu Unsloth chưa chạy, nó khởi động một server tạm và tắt server đó khi agent thoát.
- Nếu Unsloth đang chạy, nó chỉ kết nối vào.

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
Lệnh nạp model kèm agent hoặc server được viết khác nhau giữa các trang. Chỗ khác là tên model, tên agent và dấu `\` xuống dòng. Lệnh ở trên lấy từ [integrations/unsloth-start](https://unsloth.ai/docs/integrations/unsloth-start). Các biến thể khác trong docs, chép nguyên văn:

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

Tham số nào không thuộc Unsloth sẽ được chuyển nguyên cho agent:

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
Trong [basics/api](https://unsloth.ai/docs/basics/api) (mục "Control reasoning behavior"), ví dụ về cờ reasoning có comment nói một đằng, giá trị cờ làm một nẻo. Chép nguyên văn:

```bash
# Disable reasoning / thinking output
unsloth run \
  --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \
  --reasoning on  \
  --reasoning-effort medium
```

Trong khi đó [integrations/connect-python-sdk-to-unsloth](https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth) và [basics/claude-code](https://unsloth.ai/docs/basics/claude-code) ghi: "Use `--reasoning off` to turn thinking off, or `--reasoning on` to turn it on". [integrations/unsloth-start](https://unsloth.ai/docs/integrations/unsloth-start) ghi `--reasoning` nhận `on`, `off` hoặc `auto`.
:::

Để kết nối tới một Unsloth server ở máy khác, đặt hai biến môi trường trước khi chạy:

```bash
export UNSLOTH_STUDIO_URL=https://studio.example.com
export UNSLOTH_API_KEY=sk-unsloth-...
unsloth start claude
```

## Kết nối thủ công (khi không dùng `unsloth start`)

Nếu bạn muốn tự cấu hình, mỗi agent có cách riêng.

**Claude Code** đọc cấu hình từ biến môi trường. Các biến này chỉ có hiệu lực trong phiên shell hiện tại:

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

Có một điểm cần tắt. Claude Code gắn một attribution header vào đầu system prompt, và giá trị header đổi theo mỗi request. Vì phần đầu prompt luôn khác đi, KV cache mất hiệu lực. Docs Unsloth ghi inference với model local khi đó **chậm hơn 90%**. Cách tắt ngay khi chạy:

```bash
claude --settings '{"env":{"CLAUDE_CODE_ATTRIBUTION_HEADER":"0","CLAUDE_CODE_ENABLE_TELEMETRY":"0"}}' --model unsloth/gemma-4-26B-A4B-it-GGUF
```

Tùy chọn thêm: thu gọn system prompt để tái dùng KV cache tốt hơn:

```shellscript
claude --model unsloth/gemma-4-26B-A4B-it-GGUF --bare --exclude-dynamic-system-prompt-sections
```

**Codex** chỉ dùng OpenAI Responses API. Codex đã bỏ hỗ trợ Chat Completions. Cấu hình nằm trong `~/.codex/config.toml`, trên Windows là `%USERPROFILE%\.codex\config.toml`:

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

Hai lưu ý:

- `env_key` là **tên** của biến môi trường chứa key, không phải bản thân key.
- Đừng chạy `codex` trần trước khi cấu hình. Lệnh đó mở màn hình "Sign in with ChatGPT" và bạn không thoát ra được.

::: warning Docs chưa thống nhất
Trang [basics/codex](https://unsloth.ai/docs/basics/codex) mô tả vị trí profile và cách chọn profile theo nhiều cách khác nhau:

| Thông số | Cách ghi 1 | Cách ghi 2 |
| --- | --- | --- |
| Nơi đặt profile `unsloth_api` | Khối `model_provider = "unsloth_api"` / `model = ...` "create a Codex profile", không có tiêu đề bảng TOML; mục "Disconnect" nhắc khối `[profiles.unsloth_api]` trong `~/.codex/config.toml` | "The `--profile unsloth_api` flag tells Codex to load `~/.codex/unsloth_api.config.toml`" |
| Lệnh chạy | `codex --oss --profile unsloth_api` | Mục Disconnect: "Launch Codex without `-p unsloth_api`" |
| Lệnh chạy với llama.cpp | `codex --oss llama_cpp` (mục llama.cpp) | "use the same shape with the `llama_cpp` profile"; mục Setup: `codex --oss --profile llama_cpp` |
| `requires_openai_auth` trong provider llama.cpp | Provider `unsloth_api` có `requires_openai_auth = false` | Provider `llama_cpp` không ghi dòng này (docs nói mặc định đã là `false`) |
| Port | Unsloth Studio: `http://localhost:8888/v1` | llama-server: `http://localhost:8001/v1` |

Ngoài ra `unsloth start codex` ([integrations/unsloth-start](https://unsloth.ai/docs/integrations/unsloth-start)) tạo Codex home riêng, không dùng `~/.codex`.
:::

**OpenCode Desktop:** vào `/model` → **Connect provider** → **Custom**, rồi điền:

- Provider ID: `unsloth-studio`.
- Base URL: `http://localhost:8888/v1/`. Giữ nguyên đuôi `/v1/`.
- API key: `sk-unsloth-…`.
- Model ID: đúng như Unsloth phục vụ, ví dụ `unsloth/Qwen3.6-27B-GGUF`.

Khởi động lại opencode thì provider mới xuất hiện.

**Khi tự chạy `unsloth run` cho agent**, nhớ thêm `--disable-tools`:

```bash
# Serve for a coding agent: --disable-tools passes the agent's own tools through
unsloth run \
  --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  --disable-tools \
  --reasoning off \
  -p 8888
```

Lý do: mặc định Unsloth chạy server-side tools của chính nó và "nuốt" tool call của agent. Kết quả là agent vẫn trả lời nhưng không bao giờ sửa file. `--disable-tools` chuyển sang chế độ passthrough (chuyển tiếp), để agent dùng được tool Write/Edit/Bash của nó.

**Nguồn:** https://unsloth.ai/docs/integrations/unsloth-start, https://unsloth.ai/docs/basics/claude-code, https://unsloth.ai/docs/basics/codex, https://unsloth.ai/docs/integrations/opencode
