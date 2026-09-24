---
title: "Cạm bẫy khi dùng API và agent"
description: "Lỗi hay gặp khi dùng API và coding agent, dò theo triệu chứng."
---

# Cạm bẫy thường gặp

Khi API hoặc agent không chạy như ý, hãy dò theo triệu chứng trong danh sách sau. Mỗi dòng nêu triệu chứng trước, nguyên nhân và cách sửa sau.

::: warning Lỗi hay gặp khi dùng API và agent
**Kết nối và xác thực**

- **`401 Unauthorized`**: thiếu header hoặc key sai. Header phải là `Authorization: Bearer sk-unsloth-…`. Mất key thì tạo key mới ở **Settings → API**, vì key cũ không xem lại được.
- **`404 Not Found` với SDK**: OpenAI SDK cần `base_url` **có** đuôi `/v1`. Anthropic SDK cần `base_url` **không có** `/v1`, vì SDK tự thêm `/v1/messages`.
- **Sai model ID**: gọi `GET /v1/models` và chép đúng trường `id`.
- **`Lost connection to the model server`**: llama.cpp phía sau bị crash, hoặc tab model đã đóng. Nạp lại model từ **New Chat**.
- **Lộ API key**: ai có key và vào được mạng tới Unsloth đều gửi được request tới model. Output của `unsloth start ... --no-launch` cũng chứa thông tin đăng nhập.

**Request và response**

- **Quên `max_tokens` ở `/v1/messages`**: endpoint Anthropic bắt buộc có trường này.
- **Streaming ra một cục JSON, hoặc curl im lặng**: dùng `curl -N`. Trong script, dùng `print(..., flush=True)`. Nếu đi qua proxy thì tắt buffering. Qua Cloudflare tunnel (`--secure`) thì phải đặt `stream: false`.
- **Câu trả lời bị cắt ngắn**: xem **Context used** trong API monitor. Gần 100% hoặc stop reason `length` nghĩa là context đã đầy.

**Tools**

- **Agent trả lời nhưng không sửa file**: server-side tools của Unsloth đang nuốt tool call của agent. Chạy `unsloth run` với `--disable-tools`.
- **Server-side tools không chạy**: phải có `enable_tools: true` **và** liệt kê tool trong `enabled_tools`. Nếu server bind `0.0.0.0` thì tools mặc định tắt.

**Coding agent**

- **Claude Code chậm hơn 90% với model local**: attribution header làm KV cache mất hiệu lực. Cần đặt `CLAUDE_CODE_ATTRIBUTION_HEADER` = `0`.
- **Claude Code vẫn dùng model Anthropic mặc định**: kiểm tra `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` đã được export trong **cùng** shell chạy `claude`. Nếu gặp `Unable to connect to API (ConnectionRefused)` sau khi thôi dùng Unsloth, chạy `unset ANTHROPIC_BASE_URL`.
- **Codex không kết nối**: model phải là GGUF chạy qua `llama-server`. `wire_api` phải là `"responses"`, vì `"chat"` đã bị bỏ.
:::

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/integrations/connect-python-sdk-to-unsloth, https://unsloth.ai/docs/integrations/connect-curl-and-http-to-unsloth, https://unsloth.ai/docs/basics/claude-code, https://unsloth.ai/docs/basics/codex, https://unsloth.ai/docs/integrations/unsloth-start
