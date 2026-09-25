---
title: Chạy model và gọi API
description: Chạy model local trong Unsloth Studio, gọi qua API tương thích OpenAI/Anthropic, nối coding agent bằng unsloth start, Connections, MCP và tool calling.
---

# Chạy model và gọi API

Phần này nói cách dùng model trong Unsloth: chat ngay trong Studio, gọi qua API trên máy bạn, hoặc nối với model server và công cụ bên ngoài.

::: tip Tóm tắt
- **Dùng khi:** bạn đã có (hoặc sắp tải) một model và muốn chat với nó, gọi nó từ code hay từ coding agent.
- **Kết quả:** biết ba cách dùng model trong Unsloth và nên đọc trang con nào cho việc của mình.
- **Nên biết trước:** [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) (sampling, chat template, KV cache, tool calling).
:::

## Các trang trong phần này

| Trang | Giúp bạn làm gì | Đọc khi nào |
| --- | --- | --- |
| [Chat trong Studio](/inference/studio-chat) | Tải model GGUF về máy rồi chat ngay trong giao diện, không cần viết code; chọn mức quantization vừa với máy, dùng các tính năng của Chat (chạy code, đọc web, so sánh 2 model), chỉnh tham số sampling | Mới bắt đầu, muốn thử model nhanh nhất |
| [Gọi qua API (OpenAI, Anthropic)](/inference/api) | Biến model local thành API trên `localhost` để code của bạn gọi vào; chỉ cần đổi `base_url` và key. Có cách lấy key, chọn endpoint, ví dụ curl và Python, mở API cho máy khác trong mạng | Muốn gọi model từ app hoặc script |
| [Dùng với coding agent](/inference/coding-agent) | Cho Claude Code, Codex, OpenCode dùng model local thay vì model cloud, nhanh nhất bằng một lệnh `unsloth start <agent>`; có cấu hình thủ công và các lỗi dễ gặp | Muốn coding agent chạy bằng model trên máy mình |
| [Kết nối server khác](/inference/connections) | Dùng model đang chạy ở llama.cpp, vLLM hoặc Ollama ngay trong giao diện chat của Unsloth, vẫn có web search và code execution | Đã có sẵn model server riêng |
| [Thêm công cụ qua MCP](/inference/mcp) | Gắn thêm công cụ bên ngoài cho model, như tra docs, tìm model trên Hugging Face, thao tác với Vercel; có các bước thêm MCP server và lưu ý bảo mật | Muốn model dùng được dịch vụ bên ngoài |
| [Cho model gọi hàm (tool calling)](/inference/tool-calling) | Hiểu cơ chế model đề xuất hàm và chương trình của bạn chạy hàm; ba cách dùng: trong Studio, qua API, tự viết vòng lặp Python | Muốn model gọi hàm do bạn viết |

## Inference trong Unsloth là gì

Inference (suy luận) trong Unsloth là quá trình sử dụng mô hình ngôn ngữ lớn (LLM) đã được huấn luyện hoặc tinh chỉnh (fine-tune) để tạo ra câu trả lời, dự đoán hoặc phản hồi từ một câu lệnh (prompt) đầu vào. Trong Unsloth có ba cách dùng model:

1. **Chat trực tiếp** trong giao diện Unsloth Studio hoặc Unsloth Desktop. Mọi thứ chạy offline 100% trên máy bạn. Studio chạy được trên macOS, Windows, Linux, WSL, kể cả chỉ có CPU. Bạn **không bắt buộc có GPU**.
2. **Gọi qua API**: Unsloth mở model đang nạp thành một endpoint (địa chỉ HTTP nhận request) có xác thực. Endpoint này dùng chung định dạng với OpenAI và Anthropic. Nhờ vậy app Python của bạn, hoặc coding agent (tác tử lập trình như Claude Code, Codex), gọi vào máy bạn thay vì gọi lên cloud.
3. **Connections**: Unsloth làm giao diện chung cho model server bạn tự chạy (llama.cpp, vLLM, Ollama). Nó cũng nối được model cloud (OpenAI, Anthropic, OpenRouter), nhưng tài liệu này chỉ nói qua.

Về bên trong, Studio được xây trên llama.cpp và Hugging Face. Model nạp trong Unsloth, kể cả GGUF, được phục vụ qua `llama-server`.

<div class="dg">
<div class="dg-stages">
<div class="dg-stage">
<div class="dg-node">App Python / curl</div>
<div class="dg-node">Coding agent<small>Claude Code, Codex, OpenCode</small></div>
</div>
<div class="dg-stage"><div class="dg-node is-main">API Unsloth trên localhost<small>vd <code>:8888</code></small><small><code>/v1/chat/completions</code><br><code>/v1/responses</code><br><code>/v1/messages</code></small></div></div>
<div class="dg-stage"><div class="dg-node">llama-server<small>trong Unsloth</small></div></div>
<div class="dg-stage"><div class="dg-node is-end">Model local (GGUF)<small>trên CPU/GPU</small></div></div>
<div class="dg-sub" style="grid-column: 2" data-e="server-side tools"><div class="dg-node is-ghost">Python, bash, web search</div></div>
</div>
</div>

**Nguồn:** https://unsloth.ai/docs/new/studio/chat, https://unsloth.ai/docs/basics/api

## Đọc tiếp

- [Chat trong Studio](/inference/studio-chat) — cách nhanh nhất để tải một model và bắt đầu chat, không cần viết code.
- [Gọi qua API (OpenAI, Anthropic)](/inference/api) — khi bạn muốn gọi model local từ code của mình.
- [Dùng với coding agent](/inference/coding-agent) — nối Claude Code, Codex, OpenCode vào model trên máy.
