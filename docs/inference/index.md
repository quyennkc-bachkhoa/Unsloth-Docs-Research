---
title: Inference & API
description: Chạy model local trong Unsloth Studio, gọi qua API tương thích OpenAI/Anthropic, nối coding agent bằng unsloth start, Connections, MCP và tool calling.
---

# Inference & API

Phần Inference & API gồm trang này và sáu trang con. Nếu mới bắt đầu, bạn đọc theo thứ tự từ trên xuống:

- **Inference & API** (trang này): inference trong Unsloth là gì, ba cách dùng model.
  1. [Chạy model local trong Studio Chat](/inference/studio-chat): tải model GGUF về máy rồi chat ngay trong giao diện, không cần viết code. Trang này nói cách chọn mức quantization vừa với máy, các tính năng của Chat (chạy code, đọc web, so sánh 2 model) và khi nào cần chỉnh tham số sampling.
  1. [API tương thích OpenAI và Anthropic](/inference/api): biến model local thành API trên `localhost` để code của bạn gọi vào. Request viết theo đúng định dạng của OpenAI/Anthropic, nên chỉ cần đổi `base_url` và key. Trang này có cách lấy key, chọn endpoint, ví dụ curl và Python, và cách mở API cho máy khác trong mạng.
  1. [Kết nối coding agent](/inference/coding-agent): cho Claude Code, Codex, OpenCode dùng model local thay vì model cloud. Cách nhanh nhất là một lệnh `unsloth start <agent>`. Trang này cũng có cách cấu hình thủ công và các lỗi dễ gặp, như agent trả lời nhưng không sửa file.
  1. [Connections](/inference/connections): dùng model đang chạy ở llama.cpp, vLLM hoặc Ollama ngay trong giao diện chat của Unsloth, mà vẫn có web search và code execution. Model cloud cũng nối được, nhưng trang chỉ nói qua.
  1. [MCP](/inference/mcp): gắn thêm công cụ bên ngoài cho model, như tra docs, tìm model trên Hugging Face hoặc thao tác với dịch vụ như Vercel. Trang này có các bước thêm MCP server trong Studio và lưu ý bảo mật.
  1. [Tool calling](/inference/tool-calling): cơ chế để model "gọi hàm", tức model đề xuất hàm cần chạy và chương trình của bạn chạy hàm đó. Trang này nói ba cách dùng: trong Studio, qua API và tự viết vòng lặp bằng Python.

## Inference trong Unsloth là gì

Inference (suy luận) trong Unsloth là quá trình sử dụng mô hình ngôn ngữ lớn (LLM) đã được huấn luyện hoặc tinh chỉnh (fine-tune) để tạo ra câu trả lời, dự đoán hoặc phản hồi từ một câu lệnh (prompt) đầu vào.

1. **Chat trực tiếp** trong giao diện Unsloth Studio hoặc Unsloth Desktop. Mọi thứ chạy offline 100% trên máy bạn. Studio chạy được trên macOS, Windows, Linux, WSL, kể cả chỉ có CPU. Bạn **không bắt buộc có GPU**.
2. **Gọi qua API**: Unsloth mở model đang nạp thành một endpoint (địa chỉ HTTP nhận request) có xác thực. Endpoint này dùng chung định dạng với OpenAI và Anthropic. Nhờ vậy app Python của bạn, hoặc coding agent (tác tử lập trình như Claude Code, Codex), gọi vào máy bạn thay vì gọi lên cloud.
3. **Connections**: Unsloth làm giao diện chung cho model server bạn tự chạy (llama.cpp, vLLM, Ollama). Nó cũng nối được model cloud (OpenAI, Anthropic, OpenRouter), nhưng tài liệu này chỉ nói qua.

Về bên trong, Studio được xây trên llama.cpp và Hugging Face. Model nạp trong Unsloth, kể cả GGUF, được phục vụ qua `llama-server`.

::: tip Kiến thức nền
Chưa rõ sampling (temperature/top-p/top-k/min-p), chat template, KV cache hay tool calling là gì? Xem [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling).
:::

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
