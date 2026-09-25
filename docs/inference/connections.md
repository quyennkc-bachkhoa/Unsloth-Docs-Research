---
title: Kết nối server khác
description: "Nối model server local (llama.cpp, vLLM, Ollama) vào giao diện Unsloth; nhà cung cấp cloud chỉ nói qua."
---

# Kết nối server khác

Connections cho bạn dùng model không nạp trực tiếp trong Unsloth, ngay trong giao diện chat của Unsloth. Trọng tâm của trang này là **model server bạn tự chạy trên máy**: llama.cpp, vLLM, Ollama.

::: tip Tóm tắt
- **Dùng khi:** bạn đã chạy model bằng llama.cpp, vLLM hoặc Ollama và muốn chat với nó trong giao diện Unsloth.
- **Kết quả:** thêm được kết nối trong **Settings → Connections** và dùng model đó kèm web search, code execution, deep research của Unsloth.
- **Nên biết trước:** cách dùng trang Chat ở [Chat trong Studio](/inference/studio-chat).
:::

## Chọn model server

Bạn vẫn gắn được bộ công cụ của Unsloth (web search, code execution, deep research) cho model ở cả ba loại server dưới đây; chỗ khác nhau là đặc điểm, key và Base URL.

| Model server | Đặc điểm | Cần key? | Base URL mẫu | Model xuất hiện ở |
| --- | --- | --- | --- | --- |
| llama.cpp (`llama-server`) | Chạy GGUF, nhẹ | Chỉ khi server chạy với `--api-key` | `http://localhost:8080/v1` | **Connected** / **External** (nguồn ghi không thống nhất) |
| vLLM | Throughput cao, cần GPU | Chỉ khi chạy với `--api-key` | `http://localhost:8000/v1` | **Connected** |
| Ollama | Đơn giản, dễ cài | Thường không cần | `http://localhost:11434` hoặc `http://localhost:11434/v1` | **Connected** |

**Nguồn:** https://unsloth.ai/docs/integrations/connections, https://unsloth.ai/docs/integrations/connections/connect-llama.cpp-to-unsloth-run-ggufs-with-llama-server, https://unsloth.ai/docs/integrations/connections/vllm, https://unsloth.ai/docs/integrations/connections/ollama

## Các bước với model server

Bạn cần khởi động server trước, rồi mới thêm kết nối trong Unsloth.

llama.cpp:

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
Lệnh vLLM ở trên lấy từ mục "Common vLLM arguments" trong [integrations/connections/vllm](https://unsloth.ai/docs/integrations/connections/vllm). Các biến thể lệnh tối thiểu trong docs, chép nguyên văn:

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

Khi server đã chạy, thêm kết nối trong Unsloth:

1. Vào **Settings → Connections**, thêm kết nối mới.
2. Chọn llama.cpp, vLLM hoặc Ollama.
3. Dán **Base URL**.
4. Bấm **Load Models**. Nếu server không có `/models`, nhập model ID bằng tay.
5. Lưu.

<figure>

![Form thêm kết nối llama.cpp](/images/inference/connection-llamacpp.webp)

<figcaption>Form thêm kết nối: chọn llama.cpp, đặt tên và dán Base URL <code>http://localhost:8080/v1</code>. Ảnh: <a href="https://unsloth.ai/docs/integrations/connections/connect-llama.cpp-to-unsloth-run-ggufs-with-llama-server">docs Unsloth</a>.</figcaption>
</figure>

<figure>

![Nạp danh sách model hoặc nhập tay](/images/inference/connection-load-models.webp)

<figcaption>Bấm <b>Load available models</b>, hoặc nhập model ID bằng tay nếu server không trả danh sách. Ảnh: <a href="https://unsloth.ai/docs/integrations/connections/connect-llama.cpp-to-unsloth-run-ggufs-with-llama-server">docs Unsloth</a>.</figcaption>
</figure>

Với vLLM, bật tùy chọn **Reasoning model** nếu model hỗ trợ thinking.

::: warning Docs chưa thống nhất
Hai trang ghi Base URL của Ollama khi thêm vào Connections khác nhau:

| Thông số | [integrations/connections](https://unsloth.ai/docs/integrations/connections) | [integrations/connections/ollama](https://unsloth.ai/docs/integrations/connections/ollama) |
| --- | --- | --- |
| Base URL | `http://localhost:11434/v1` | "In most local setups": `http://localhost:11434`; "If Unsloth asks for an OpenAI-compatible base URL": `http://localhost:11434/v1` |
| Model mẫu | `qwen3:14b` | `qwen3.6:35b-a3b`; model ID nhập tay ví dụ `qwen3.6` |
:::

**Nguồn:** https://unsloth.ai/docs/integrations/connections, https://unsloth.ai/docs/integrations/connections/connect-llama.cpp-to-unsloth-run-ggufs-with-llama-server, https://unsloth.ai/docs/integrations/connections/vllm, https://unsloth.ai/docs/integrations/connections/ollama

## Tính năng đi kèm

- **Prompt caching:** tái dùng phần đầu giống nhau của những prompt dài để giảm độ trễ. Với llama.cpp, caching bật mặc định. Muốn tắt, thêm `--no-cache-prompt` khi khởi động `llama-server`. Chỉnh ở mục **Prompt caching** trong side panel.

<figure>

![Sơ đồ prompt caching](/images/inference/prompt-caching.webp)

<figcaption>Prompt caching: phần đầu prompt giống lần trước thì dùng lại (cache hit); khác từ token đầu thì phải tính lại (cache miss). Ảnh: <a href="https://unsloth.ai/docs/integrations/connections/connect-llama.cpp-to-unsloth-run-ggufs-with-llama-server">docs Unsloth</a>.</figcaption>
</figure>

**Nguồn:** https://unsloth.ai/docs/integrations/connections

## Nhà cung cấp cloud (nói qua)

Connections cũng nhận model cloud của **OpenAI, Anthropic, OpenRouter**: tạo API key ở dashboard của hãng, vào **Settings → Connections → Add Connection**, dán key, bấm **Reload Models** rồi chọn model ở nhóm **Connected**. Với các model này, Unsloth dùng code execution, web search và thinking của chính nhà cung cấp. Phần này nằm ngoài trọng tâm local, chi tiết xem nguồn.

**Nguồn:** https://unsloth.ai/docs/integrations/connections/openai, https://unsloth.ai/docs/integrations/connections/anthropic-claude, https://unsloth.ai/docs/integrations/connections/openrouter

## Đọc tiếp

- [Thêm công cụ qua MCP](/inference/mcp) — gắn thêm công cụ bên ngoài cho model, như tra docs hay thao tác với dịch vụ khác.
- [Chat trong Studio](/inference/studio-chat) — các tính năng của trang Chat mà model kết nối cũng dùng được.
- [Chạy model đã xuất](/export-deploy/chay-model) — cách chạy model của bạn bằng llama.cpp, vLLM hay Ollama trước khi nối vào Unsloth.
