---
title: "Connections"
description: "Dùng model của OpenAI, Anthropic, OpenRouter hoặc llama.cpp, vLLM, Ollama trong giao diện Unsloth."
---

# Connections: nhà cung cấp API và model server

Connections cho bạn dùng model không chạy trong Unsloth, ngay trong giao diện chat của Unsloth. Bạn vẫn gắn được bộ công cụ của Unsloth cho các model này: web search, code execution, deep research. Có hai nhóm kết nối: nhà cung cấp cloud và model server bạn tự chạy.

| Kết nối | Loại | Cần gì | Base URL mẫu | Model xuất hiện ở |
| --- | --- | --- | --- | --- |
| OpenAI (cả ChatGPT/Codex subscription) | Cloud | API key từ OpenAI dashboard | (không cần) | **Connected** |
| Anthropic | Cloud | API key từ Anthropic Console | (không cần) | **Connected** |
| OpenRouter | Cloud, nhiều model qua 1 key | API key OpenRouter | (không cần) | **Connected** |
| llama.cpp (`llama-server`) | Model server | Key chỉ khi server chạy với `--api-key` | `http://localhost:8080/v1` | **Connected** / **External** (nguồn ghi không thống nhất) |
| vLLM | Model server, throughput cao | Key chỉ khi chạy với `--api-key` | `http://localhost:8000/v1` | **Connected** |
| Ollama | Model server đơn giản | Thường không cần key | `http://localhost:11434` hoặc `http://localhost:11434/v1` | **Connected** |

## Các bước với nhà cung cấp cloud

1. Tạo API key trên dashboard của nhà cung cấp.
2. Trong Unsloth: **Settings** → **Connections** → **Add Connection**.
3. Chọn nhà cung cấp, dán API key.
4. Bấm **Reload Models** để lấy danh sách model mà tài khoản được dùng. Chọn model muốn bật rồi lưu.
5. Chọn model trong dropdown **Select Model**, ở nhóm **Connected**.

Với OpenRouter, nếu **Load Models** không trả về model bạn cần, hãy nhập tay model ID, ví dụ:

```
openai/gpt-5.5 
anthropic/claude-sonnet-4.6 
google/gemini-3-pro
```

**Nguồn:** https://unsloth.ai/docs/integrations/connections, https://unsloth.ai/docs/integrations/connections/openai, https://unsloth.ai/docs/integrations/connections/anthropic-claude, https://unsloth.ai/docs/integrations/connections/openrouter, https://unsloth.ai/docs/integrations/connections/connect-llama.cpp-to-unsloth-run-ggufs-with-llama-server, https://unsloth.ai/docs/integrations/connections/vllm, https://unsloth.ai/docs/integrations/connections/ollama

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

Với vLLM, bật tùy chọn **Reasoning model** nếu model hỗ trợ thinking.

::: warning Docs chưa thống nhất
Hai trang ghi Base URL của Ollama khi thêm vào Connections khác nhau:

| Thông số | [integrations/connections](https://unsloth.ai/docs/integrations/connections) | [integrations/connections/ollama](https://unsloth.ai/docs/integrations/connections/ollama) |
| --- | --- | --- |
| Base URL | `http://localhost:11434/v1` | "In most local setups": `http://localhost:11434`; "If Unsloth asks for an OpenAI-compatible base URL": `http://localhost:11434/v1` |
| Model mẫu | `qwen3:14b` | `qwen3.6:35b-a3b`; model ID nhập tay ví dụ `qwen3.6` |
:::

**Nguồn:** https://unsloth.ai/docs/integrations/connections, https://unsloth.ai/docs/integrations/connections/openai, https://unsloth.ai/docs/integrations/connections/anthropic-claude, https://unsloth.ai/docs/integrations/connections/openrouter, https://unsloth.ai/docs/integrations/connections/connect-llama.cpp-to-unsloth-run-ggufs-with-llama-server, https://unsloth.ai/docs/integrations/connections/vllm, https://unsloth.ai/docs/integrations/connections/ollama

## Tính năng đi kèm

Khi dùng model qua Connections, bạn có thêm các tính năng sau:

- **Prompt caching:** tái dùng phần đầu giống nhau của những prompt dài, để giảm độ trễ và chi phí. Hỗ trợ OpenAI, Anthropic, llama.cpp. Chỉnh ở mục **Prompt caching** trong side panel. Với llama.cpp, caching bật mặc định. Muốn tắt, thêm `--no-cache-prompt` khi khởi động `llama-server`.
- **Code execution phía nhà cung cấp:**
  - Anthropic dùng Code execution tool của Claude.
  - OpenAI dùng container tái sử dụng. Bạn tạo, xóa, chọn container trong **Code Execution** settings. Ở thread mới, chọn lại cùng container để giữ file và trạng thái.
- **Web search & Thinking phía nhà cung cấp:** hỗ trợ model của OpenAI, Anthropic, OpenRouter, Mistral, Gemini, Kimi. Nút **Think** đổi theo model: có model chỉ bật tắt, có model cho chọn mức reasoning effort.
- **Image generation:** có nút "Edit Image" và nút tải ảnh ở độ phân giải gốc.

**Nguồn:** https://unsloth.ai/docs/integrations/connections, https://unsloth.ai/docs/integrations/connections/openai, https://unsloth.ai/docs/integrations/connections/anthropic-claude, https://unsloth.ai/docs/integrations/connections/openrouter, https://unsloth.ai/docs/integrations/connections/connect-llama.cpp-to-unsloth-run-ggufs-with-llama-server, https://unsloth.ai/docs/integrations/connections/vllm, https://unsloth.ai/docs/integrations/connections/ollama
