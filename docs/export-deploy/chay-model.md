---
title: Chạy model đã xuất
description: "Đưa model đã export lên engine để chạy: Ollama, llama-server, vLLM, LM Studio."
---

# Chạy model đã xuất

Sau khi export, bạn cần một engine (phần mềm chạy model) để dùng model. Trang này gom cách chạy trên bốn engine mà docs Unsloth hướng dẫn: Ollama, llama-server, vLLM và LM Studio.

::: tip Tóm tắt
- **Dùng khi:** bạn đã có file export (GGUF hoặc merged 16-bit) và muốn chạy, chat hoặc gọi model qua API.
- **Kết quả:** chọn được engine hợp với định dạng mình có, và có lệnh cài, nạp model, mở server cho từng engine.
- **Nên biết trước:** [Export và deploy](/export-deploy/) (bảng định dạng và engine), [Xuất file GGUF](/export-deploy/gguf).
:::

## Chọn engine nào

Engine phải khớp với định dạng bạn đã xuất: ba engine đọc GGUF, còn vLLM chạy model safetensors.

| Engine | Định dạng cần | Hợp khi |
| --- | --- | --- |
| [Ollama](#ollama) | GGUF | Muốn Unsloth tự tạo `Modelfile` kèm chat template đã dùng lúc fine-tune |
| [llama-server](#llama-server) | GGUF | Cần endpoint tương thích OpenAI để gọi bằng thư viện `openai` |
| [vLLM](#vllm) | Merged 16-bit, hoặc LoRA adapter | Muốn serve thư mục model merged 16-bit (vLLM cũng chạy được LoRA adapter, docs có trang LoRA Hot Swapping) |
| [LM Studio](#lm-studio) | GGUF | Muốn dùng ứng dụng để chat hoặc bật API tương thích OpenAI |

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama, https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint, https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide, https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio

## Ollama {#ollama}

Unsloth đưa model sang Ollama qua đường GGUF. Điểm tiện là Unsloth **tự tạo `Modelfile`**. Đây là file cấu hình Ollama cần, trong đó có chat template bạn đã dùng lúc fine-tune.

Quy trình theo docs:

1. Cài Ollama (trong notebook Colab).
2. Export model sang GGUF. Chỉ bật `True` cho **một** dòng quant, thường là dòng đầu `Q8_0`.
3. Chạy Ollama ở chế độ nền. Nếu không dùng Colab, bạn chỉ cần chạy `ollama serve` trong terminal.
4. Dùng `Modelfile` do Unsloth sinh ra để tạo model Ollama, rồi gọi inference.

::: info Code của trang Ollama bị thiếu
Đã đối chiếu cả bản gốc chưa lọc của trang. Các bước cài Ollama, export GGUF, in `Modelfile`, tạo model và gọi inference đều chỉ có dạng ảnh, không có khối code văn bản. Lệnh duy nhất ở dạng văn bản là `ollama serve`. Chi tiết xem tutorial "Finetune Llama-3 and Use In Ollama" — cần kiểm tra lại.
:::

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama

## llama-server (endpoint OpenAI) {#llama-server}

`llama-server` là server của llama.cpp. Nó phục vụ file GGUF qua một endpoint tương thích OpenAI, nên bạn gọi model bằng thư viện `openai` như gọi API thông thường.

Đầu tiên, build llama.cpp như ở trang [Xuất file GGUF](/export-deploy/gguf#luu-gguf-thu-cong). Nếu chỉ chạy CPU hoặc Mac/Metal, đổi `-DGGML_CUDA=ON` thành `-DGGML_CUDA=OFF`. Ví dụ trong docs dùng Devstral 2. Bước tải model:

```python
# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id = "unsloth/Devstral-2-123B-Instruct-2512-GGUF",
    local_dir = "Devstral-2-123B-Instruct-2512-GGUF",
    allow_patterns = ["*UD-Q2_K_XL*", "*mmproj-F16*"],
)
```

Chạy server:

```bash
./llama.cpp/llama-server \
    --model Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf \
    --mmproj Devstral-Small-2-24B-Instruct-2512-GGUF/mmproj-F16.gguf \
    --alias "unsloth/Devstral-Small-2-24B-Instruct-2512" \
    --threads -1 \
    --n-gpu-layers 999 \
    --prio 3 \
    --min-p 0.01 \
    --ctx-size 16384 \
    --port 8001 \
    --jinja
```

::: warning Docs chưa thống nhất
Trên cùng trang [llama-server & OpenAI endpoint](https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint), bước tải và bước serve dùng **hai model khác nhau**. Vì vậy nếu chép y nguyên, server sẽ không tìm thấy file `--model`:

| Thông số | Bước tải `snapshot_download` ([nguồn](https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint)) | Bước `llama-server` ([nguồn](https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint)) |
| --- | --- | --- |
| Model | `unsloth/Devstral-2-123B-Instruct-2512-GGUF` | `Devstral-Small-2-24B-Instruct-2512` |
| Thư mục | `Devstral-2-123B-Instruct-2512-GGUF` | `Devstral-Small-2-24B-Instruct-2512-GGUF` |
| Quant | `*UD-Q2_K_XL*` | `UD-Q4_K_XL.gguf` |

Khi làm theo, hãy cho `--model`/`--mmproj` trỏ đúng file bạn đã thực sự tải về.
:::

Gọi server từ Python (sau khi chạy `pip install openai`):

```python
from openai import OpenAI
import json
openai_client = OpenAI(
    base_url = "http://127.0.0.1:8001/v1",
    api_key = "sk-no-key-required",
)
completion = openai_client.chat.completions.create(
    model = "unsloth/Devstral-Small-2-24B-Instruct-2512",
    messages = [{"role": "user", "content": "What is 2+2?"},],
)
print(completion.choices[0].message.content)
```

::: warning Lỗi thường gặp
`--jinja` chèn thêm system message: khi bật `--jinja` và model hỗ trợ tools, llama-server tự nối thêm câu "Respond in JSON format, either with tool_call … or with response …" vào system message. Câu chèn thêm này có thể làm hỏng model fine-tune. Nếu tắt bằng `--no-jinja`, bạn mất hỗ trợ `tools`. Docs khuyên thêm một prompt tool calling riêng cho mọi fine-tune.
:::

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint

## vLLM {#vllm}

vLLM chạy model ở dạng merged 16-bit. Quy trình gồm ba bước: cài vLLM, lưu model dạng merged 16-bit, rồi serve thư mục model đó.

Cài đặt (GPU NVIDIA):

```bash
pip install --upgrade pip
pip install uv
uv pip install -U vllm --torch-backend=auto
```

Với GPU AMD, dùng Docker image nightly `rocm/vllm-dev:nightly`.

Lưu model cho vLLM (merged 16-bit):

```python
model.save_pretrained_merged("finetuned_model", tokenizer, save_method = "merged_16bit")
## OR to upload to HuggingFace:
model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")
```

Serve model ở một terminal khác:

```bash
vllm serve finetuned_model
```

Nếu lệnh trên không chạy, dùng đường dẫn đầy đủ:

```bash
vllm serve /mnt/disks/daniel/finetuned_model
```

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide

## LM Studio {#lm-studio}

LM Studio là ứng dụng chạy file GGUF. Quy trình có ba bước: export GGUF từ Unsloth, import vào LM Studio, rồi load model để chat hoặc bật API tương thích OpenAI.

Import file GGUF bằng CLI `lms`:

```bash
lms import /path/to/model.gguf
```

Nếu bạn đã `push_to_hub_gguf`, có thể tải thẳng từ Hugging Face:

```bash
# Download from HF by repo name
lms get hf_username/my_model_gguf

# Pick a quantization with @
lms get hf_username/my_model_gguf@Q4_K_M
```

Load model và bật server. Địa chỉ mặc định thường là `http://localhost:1234/v1`:

```bash
lms load <model-identifier> --gpu=auto --context-length=8192
```

```bash
lms server start --port 1234
```

```bash
curl http://localhost:1234/v1/models
```

Mẹo debug template: chạy `lms log stream` để xem prompt thô mà LM Studio gửi vào model.

**Nguồn:** https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio

## Đọc tiếp

- [Mở cho máy khác (LAN, Cloudflare)](/export-deploy/lan-remote) — trang kế tiếp: cho máy khác trong mạng nội bộ hoặc ngoài Internet gọi được server vừa mở.
- [Gọi qua API (OpenAI, Anthropic)](/inference/api) — gọi model qua endpoint tương thích OpenAI từ code.
- [Lỗi thường gặp](/export-deploy/loi-thuong-gap) — khi model chạy trên engine cho kết quả khác lúc train.
