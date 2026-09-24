---
title: Lộ trình học
description: Lộ trình 5 bước từ đọc kiến thức nền, cài Unsloth, chạy model và API đến fine-tune và export.
---

# Lộ trình học

## Toàn cảnh lộ trình

Lộ trình có 5 bước, cộng một bước mở rộng về RL (Reinforcement Learning, học tăng cường). Mỗi bước ghi rõ trang [Kiến thức nền LLM](/kien-thuc-nen/) cần đọc **trước** khi bắt tay làm:

| Trước khi... | Đọc nhóm Kiến thức nền | Các trang |
| --- | --- | --- |
| Cài đặt (bước 2) | **Nhóm A** Cấu tạo mô hình | [Token & context](/kien-thuc-nen/token-va-context), [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh), [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer), [Dense & MoE](/kien-thuc-nen/dense-va-moe) |
| Cài đặt (bước 2) | **Nhóm B** Số học & bộ nhớ | [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho), [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa) |
| Chạy inference (bước 3) | **Nhóm D** Suy luận | [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) |
| Fine-tune (bước 4) | **Nhóm C** Huấn luyện | [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen), [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora), [RL & preference](/kien-thuc-nen/rl-va-preference) |

```mermaid
flowchart TD
  A["Nhóm A: Cấu tạo mô hình"] --> S1
  B["Nhóm B: Số học & bộ nhớ"] --> S1
  S1["Bước 1: Đọc nền"] --> S2["Bước 2: Cài đặt Unsloth"]
  D["Nhóm D: Suy luận & sampling"] --> S3
  S2 --> S3["Bước 3: Chạy model & API"]
  C["Nhóm C: Huấn luyện"] --> S4
  S3 --> S4["Bước 4: Dữ liệu + fine-tune"]
  S4 --> S5["Bước 5: Export & deploy"]
  S4 -. "mở rộng" .-> RL["RL: GRPO, DPO"]
  S5 -. "nạp lại model để chat" .-> S3
```

Workflow Studio trong docs cũng đi đúng thứ tự này: khởi chạy Unsloth → nạp model → nhập dữ liệu → làm sạch dữ liệu bằng Data Recipes → train → chat với model đã train để so với model gốc → lưu/export.

**Nguồn:** https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners

## Bước 1: Đọc kiến thức nền

**Mục tiêu:** hiểu đủ để chọn đúng model và đúng mức quantization (lượng tử hóa) cho máy của bạn, trước khi tải bất cứ thứ gì.

**Việc làm:**

1. Đọc **Nhóm A** (Cấu tạo mô hình): [Token & context](/kien-thuc-nen/token-va-context) → [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) → [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) → [Dense & MoE](/kien-thuc-nen/dense-va-moe).
2. Đọc **Nhóm B** (Số học & bộ nhớ): [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) → [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).
3. Đọc [Tổng quan kiến trúc](/tong-quan) để chọn Desktop, Studio hay Core.

**Trang nội bộ:** [Kiến thức nền LLM](/kien-thuc-nen/), [Tổng quan kiến trúc](/tong-quan), [Thuật ngữ](/thuat-ngu).

**Docs Unsloth gốc:** [Fine-tuning for Beginners](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners), [Unsloth Requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements).

**Dấu hiệu hoàn thành:** đọc được tên model mẫu trong docs như `unsloth/gemma-4-26B-A4B-it-GGUF` biến thể `UD-Q4_K_XL` và tự giải thích từng phần ("26B", "A4B", "it", "GGUF", "Q4"); biết máy mình có bao nhiêu VRAM (bộ nhớ card đồ họa) và RAM.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners, https://unsloth.ai/docs/basics/api

## Bước 2: Cài đặt Unsloth

**Mục tiêu:** có một bản Unsloth chạy được trên máy (hoặc trên Colab).

**Việc làm:** chọn một trong ba cách docs đưa ra.

- **Desktop (dễ nhất):** tải app tại https://unsloth.ai/download, cài, mở app.
- **Studio thủ công:** chạy script cài rồi khởi động Studio.

  macOS, Linux, WSL:

  ```bash
  curl -fsSL https://unsloth.ai/install.sh | sh
  ```

  Windows PowerShell:

  ```bash
  irm https://unsloth.ai/install.ps1 | iex
  ```

  Khởi động:

  ```bash
  unsloth studio -H 0.0.0.0 -p 8888
  ```

  Sau đó mở `http://127.0.0.1:8888` và tạo mật khẩu. Chạy lại lệnh cài để cập nhật.
- **Core (code):** cài gói Python theo hướng dẫn uv/pip của docs.
- **Không có GPU:** dùng notebook Colab miễn phí của Studio (GPU T4, docs ghi chạy/train được hầu hết model tới 22B tham số).

::: warning Mở Studio ra mạng
Lệnh `-H 0.0.0.0` cho phép thiết bị khác trong mạng truy cập. README cảnh báo server-side tools (chạy code, web search) có thể đang bật; giữ mật khẩu an toàn hoặc dùng `--disable-tools` khi phơi Unsloth ra ngoài. Chỉ dùng trên máy cá nhân thì có thể bind `-H 127.0.0.1`.
:::

**Trang nội bộ:** [Cài đặt & phần cứng](/cai-dat).

**Docs Unsloth gốc:** [Unsloth Installation](https://unsloth.ai/docs/get-started/install), [Studio Install](https://unsloth.ai/docs/new/studio/install), [uv, pip install & venv](https://unsloth.ai/docs/get-started/install/pip-install).

**Dấu hiệu hoàn thành:** app Desktop mở được, hoặc trình duyệt vào được Studio sau khi đăng nhập, hoặc `from unsloth import FastLanguageModel` chạy không lỗi.

**Nguồn:** https://unsloth.ai/docs/get-started/install, https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://github.com/unslothai/unsloth

## Bước 3: Chạy model và API

**Đọc trước:** **Nhóm D** [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling) (KV cache, temperature, top-p, top-k, min-p, chat template, tool calling).

**Mục tiêu:** chat được với một model local và gọi được nó từ code qua API.

**Việc làm:**

1. Mở dropdown **Select model**, tìm model, chọn biến thể quantization vừa RAM/VRAM. Docs dùng ví dụ `unsloth/gemma-4-26B-A4B-it-GGUF` với biến thể khuyến nghị `UD-Q4_K_XL`.
2. Gửi một tin nhắn thử; thử đổi temperature, top-p, top-k, system prompt.
3. Tạo API key: avatar Unsloth → **Settings** → **API** → đặt tên → **Create**, chép key `sk-unsloth-…` (chỉ hiện một lần).
4. Hoặc nạp model từ terminal; lệnh in ra URL endpoint và API key:

   ```bash
   unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
   ```

5. Gọi `GET /v1/models` để lấy ID model, rồi trỏ OpenAI SDK (`/v1/chat/completions`) hoặc Anthropic SDK (`/v1/messages`) vào base URL của Unsloth.

   ```bash
   curl http://localhost:8888/v1/models \
     -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx"
   ```

6. (Tùy chọn) Nối agent: `unsloth start claude`.

**Trang nội bộ:** [Inference & API](/inference), [Model catalog](/model-catalog).

**Docs Unsloth gốc:** [Unsloth API](https://unsloth.ai/docs/basics/api), [Studio Chat](https://unsloth.ai/docs/new/studio/chat), [Unsloth Start](https://unsloth.ai/docs/integrations/unsloth-start).

**Dấu hiệu hoàn thành:** Studio báo model đã nạp xong và trả lời tin nhắn thử; `GET /v1/models` trả về JSON dạng `{"data": [{"id": "gemma-4-26B-A4B-it-GGUF", ...}]}`; request từ code hiện trong API monitor. Nếu gặp `401 Unauthorized` là thiếu header hoặc sai key.

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs

## Bước 4: Chuẩn bị dữ liệu và fine-tune

**Đọc trước:** **Nhóm C** [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen) (loss, learning rate, epoch, batch, gradient accumulation, overfitting), [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora) (rank, alpha, target modules), [RL & preference](/kien-thuc-nen/rl-va-preference) (SFT khác RL thế nào).

**Mục tiêu:** hoàn thành một lượt fine-tune đầu tiên và chat được với model vừa train.

**Việc làm (theo Studio):**

1. **Model & phương pháp:** chọn loại (Text, Vision, Audio, Embeddings) và phương pháp QLoRA / LoRA / Full Fine-tuning. Lần đầu nên chọn QLoRA vì tốn VRAM ít nhất. Không chọn model GGUF (GGUF chỉ dùng cho inference). Model gated như Llama, Gemma cần Hugging Face token.
2. **Dữ liệu:** lấy từ Hugging Face Hub hoặc tải file local (`PDF`, `DOCX`, `JSONL`, `JSON`, `CSV`, `Parquet`). Chọn format `auto`, `alpaca`, `chatml` hoặc `sharegpt`. Đặt eval split để có biểu đồ Eval Loss. Nếu chỉ có tài liệu thô, dùng Data Recipes để sinh dataset.
3. **Hyperparameter:** giữ mặc định cho lần đầu (learning rate `2e-4`, context length `2048`, rank `16`, alpha `32`, epochs `3`, batch size `4`, gradient accumulation `8`). Dùng "Dataset slice" để thử nhanh trên vài dòng.
4. **Train:** bấm **Start Training**, theo dõi loss, learning rate, gradient norm và GPU monitor. Lưu config ra YAML để chạy lại.

**[Nhận định]** Chạy một lượt nhỏ bằng Dataset slice trước khi train cả dataset giúp phát hiện lỗi định dạng dữ liệu sớm.

**Trang nội bộ:** [Dữ liệu](/du-lieu), [Fine-tuning](/fine-tuning).

**Docs Unsloth gốc:** [Get started with Unsloth Studio](https://unsloth.ai/docs/new/studio/start), [Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide), [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [Data Recipes](https://unsloth.ai/docs/new/studio/data-recipe), [What Model Should I Use?](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use).

**Dấu hiệu hoàn thành:** thanh tiến độ đạt 100% steps; đường training loss có xu hướng giảm; model đã train xuất hiện ở tab **Fine-tuned** trong Select model và chat được để so với model gốc.

**Nguồn:** https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners

## Bước 5: Export và deploy

**Mục tiêu:** đưa model đã train ra khỏi Unsloth để dùng ở công cụ khác hoặc phục vụ cho thiết bị khác.

**Việc làm:**

1. Mở mục **Export** trong Studio, xuất checkpoint sang GGUF, safetensors hoặc LoRA. Docs ghi các bản export dùng được với Unsloth, llama.cpp, Ollama, vLLM, LM Studio.
2. Nạp lại bản export trong Unsloth và chat thử.
3. Phục vụ cho thiết bị khác: qua LAN (`-H 0.0.0.0`) hoặc HTTPS qua Cloudflare:

   ```bash
   unsloth studio --secure
   ```

   Qua tunnel Cloudflare, docs lưu ý server-sent events không đi qua được, nên đặt `stream: false` khi gọi API.

::: tip Kiến thức nền
Khi export sang GGUF cần chọn mức quantization: xem lại [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa). Merge LoRA vào model gốc: xem [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora).
:::

**Trang nội bộ:** [Export & deploy](/export-deploy), [Ứng dụng RAG](/ung-dung-rag).

**Docs Unsloth gốc:** [Model Export](https://unsloth.ai/docs/new/studio/export), [Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf), [Inference & Deployment](https://unsloth.ai/docs/basics/inference-and-deployment).

**Dấu hiệu hoàn thành:** có file GGUF hoặc safetensors trên đĩa; file đó chạy được trong Unsloth hoặc một công cụ khác như llama.cpp, Ollama.

**Nguồn:** https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/basics/api, https://github.com/unslothai/unsloth

## Mở rộng: Reinforcement Learning

**Mục tiêu:** sau khi fine-tune có giám sát thành thạo, thử các phương pháp RL mà docs liệt kê: GRPO, DPO, cùng FP8 RL và vision RL.

**Việc làm:** đọc lại [RL & preference](/kien-thuc-nen/rl-va-preference), sau đó chạy một notebook RL từ bảng notebook miễn phí trên GitHub (ví dụ "Qwen3: Advanced GRPO", "gpt-oss (20B): GRPO").

**Trang nội bộ:** [Reinforcement Learning](/reinforcement-learning).

**Docs Unsloth gốc:** [Reinforcement Learning (RL) Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).

**Dấu hiệu hoàn thành:** chạy xong một notebook GRPO và giải thích được reward (phần thưởng) được tính thế nào trong notebook đó.

**Nguồn:** https://unsloth.ai/docs, https://github.com/unslothai/unsloth
