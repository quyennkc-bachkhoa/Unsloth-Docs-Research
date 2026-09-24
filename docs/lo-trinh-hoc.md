---
title: Lộ trình học
description: Lộ trình 5 bước từ đọc kiến thức nền, cài Unsloth, chạy model và API đến fine-tune và export.
---

# Lộ trình học

## Toàn cảnh lộ trình

Trang này chỉ cho bạn học Unsloth theo thứ tự nào, từ đọc lý thuyết đến khi có một model tự train. Lộ trình có 5 bước, cộng một bước mở rộng về RL (Reinforcement Learning, học tăng cường).

Mỗi bước có ghi trang [Kiến thức nền LLM](/kien-thuc-nen/) nên đọc **trước** khi bắt tay làm:

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

Workflow của Studio trong docs cũng đi đúng thứ tự này:

1. Khởi chạy Unsloth.
2. Nạp model.
3. Nhập dữ liệu.
4. Làm sạch dữ liệu bằng Data Recipes.
5. Train.
6. Chat với model đã train để so với model gốc.
7. Lưu hoặc export.

**Nguồn:** https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners

## Bước 1: Đọc kiến thức nền

**Mục tiêu:** trước khi tải bất cứ thứ gì, bạn hiểu đủ để chọn đúng model và đúng mức quantization (lượng tử hóa) cho máy của mình.

**Việc làm:**

1. Đọc **Nhóm A** (Cấu tạo mô hình): [Token & context](/kien-thuc-nen/token-va-context) → [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh) → [Kiến trúc Transformer](/kien-thuc-nen/kien-truc-transformer) → [Dense & MoE](/kien-thuc-nen/dense-va-moe).
2. Đọc **Nhóm B** (Số học & bộ nhớ): [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) → [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).
3. Đọc [Tổng quan kiến trúc](/tong-quan) để chọn giữa Desktop, Studio và Core.

**Trang nội bộ:** [Kiến thức nền LLM](/kien-thuc-nen/), [Tổng quan kiến trúc](/tong-quan), [Thuật ngữ](/thuat-ngu).

**Docs Unsloth gốc:** [Fine-tuning for Beginners](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners), [Unsloth Requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements).

**Dấu hiệu hoàn thành:**

- Bạn đọc được tên model mẫu trong docs, `unsloth/gemma-4-26B-A4B-it-GGUF` biến thể `UD-Q4_K_XL`, và tự giải thích từng phần: "26B", "A4B", "it", "GGUF", "Q4".
- Bạn biết máy mình có bao nhiêu VRAM (bộ nhớ card đồ họa) và RAM.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-for-beginners, https://unsloth.ai/docs/basics/api

## Bước 2: Cài đặt Unsloth

**Mục tiêu:** có một bản Unsloth chạy được trên máy, hoặc trên Colab.

**Việc làm:** chọn một trong ba cách docs đưa ra. Nếu máy không có GPU, xem dòng cuối.

- **Desktop (dễ nhất):** tải app tại https://unsloth.ai/download, cài, rồi mở app.
- **Studio thủ công:** chạy script cài, rồi khởi động Studio.

  macOS, Linux, WSL:

  ```bash
  curl -fsSL https://unsloth.ai/install.sh | sh
  ```

  Windows PowerShell:

  ```bash
  irm https://unsloth.ai/install.ps1 | iex
  ```

  Khởi động bằng lệnh dưới đây. Đây là lệnh trong trang cài đặt; README ghi lệnh khác, xem hộp bên dưới.

  ```bash
  unsloth studio -H 0.0.0.0 -p 8888
  ```

  Theo trang Get started, bạn mở `http://127.0.0.1:8888` trên trình duyệt rồi tạo mật khẩu. Muốn cập nhật thì chạy lại lệnh cài.
- **Core (code):** cài gói Python theo hướng dẫn uv/pip của docs.
- **Không có GPU:** dùng notebook Colab miễn phí của Studio. Notebook này chạy trên GPU T4; docs ghi nó chạy và train được hầu hết model tới 22B tham số.

::: danger Mở Studio ra mạng
`-H 0.0.0.0` cho phép thiết bị khác trong mạng truy cập Unsloth. Nếu server-side tools (chạy code, web search) đang bật, ai có mật khẩu hoặc API key đều chạy được code trên máy bạn.
:::

::: warning Docs chưa thống nhất
Trang cài đặt của docs (Nguồn A) và README trên GitHub (Nguồn B) ghi khác nhau về cách khởi động và bảo mật Studio:

| Thông số | Nguồn A: [get-started/install](https://unsloth.ai/docs/get-started/install), [new/studio](https://unsloth.ai/docs/new/studio), [new/studio/start](https://unsloth.ai/docs/new/studio/start), [basics/api](https://unsloth.ai/docs/basics/api) | Nguồn B: [GitHub README](https://github.com/unslothai/unsloth) |
| --- | --- | --- |
| Lệnh khởi động Studio | `unsloth studio -H 0.0.0.0 -p 8888` | `unsloth studio` (mục Launch); `unsloth studio -p 8888` (mục cài bản developer) |
| Địa chỉ bind | `-H 0.0.0.0` nằm ngay trong lệnh khởi động cơ bản | `-H 0.0.0.0` xếp vào mục "Remote HTTPS & LAN Access"; muốn không ra mạng thì dùng `-H 127.0.0.1` |
| Tạo mật khẩu | Mở `http://127.0.0.1:8888` trên trình duyệt, tạo mật khẩu mới tại đó ([new/studio/start](https://unsloth.ai/docs/new/studio/start)) | Khi mở ra mạng (`--secure`, `--cloudflare`, `-H` không phải loopback), terminal hỏi một lần mật khẩu admin mới |
| Server-side tools mặc định | Với `unsloth run`: bật khi bind `127.0.0.1`, **tắt** khi bind `0.0.0.0` ([basics/api](https://unsloth.ai/docs/basics/api)) | "Server-side tools are **on** by default - so be careful!"; dùng `--disable-tools` khi mở Unsloth ra ngoài |
:::

**Trang nội bộ:** [Cài đặt & phần cứng](/cai-dat).

**Docs Unsloth gốc:** [Unsloth Installation](https://unsloth.ai/docs/get-started/install), [Studio Install](https://unsloth.ai/docs/new/studio/install), [uv, pip install & venv](https://unsloth.ai/docs/get-started/install/pip-install).

**Dấu hiệu hoàn thành:** một trong ba điều sau xảy ra:

- App Desktop mở được.
- Trình duyệt vào được Studio sau khi đăng nhập.
- `from unsloth import FastLanguageModel` chạy không lỗi.

**Nguồn:** https://unsloth.ai/docs/get-started/install, https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/basics/api, https://github.com/unslothai/unsloth

## Bước 3: Chạy model và API

**Đọc trước:** **Nhóm D** [Suy luận & sampling](/kien-thuc-nen/suy-luan-va-sampling). Trang này giải thích KV cache, temperature, top-p, top-k, min-p, chat template và tool calling.

**Mục tiêu:** chat được với một model chạy trên máy bạn, và gọi được model đó từ code qua API.

**Việc làm:**

1. Mở dropdown **Select model**, tìm model, rồi chọn biến thể quantization vừa với RAM và VRAM của bạn. Docs dùng ví dụ `unsloth/gemma-4-26B-A4B-it-GGUF` với biến thể khuyến nghị `UD-Q4_K_XL`.
2. Gửi một tin nhắn thử. Thử đổi temperature, top-p, top-k và system prompt để xem câu trả lời thay đổi ra sao.
3. Tạo API key: avatar Unsloth → **Settings** → **API** → đặt tên → **Create**. Chép key `sk-unsloth-…` ngay, vì key chỉ hiện một lần.
4. Cách khác: nạp model từ terminal. Lệnh sẽ in ra URL endpoint và API key:

   ```bash
   unsloth run --model unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL
   ```

5. Gọi `GET /v1/models` để lấy ID model. Sau đó trỏ OpenAI SDK (`/v1/chat/completions`) hoặc Anthropic SDK (`/v1/messages`) vào base URL của Unsloth.

   ```bash
   curl http://localhost:8888/v1/models \
     -H "Authorization: Bearer sk-unsloth-xxxxxxxxxxxx"
   ```

6. (Tùy chọn) Nối agent: `unsloth start claude`.

::: warning Docs chưa thống nhất
Cổng trong base URL của API được ghi khác nhau:

- "typically `http://localhost:8000` or `http://localhost:8888`": [basics/api](https://unsloth.ai/docs/basics/api)
- Ví dụ `curl` dùng `http://localhost:8888`: [basics/api](https://unsloth.ai/docs/basics/api)
- `unsloth run` chạy trên "default port" nhưng không ghi số cổng, chỉ in URL ra console: [basics/api](https://unsloth.ai/docs/basics/api)
- Docker map `-p 8000:8000 -p 8888:8888` và gọi cổng 8888 là của JupyterLab: [GitHub README](https://github.com/unslothai/unsloth)

Cách chắc chắn nhất: base URL thực tế của máy bạn hiển thị ở đầu trang API monitor ([basics/api](https://unsloth.ai/docs/basics/api)).
:::

**Trang nội bộ:** [Inference & API](/inference), [Model catalog](/model-catalog).

**Docs Unsloth gốc:** [Unsloth API](https://unsloth.ai/docs/basics/api), [Studio Chat](https://unsloth.ai/docs/new/studio/chat), [Unsloth Start](https://unsloth.ai/docs/integrations/unsloth-start).

**Dấu hiệu hoàn thành:**

- Studio báo model đã nạp xong và trả lời tin nhắn thử.
- `GET /v1/models` trả về JSON dạng `{"data": [{"id": "gemma-4-26B-A4B-it-GGUF", ...}]}`.
- Request gửi từ code hiện trong API monitor.

Nếu gặp `401 Unauthorized`, bạn đang thiếu header hoặc dùng sai key.

**Nguồn:** https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs, https://github.com/unslothai/unsloth

## Bước 4: Chuẩn bị dữ liệu và fine-tune

**Đọc trước:** **Nhóm C**, gồm ba trang:

- [Quá trình huấn luyện](/kien-thuc-nen/qua-trinh-huan-luyen): loss, learning rate, epoch, batch, gradient accumulation, overfitting.
- [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora): rank, alpha, target modules.
- [RL & preference](/kien-thuc-nen/rl-va-preference): SFT khác RL thế nào.

**Mục tiêu:** hoàn thành lượt fine-tune đầu tiên và chat được với model vừa train.

**Việc làm (theo Studio):**

1. **Model & phương pháp:** chọn loại model (Text, Vision, Audio, Embeddings) và phương pháp QLoRA, LoRA hoặc Full Fine-tuning. Lần đầu nên chọn QLoRA vì tốn VRAM ít nhất. Không chọn model GGUF, vì GGUF chỉ dùng cho inference. Model gated như Llama, Gemma cần Hugging Face token.
2. **Dữ liệu:** lấy từ Hugging Face Hub, hoặc tải file local (`PDF`, `DOCX`, `JSONL`, `JSON`, `CSV`, `Parquet`). Chọn format `auto`, `alpaca`, `chatml` hoặc `sharegpt`. Đặt eval split để có biểu đồ Eval Loss. Nếu bạn chỉ có tài liệu thô, dùng Data Recipes để sinh dataset.
3. **Hyperparameter:** lần đầu cứ giữ mặc định: learning rate `2e-4`, context length `2048`, rank `16`, alpha `32`, epochs `3`, batch size `4`, gradient accumulation `8`. Dùng "Dataset slice" để thử nhanh trên vài dòng.
4. **Train:** bấm **Start Training**. Theo dõi loss, learning rate, gradient norm và GPU monitor. Lưu config ra YAML để lần sau chạy lại.

::: warning Docs chưa thống nhất
Các trang docs liệt kê khác nhau về định dạng file và phần cứng hỗ trợ:

| Thông số | Giá trị và nguồn |
| --- | --- |
| Định dạng file dữ liệu tải lên | `PDF`, `DOCX`, `JSONL`, `JSON`, `CSV`, `Parquet` ([new/studio/start](https://unsloth.ai/docs/new/studio/start)); PDF, CSV, JSON, DOCX, TXT ([new/studio](https://unsloth.ai/docs/new/studio), Auto-create datasets); PDF, CSV hoặc JSONL ([new/studio](https://unsloth.ai/docs/new/studio), Workflow); PDF, CSV hoặc JSON ([desktop](https://unsloth.ai/docs/desktop)) |
| Phần cứng train được | "start training instantly on NVIDIA" ([new/studio](https://unsloth.ai/docs/new/studio), No-code training); "MacOS: Training, MLX and GGUF inference all work" ([new/studio](https://unsloth.ai/docs/new/studio)); "AMD training ... on AMD GPUs across Windows, WSL and Linux" ([GitHub README](https://github.com/unslothai/unsloth)); không nguồn nào ghi rõ về train trên Intel hay CPU |
:::

**[Nhận định]** Chạy một lượt nhỏ bằng Dataset slice trước khi train cả dataset giúp bạn phát hiện lỗi định dạng dữ liệu sớm.

**Trang nội bộ:** [Dữ liệu](/du-lieu), [Fine-tuning](/fine-tuning/).

**Docs Unsloth gốc:** [Get started with Unsloth Studio](https://unsloth.ai/docs/new/studio/start), [Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [Datasets Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide), [LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [Data Recipes](https://unsloth.ai/docs/new/studio/data-recipe), [What Model Should I Use?](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use).

**Dấu hiệu hoàn thành:**

- Thanh tiến độ đạt 100% steps.
- Đường training loss có xu hướng giảm.
- Model đã train xuất hiện ở tab **Fine-tuned** trong Select model, và bạn chat được với nó để so với model gốc.

**Nguồn:** https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/desktop, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners, https://github.com/unslothai/unsloth

## Bước 5: Export và deploy

**Mục tiêu:** đưa model đã train ra khỏi Unsloth, để dùng ở công cụ khác hoặc phục vụ cho thiết bị khác.

**Việc làm:**

1. Mở mục **Export** trong Studio, xuất checkpoint sang GGUF, safetensors hoặc LoRA. Docs ghi các bản export dùng được với Unsloth, llama.cpp, Ollama, vLLM, LM Studio.
2. Nạp lại bản export trong Unsloth và chat thử.
3. Phục vụ cho thiết bị khác, theo một trong hai cách: qua LAN (`-H 0.0.0.0`), hoặc HTTPS qua Cloudflare:

   ```bash
   unsloth studio --secure
   ```

   Docs lưu ý: server-sent events không đi qua được tunnel Cloudflare. Vì vậy bạn đặt `stream: false` khi gọi API.

::: warning Docs chưa thống nhất
Cách bật truy cập LAN được ghi khác nhau:

- Bấm **Start** trên thẻ **LAN access**, hoặc khởi chạy với `-H 0.0.0.0`: [desktop](https://unsloth.ai/docs/desktop)
- `Settings > API keys > LAN access`: [GitHub README](https://github.com/unslothai/unsloth)
- Với `unsloth run`, bind `0.0.0.0` kèm cổng, ví dụ `-H 0.0.0.0 -p 8888`: [basics/api](https://unsloth.ai/docs/basics/api)

Các nguồn cũng ghi khác nhau về việc server-side tools có bật mặc định khi mở ra mạng hay không: xem hộp "Docs chưa thống nhất" ở Bước 2 phía trên.
:::

::: tip Kiến thức nền
Khi export sang GGUF cần chọn mức quantization: xem lại [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa). Merge LoRA vào model gốc: xem [LoRA & QLoRA](/kien-thuc-nen/lora-va-qlora).
:::

**Trang nội bộ:** [Export & deploy](/export-deploy), [Ứng dụng RAG](/ung-dung-rag).

**Docs Unsloth gốc:** [Model Export](https://unsloth.ai/docs/new/studio/export), [Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf), [Inference & Deployment](https://unsloth.ai/docs/basics/inference-and-deployment).

**Dấu hiệu hoàn thành:** có file GGUF hoặc safetensors trên đĩa, và file đó chạy được trong Unsloth hoặc một công cụ khác như llama.cpp, Ollama.

**Nguồn:** https://unsloth.ai/docs/new/studio, https://unsloth.ai/docs/new/studio/start, https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/desktop, https://github.com/unslothai/unsloth

## Mở rộng: Reinforcement Learning

**Mục tiêu:** khi đã thành thạo fine-tune có giám sát, bạn thử các phương pháp RL mà docs liệt kê: GRPO, DPO, cùng FP8 RL và vision RL.

**Việc làm:**

1. Đọc lại [RL & preference](/kien-thuc-nen/rl-va-preference).
2. Chạy một notebook RL từ bảng notebook miễn phí trên GitHub, ví dụ "Qwen3: Advanced GRPO" hoặc "gpt-oss (20B): GRPO".

**Trang nội bộ:** [Reinforcement Learning](/reinforcement-learning).

**Docs Unsloth gốc:** [Reinforcement Learning (RL) Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).

**Dấu hiệu hoàn thành:** bạn chạy xong một notebook GRPO và giải thích được reward (phần thưởng) được tính thế nào trong notebook đó.

**Nguồn:** https://unsloth.ai/docs, https://github.com/unslothai/unsloth
