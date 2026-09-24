---
layout: home
title: "Unsloth Research"
description: "Nghiên cứu Unsloth bằng tiếng Việt: chạy, fine-tune và triển khai LLM trên phần cứng của bạn."

hero:
  name: "Unsloth Research"
  text: "Chạy và huấn luyện LLM ngay trên máy của bạn"
  tagline: "Unsloth là framework mã nguồn mở để chạy (inference) và huấn luyện (training) LLM trên phần cứng local, có giao diện Desktop, Studio và thư viện Python Core."
  actions:
    - theme: brand
      text: "Bắt đầu với Unsloth"
      link: /tong-quan
    - theme: alt
      text: "Kiến thức nền LLM"
      link: /kien-thuc-nen/

features:
  - title: "Tổng quan kiến trúc"
    details: "Ba dạng sản phẩm Desktop, Studio, Core; CLI unsloth và API tương thích OpenAI/Anthropic."
    link: /tong-quan
    linkText: "Xem kiến trúc"
  - title: "Lộ trình học 5 bước"
    details: "Từ đọc kiến thức nền, cài đặt, chạy model đến fine-tune và export."
    link: /lo-trinh-hoc
    linkText: "Xem lộ trình"
  - title: "Cài đặt & phần cứng"
    details: "App Desktop cho macOS, Windows, Linux; Studio cài bằng script; Core cài bằng uv/pip."
    link: /cai-dat
    linkText: "Cài đặt"
  - title: "Inference & API"
    details: "Chạy GGUF, safetensors; phục vụ model qua endpoint /v1/chat/completions và /v1/messages."
    link: /inference
    linkText: "Chạy model"
  - title: "Fine-tuning"
    details: "QLoRA, LoRA, full fine-tuning; docs ghi nhận nhanh hơn 2 lần và ít hơn 70% VRAM."
    link: /fine-tuning
    linkText: "Huấn luyện"
  - title: "Export & deploy"
    details: "Xuất model sang GGUF, safetensors hoặc LoRA để dùng với llama.cpp, Ollama, vLLM."
    link: /export-deploy
    linkText: "Triển khai"
---

## Unsloth là gì

Unsloth là framework mã nguồn mở (open-source) để chạy và huấn luyện LLM (Large Language Model, mô hình ngôn ngữ lớn) trên phần cứng của chính bạn, thông qua một giao diện người dùng cũng mã nguồn mở. Docs chia Unsloth thành ba cách dùng:

| Dạng | Mô tả ngắn theo docs |
| --- | --- |
| **Unsloth Desktop** | App native cho macOS, Windows, Linux (bản Beta), xây trên Tauri |
| **Unsloth Studio** | Web UI chạy trên trình duyệt, không cần viết code (bản Beta) |
| **Unsloth Core** | Gói Python gốc, dùng code để train và inference |

Ngoài LLM văn bản, Unsloth còn chạy và train model diffusion (sinh ảnh/video), audio (TTS, nhận dạng giọng nói), embedding và vision.

::: tip Kiến thức nền
Chưa phân biệt được LLM, model embedding hay multimodal? Xem [Phân loại mô hình](/kien-thuc-nen/phan-loai-mo-hinh).
:::

Chi tiết từng dạng: [Tổng quan kiến trúc](/tong-quan).

**Nguồn:** https://unsloth.ai/docs, https://unsloth.ai/docs/desktop, https://unsloth.ai/docs/get-started/install, https://github.com/unslothai/unsloth

## Giải quyết vấn đề gì

Theo docs, Unsloth gom trọn chuỗi "chạy model → chuẩn bị dữ liệu → train → deploy" vào một công cụ chạy local:

| Vấn đề | Unsloth cung cấp (theo docs) |
| --- | --- |
| Train LLM tốn VRAM (bộ nhớ card đồ họa) và chậm | Trang docs chính ghi fine-tune nhanh hơn 2 lần, ít hơn 70% VRAM, "không giảm độ chính xác"; README ghi mức khác theo từng model (xem hộp bên dưới) |
| Không muốn viết code train | Studio/Desktop: tải lên tài liệu (trang Desktop ghi PDF, CSV, JSON; các trang khác liệt kê định dạng khác, xem [Tổng quan](/tong-quan)) rồi bấm train (LoRA, full fine-tuning, pretraining) |
| Không có dataset sẵn | Data Recipes: biến PDF, CSV, DOCX thành dataset qua workflow dạng graph-node |
| Muốn dùng model local trong agent/ứng dụng | API tương thích OpenAI và Anthropic; lệnh `unsloth start claude` nối Claude Code, Codex với model local |
| Muốn mang model đã train đi chỗ khác | Export sang GGUF, safetensors, NVFP4, FP8... cho llama.cpp, Ollama, vLLM, LM Studio |
| Lo ngại dữ liệu rời máy | Không thu thập telemetry; có thể chạy hoàn toàn offline |

Docs còn nêu Unsloth hỗ trợ inference và training cho hơn 500 model, và đội Unsloth từng phối hợp sửa lỗi nghiêm trọng cho các họ model như gpt-oss, Qwen3, Llama 4, Gemma, Phi-4.

::: warning Docs chưa thống nhất
Mức tăng tốc và mức tiết kiệm VRAM khi train ghi khác nhau, và không nguồn nào nêu điều kiện đo:

| Thông số | Nguồn A: [docs](https://unsloth.ai/docs), [new/studio](https://unsloth.ai/docs/new/studio), [desktop](https://unsloth.ai/docs/desktop) | Nguồn B: bảng Free Notebooks, [GitHub README](https://github.com/unslothai/unsloth) |
| --- | --- | --- |
| Tốc độ train | 2x nhanh hơn | 1.5x (Gemma 4 E2B, Qwen3.5 4B, Orpheus-TTS 3B) đến 2x (gpt-oss 20B, Llama 3.1 8B...) |
| Bộ nhớ | Ít hơn 70% VRAM | Ít hơn 20% (embeddinggemma 300M) đến 80% (gpt-oss 20B GRPO) |
:::

::: tip Kiến thức nền
VRAM và cách ước lượng bộ nhớ cho một model: xem [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho). Định dạng GGUF, FP8, NVFP4: xem [Độ chính xác & lượng tử hóa](/kien-thuc-nen/do-chinh-xac-va-luong-tu-hoa).
:::

**Nguồn:** https://unsloth.ai/docs, https://unsloth.ai/docs/desktop, https://unsloth.ai/docs/new/studio, https://github.com/unslothai/unsloth

## Dành cho ai

- **Người muốn chạy thử LLM local không cần code:** docs khuyến nghị cài app Desktop, chọn model và mức quantization (lượng tử hóa) vừa với máy rồi chat ngay.
- **Lập trình viên muốn tích hợp model local:** dùng endpoint `/v1/chat/completions` (OpenAI SDK) hoặc `/v1/messages` (Anthropic SDK), hoặc nối agent như Claude Code, Codex, OpenCode.
- **Người mới fine-tune:** docs có mục "Fine-tuning for Beginners" trả lời các câu hỏi như chọn model instruct hay base, dataset cần bao lớn, GPU có đủ VRAM không.
- **Người viết code train:** dùng Unsloth Core (gói Python) hoặc notebook Colab miễn phí.

Nền tảng theo docs: macOS, Windows, Linux, WSL; GPU NVIDIA, AMD, Intel, Mac và CPU. Docs lưu ý phần cứng cũ có thể không được hỗ trợ tốt. Các trang ghi khác nhau về việc train được trên phần cứng nào (xem hộp "Docs chưa thống nhất" ở [Tổng quan kiến trúc](/tong-quan)).

**[Nhận định]** Website này viết cho người đã biết Python/FastAPI nhưng mới với AI: phần "Unsloth" tập trung vào thao tác, còn khái niệm (token, LoRA, quantization...) được tách sang route Kiến thức nền.

**Nguồn:** https://unsloth.ai/docs/desktop, https://unsloth.ai/docs/basics/api, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners, https://github.com/unslothai/unsloth

## Cách dùng website này

Website có hai route, chuyển qua thanh điều hướng trên cùng:

| Route | Nội dung | Khi nào đọc |
| --- | --- | --- |
| **Unsloth** (`/`) | Kiến trúc, cài đặt, inference & API, model catalog, fine-tuning, RL, dữ liệu, export, ứng dụng RAG, thuật ngữ, nguồn | Khi thao tác với Unsloth |
| **Kiến thức nền LLM** (`/kien-thuc-nen/`) | Token, kiến trúc Transformer, dense/MoE, bộ nhớ, lượng tử hóa, quá trình huấn luyện, LoRA, RL, sampling | Trước khi cài đặt, trước khi chạy inference, trước khi fine-tune |

Thứ tự đọc gợi ý:

1. [Tổng quan kiến trúc](/tong-quan): nắm ba dạng Desktop, Studio, Core và cách chúng nối với CLI, API.
2. [Lộ trình học](/lo-trinh-hoc): năm bước, mỗi bước chỉ rõ trang Kiến thức nền cần đọc trước.
3. Các trang chi tiết theo thanh bên trái.

Mọi thông số trên website lấy từ docs chính thức của Unsloth; mỗi mục có dòng **Nguồn**. Câu gắn **[Nhận định]** là ý kiến người viết. Danh sách đầy đủ: [Nguồn tham khảo](/nguon).

**Nguồn:** https://unsloth.ai/docs, https://github.com/unslothai/unsloth
