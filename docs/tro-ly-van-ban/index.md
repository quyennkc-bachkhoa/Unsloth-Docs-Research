---
title: Tổng quan và kết luận nhanh
description: "Dựng trợ lý văn bản hành chính tiếng Việt trên máy chủ 2×L40S (96 GB), chọn model tiếng Việt, fine-tune bằng Unsloth cho văn bản hành chính nhà nước và host local bằng vLLM, llama.cpp, Ollama."
---

# Tổng quan và kết luận nhanh

Phần này áp dụng những gì website đã nghiên cứu về Unsloth để dựng một trợ lý AI soạn và tra cứu văn bản hành chính, chạy trên máy chủ của đơn vị. Trang này nêu yêu cầu, trả lời gọn từng câu hỏi, rồi dẫn sang trang chi tiết.

::: tip Tóm tắt
- **Dùng khi:** cần một câu trả lời tổng thể cho việc dựng trợ lý AI tiếng Việt chạy local trên máy chủ 2×L40S.
- **Kết quả:** câu trả lời ngắn cho 4 câu hỏi của dự án, sơ đồ toàn hệ thống, và đường đọc tiếp cho từng phần.
- **Nên biết trước:** [Tổng quan kiến trúc](/tong-quan) (Unsloth gồm những gì), [Tham số & bộ nhớ](/kien-thuc-nen/tham-so-va-bo-nho) (vì sao model 27B cần khoảng 55 GB ở BF16).
:::

::: warning Dữ liệu có hạn dùng
Model, bảng xếp hạng và văn bản pháp luật trong phần này được tra ngày **2026-09-25**. Các thông tin này thay đổi nhanh.
:::

## Yêu cầu

**Phần cứng:** máy chủ 2 card NVIDIA L40S, mỗi card 48 GB, tổng **96 GB VRAM**.

Bốn câu hỏi cần trả lời:

1. Hiện tại model nào tốt nhất, phù hợp nhất cho tiếng Việt?
2. Dùng Unsloth fine-tune như thế nào, và fine-tune để làm gì?
3. Áp dụng cho **văn bản hành chính nhà nước**.
4. Có mấy cách host model local: vLLM, llama.cpp, Ollama?

**Nguồn:** thông số L40S: https://www.nvidia.com/en-us/data-center/l40s/

## Các trang trong phần này

| Trang | Giúp bạn làm gì | Đọc khi nào |
| --- | --- | --- |
| [Chọn model cho tiếng Việt](/tro-ly-van-ban/chon-model) | So bảng điểm tiếng Việt, chọn ứng viên vừa 96 GB | Trước tiên, trước khi tải model |
| [Fine-tune bằng Unsloth](/tro-ly-van-ban/fine-tune) | Hiểu fine-tune để làm gì, chia việc với RAG, cấu hình 1 hoặc 2 card | Khi đã chốt model |
| [Tìm và chuẩn bị dữ liệu](/tro-ly-van-ban/van-ban-hanh-chinh) | Tìm dataset trên Hugging Face, kiểm tra trước khi dùng, lưu ý pháp lý | Trước khi làm dataset |
| [Host model trên máy chủ](/tro-ly-van-ban/hosting) | So sánh vLLM, llama.cpp, Ollama, chia 2 GPU, số người dùng phục vụ được | Khi đem model ra cho đơn vị dùng |

## Trả lời ngắn

Mỗi câu một đoạn. Chi tiết và nguồn nằm ở trang tương ứng.

### 1. Model nào hợp tiếng Việt

**Qwen3.6-27B hoặc Qwen3.8-27B, và Gemma 4 31B.** Đây là hai lựa chọn chính, cả hai đều Apache 2.0.

- Trên bảng điểm tiếng Việt SEA-HELM (cập nhật 18/09/2026), Gemma 4 31B được 77.09 và Qwen3.6-27B được 76.12. Chênh lệch nằm trong sai số, nên coi như ngang nhau.
- Các model "chuyên tiếng Việt" đời cũ (Vistral, PhoGPT, SeaLLMs) đã bị vượt xa.
- Các model tiếng Việt đứng đầu VMLU đều không công bố trọng số, nên không tải về được.

**[Nhận định]** Hãy chạy thử cả hai trên văn bản thật của đơn vị. Không benchmark nào đo văn bản hành chính. → [Chọn model cho tiếng Việt](/tro-ly-van-ban/chon-model)

### 2. Fine-tune để làm gì, làm thế nào

**Fine-tune dạy model viết đúng văn phong và thể thức của đơn vị.** Nội dung văn bản pháp luật thì để RAG tra cứu, vì văn bản thay đổi hiệu lực liên tục. Docs Unsloth cũng khuyên kết hợp hai cách này.

Làm theo 6 bước: chọn model → dữ liệu → cấu hình → train → đánh giá → export.

- **Gemma 4 31B:** QLoRA vừa **một** card L40S (22 GB).
- **Qwen 27B:** LoRA 16-bit cần khoảng 56 GB, chia 2 card bằng `device_map = "balanced"`.

→ [Fine-tune bằng Unsloth](/tro-ly-van-ban/fine-tune)

### 3. Dữ liệu lấy ở đâu

- **Văn bản pháp luật:** Hugging Face có nhiều dataset. Lọc tiếng Việt và gõ `legal` ra 50 kết quả (26/09/2026).
- **Công văn:** gõ `công văn` hoặc `cong van` ra 0 kết quả.

→ [Tìm và chuẩn bị dữ liệu](/tro-ly-van-ban/van-ban-hanh-chinh)

### 4. Mấy cách host local

**Ba engine chính, cả ba có API tương thích OpenAI:**

| Engine | Vai trò |
| --- | --- |
| **vLLM** | Production cho nhiều người dùng. Chạy FP8 trên L40S |
| **llama.cpp** (llama-server) | Thử nghiệm với file GGUF, chỉnh chia GPU chi tiết |
| **Ollama** | Dễ dùng nhất. Mặc định chỉ xử lý 1 request mỗi lúc |

- L40S **không có NVLink**, nên nếu model vừa một card thì chạy mỗi card một bản sẽ đơn giản hơn chia một model ra hai card.
- Docs Unsloth có hướng dẫn cho cả ba engine.

→ [Host model trên máy chủ](/tro-ly-van-ban/hosting), [Chạy model đã xuất](/export-deploy/chay-model)

**Nguồn:** https://leaderboard.sea-lion.ai/detailed/VI, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me, https://unsloth.ai/docs/models/gemma-4/train, https://unsloth.ai/docs/models/qwen3.5/fine-tune, https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth, https://vanban.chinhphu.vn/default.aspx?pageid=27160&docid=199378, https://www.nvidia.com/en-us/data-center/l40s/, https://unsloth.ai/docs/basics/inference-and-deployment

## Toàn bộ hệ thống

**[Nhận định]** Một cách ghép các phần lại với nhau.

<div class="dg">
  <div class="dg-title">Từ dữ liệu tới người dùng trên máy 2×L40S</div>
  <div class="dg-flow">
    <div class="dg-node is-ghost">Văn bản đã ban hành của đơn vị</div>
    <div class="dg-node"><div>Dataset hội thoại</div><small>lọc mật, ẩn danh</small></div>
    <div class="dg-node is-main"><div>Unsloth fine-tune Qwen 27B hoặc Gemma 4 31B</div><small>LoRA / QLoRA</small></div>
    <div class="dg-node"><div>vLLM (FP8)</div><small>merged 16-bit</small></div>
    <div class="dg-node is-end">Cán bộ soạn thảo, tra cứu</div>
  </div>
  <div class="dg-cap">Kho văn bản pháp luật còn hiệu lực đi qua RAG và cấp căn cứ cho model lúc trả lời. Không đưa kho này vào fine-tune.</div>
</div>

Thứ tự triển khai gợi ý:

1. Chạy thử 2 model ứng viên với prompt tốt và RAG, chưa fine-tune.
2. Làm bộ đánh giá nội bộ (vài trăm văn bản, có cán bộ chấm).
3. Làm dataset, fine-tune model thắng ở bước 1.
4. So model đã fine-tune với model gốc trên bộ đánh giá.
5. Đưa lên vLLM, mở trong mạng nội bộ.

**Nguồn:** https://unsloth.ai/docs/get-started/fine-tuning-llms-guide, https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide

## Đọc tiếp

- [Chọn model cho tiếng Việt](/tro-ly-van-ban/chon-model) — trang kế tiếp: bước đầu tiên là chọn model nền.
- [Lộ trình học](/lo-trinh-hoc) — nếu chưa quen Unsloth, đi theo 5 bước từ đầu.
- [Kiến thức nền LLM](/kien-thuc-nen/) — các khái niệm LoRA, lượng tử hóa, KV cache dùng trong phần này.
